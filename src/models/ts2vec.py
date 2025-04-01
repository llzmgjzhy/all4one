import torch
import torch.nn.functional as F
import numpy as np
from models.model import TSEncoder
from utils import (
    split_with_nan,
    centerize_vary_length_series,
    take_per_row,
    torch_pad_nan,
)
from models.loss import hierarchical_contrastive_loss


class TS2Vec:
    """The TS2Vec model"""

    def __init__(
        self,
        config,
        input_dims,
        output_dims=320,
        hidden_dims=64,
        depth=10,
        device="cuda",
        max_train_length=None,
        temporal_unit=0,
        after_iter_callback=None,
        after_epoch_callback=None,
    ):
        """Initialize the TS2Vec model.

        Args:
                input_dims (int): The input dimension. For a univariate time series, this should be 1.
                output_dims (int): The representation dimension.
                hidden_dims (int): The hidden dimension of the encoder.
                depth (int): The number of hidden residual blocks in the encoder.
                device (int): The gpu used for training and inference.
                max_train_length (int): The maximum allowed sequence length for training. For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length>.
                temporal_unit (int): The minimum unit to perform temporal contrast. When training on a very long sequence, this param helps to reduce the cost of time and memory.
                after_iter_callback (Union[Callable, NoneType]): A callback function that would be called after each iteration.
                after_epoch_callback (Union[Callable, NoneType]): A callback function that would be called after each epoch.
        """

        super().__init__()
        self.device = device
        self.max_train_length = max_train_length
        self.temporal_unit = temporal_unit

        self._net = TSEncoder(
            input_dims=input_dims,
            output_dims=output_dims,
            hidden_dims=hidden_dims,
            depth=depth,
        ).to(self.device)
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)

        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback

        self.n_epochs = 0
        self.n_iters = 0

    def fit(self, train_data, n_epochs=None, verbose=False):
        """Training the TS2Vec model.

        Args:
          train_data : The training data. It should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
          n_epochs (Union[int, NoneType]): The number of epochs.
          verbose (bool): Whether to print the training information.
        """

        assert train_data.ndim == 3

        if self.max_train_length is not None:
            sections = train_data.shape[1] // self.max_train_length
            if sections >= 2:
                train_data = np.concatenate(
                    split_with_nan(train_data, sections, axis=1), axis=0
                )

        temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
        if temporal_missing[0] or temporal_missing[-1]:
            train_data = centerize_vary_length_series(train_data)

        train_data = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]

        ts_l = train_data.shape[1]
        crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l + 1)
        crop_left = np.random.randint(ts_l - crop_l + 1)
        crop_right = crop_left + crop_l
        crop_eleft = np.random.randint(crop_left + 1)
        crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
        crop_offset = np.random.randint(
            low=-crop_eleft, high=ts_l - crop_eright + 1, size=train_data.size(0)
        )

        out1 = self._net(
            take_per_row(train_data, crop_offset + crop_eleft, crop_right - crop_eleft)
        )
        out1 = out1[:, -crop_l:]

        out2 = self._net(
            take_per_row(train_data, crop_offset + crop_left, crop_eright - crop_left)
        )
        out2 = out2[:, :crop_l]

        loss = hierarchical_contrastive_loss(
            out1, out2, temporal_unit=self.temporal_unit
        )

        return loss

    def _eval_with_pooling(self, x, mask, slicing=None, encoding_window=None):
        out = self.net(x.to(self.device, non_blocking=True), mask)
        if encoding_window == "full_series":
            if slicing is not None:
                out = out[:, slicing]
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size=out.size(1),
            ).transpose(1, 2)

        elif isinstance(encoding_window, int):
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size=encoding_window,
                stride=1,
                padding=encoding_window // 2,
            ).transpose(1, 2)
            if encoding_window % 2 == 0:
                out = out[:, :-1]
            if slicing is not None:
                out = out[:, slicing]

        elif encoding_window == "multiscale":
            p = 0
            reprs = []
            while (1 << p) + 1 < out.size(1):
                t_out = F.max_pool1d(
                    out.transpose(1, 2),
                    kernel_size=(1 << (p + 1)) + 1,
                    stride=1,
                    padding=1 << p,
                ).transpose(1, 2)
                if slicing is not None:
                    t_out = t_out[:, slicing]
                reprs.append(t_out)
                p += 1
            out = torch.cat(reprs, dim=-1)

        else:
            if slicing is not None:
                out = out[:, slicing]

        return out.cpu()

    def encode(
        self,
        data,
        mask=None,
        encoding_window=None,
        causal=False,
        sliding_length=None,
        sliding_padding=0,
    ):
        """Compute representations using the model.

        Args:
            data: This should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            mask (str): The mask used by encoder can be specified with this parameter. This can be set to 'binomial', 'continuous', 'all_true', 'all_false' or 'mask_last'.
            encoding_window (Union[str, int]): When this param is specified, the computed representation would the max pooling over this window. This can be set to 'full_series', 'multiscale' or an integer specifying the pooling kernel size.
            causal (bool): When this param is set to True, the future informations would not be encoded into representation of each timestamp.
            sliding_length (Union[int, NoneType]): The length of sliding window. When this param is specified, a sliding inference would be applied on the time series.
            sliding_padding (int): This param specifies the contextual data length used for inference every sliding windows.
        """
        assert self.net is not None, "please train or load a net first"
        assert data.ndim == 3
        n_samples, ts_l, _ = data.shape

        org_training = self.net.training
        self.net.eval()

        output = []

        with torch.no_grad():
            if sliding_length is not None:
                reprs = []
                for i in range(0, ts_l, sliding_length):
                    l = i - sliding_padding
                    r = i + sliding_length + (sliding_padding if not causal else 0)
                    x_sliding = torch_pad_nan(
                        data[:, max(l, 0) : min(r, ts_l)],
                        left=-l if l < 0 else 0,
                        right=r - ts_l if r > ts_l else 0,
                        dim=1,
                    )
                    out = self._eval_with_pooling(
                        x_sliding,
                        mask,
                        slicing=slice(
                            sliding_padding, sliding_padding + sliding_length
                        ),
                        encoding_window=encoding_window,
                    )
                    reprs.append(out)
                out = torch.cat(reprs, dim=1)
                if encoding_window == "full_series":
                    out = F.max_pool1d(
                        out.transpose(1, 2).contiguous(),
                        kernel_size=out.size(1),
                    ).squeeze(1)

            else:
                out = self._eval_with_pooling(
                    data, mask, encoding_window=encoding_window
                )
                if encoding_window == "full_series":
                    out = out.squeeze(1)

        output.append(out)

        output = torch.cat(output, dim=0)

        return output