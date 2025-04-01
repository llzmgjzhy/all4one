import numpy as np
import torch
from torch.utils.data import Dataset


class VSBDataset(Dataset):
    def __init__(self, data, indices):
        super(VSBDataset, self).__init__()
        self.data = data
        self.d_piece = data.d_piece
        self.IDs = indices
        self.df = data.feature_df.loc[self.IDs]
        self.labels = self.data.labels.loc[self.IDs]

    def __getitem__(self, idx):
        """
        For a given integer index, returns the corresponding (sequence_length, feat_dim) array
        Args:
            idx: integer index of a sample in dataset
        Returns:
            X: (sequence_length, feat_dim ) tensor corresponding to the sample
            y: (num_labels, ) tensor corresponding to the label
            ID: ID of the sample"""
        X = self.df.loc[self.IDs[idx]].values  # origin shape is [3,800000]

        # transpose work will be done in model module

        # we need to transpose it to [480,5000], cause we set feat_dim=5000
        X = X.reshape(X.shape[0], -1, self.d_piece)
        # X = np.transpose(X, (1, 0, 2)).reshape(-1, self.d_piece)
        X = X.reshape(-1, self.d_piece)

        y = self.labels.loc[self.IDs[idx]].values

        return torch.from_numpy(X), torch.from_numpy(y), self.IDs[idx]

    def __len__(self):
        return len(self.IDs)


def collate_superv(data, max_len=None):
    """Build mini-batch tensors from a list of (X, mask) tuples. Mask input. Create
    Args:
        data: len(batch_size) list of tuples (X, y).
            - X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
            - y: torch tensor of shape (num_labels,) : class indices or numerical targets
                (for classification or regression, respectively). num_labels > 1 for multi-task models
        max_len: global fixed sequence length. Used for architectures requiring fixed length input,
            where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
    Returns:
        X: (batch_size, padded_length, feat_dim) torch tensor of masked features (input)
        targets: (batch_size, padded_length, feat_dim) torch tensor of unmasked features (output)
        target_masks: (batch_size, padded_length, feat_dim) boolean torch tensor
            0 indicates masked values to be predicted, 1 indicates unaffected/"active" feature values
        padding_masks: (batch_size, padded_length) boolean tensor, 1 means keep vector at this position, 0 means padding
    """

    batch_size = len(data)
    features, labels, IDs = zip(*data)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [
        X.shape[0] for X in features
    ]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)
    X = torch.zeros(
        batch_size, max_len, features[0].shape[-1]
    )  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]

    targets = torch.stack(labels, dim=0)  # (batch_size, num_labels)

    padding_masks = padding_mask(
        torch.tensor(lengths, dtype=torch.int16), max_len=max_len
    )  # (batch_size, padded_length) boolean tensor, "1" means keep

    return X, targets, padding_masks, IDs


def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    batch_size = lengths.numel()
    max_len = (
        max_len or lengths.max_val()
    )  # trick works because of overloading of 'or' operator for non-boolean types
    return (
        torch.arange(0, max_len, device=lengths.device)
        .type_as(lengths)
        .repeat(batch_size, 1)
        .lt(lengths.unsqueeze(1))
    )
