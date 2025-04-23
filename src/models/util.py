import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0

    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(
            in_channels, out_channels, kernel_size, dilation=dilation
        )
        self.conv2 = SamePadConv(
            out_channels, out_channels, kernel_size, dilation=dilation
        )
        self.projector = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels or final
            else None
        )

    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual


class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        self.net = nn.Sequential(
            *[
                ConvBlock(
                    channels[i - 1] if i > 0 else in_channels,
                    channels[i],
                    kernel_size=kernel_size,
                    dilation=2**i,
                    final=(i == len(channels) - 1),
                )
                for i in range(len(channels))
            ]
        )
        self.net.__class__.__name__ = "DilatedConvEncoder"

    def forward(self, x):
        return self.net(x)


def pad_nan_to_target(array, target_length, axis=0, both_side=False):
    assert array.dtype in [np.float16, np.float32, np.float64]
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    if both_side:
        npad[axis] = (pad_size // 2, pad_size - pad_size // 2)
    else:
        npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode="constant", constant_values=np.nan)


def split_with_nan(x, sections, axis=0):
    assert x.dtype in [np.float16, np.float32, np.float64]
    arrs = np.array_split(x, sections, axis=axis)
    target_length = arrs[0].shape[axis]
    for i in range(len(arrs)):
        arrs[i] = pad_nan_to_target(arrs[i], target_length, axis=axis)
    return arrs


def centerize_vary_length_series(x):
    prefix_zeros = np.argmax(~np.isnan(x).all(axis=-1), axis=1)
    suffix_zeros = np.argmax(~np.isnan(x[:, ::-1]).all(axis=-1), axis=1)
    offset = (prefix_zeros + suffix_zeros) // 2 - prefix_zeros
    rows, column_indices = np.ogrid[: x.shape[0], : x.shape[1]]
    offset[offset < 0] += x.shape[1]
    column_indices = column_indices - offset[:, np.newaxis]
    return x[rows, column_indices]


def take_per_row(A, indx, num_elem):
    all_indx = indx[:, None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:, None], all_indx]


def torch_pad_nan(arr, left=0, right=0, dim=0):
    device = arr.device
    if left > 0:
        padshape = list(arr.shape)
        padshape[dim] = left
        arr = torch.cat(
            (torch.full(padshape, float("nan"), device=device), arr), dim=dim
        )
    if right > 0:
        padshape = list(arr.shape)
        padshape[dim] = right
        arr = torch.cat(
            (arr, torch.full(padshape, float("nan"), device=device)), dim=dim
        )
    return arr


def tensor_line_plots(data, height=256, width=256, flip=False):
    """
    Efficiently generate line graph tensors (supporting GPU and gradient)
    Args:
        data: [B, T, N] input tensor, B is batch size, T is time length, N is number of dims.Input tensor has been normalized.
        flip: whether to flip the y axis
    Returns:
        images: [b, height, width]
    """

    B, seq_len, _ = data.shape
    device = data.device
    y = data.squeeze()  # [B, T, N] -> [B, T] if N =1

    # genarate x axis (Uniformly distributed to the width)
    x = torch.linspace(0, width - 1, seq_len, device=device).expand(
        B, seq_len
    )  # [B, T]

    # initialize image template
    images = torch.zeros(B, height, width, device=device)

    # turn axis into int index
    x_idx = x.round().long().clamp(0, width - 1)  # [B, T]

    y_min = y.min(dim=1, keepdim=True)[0]
    y_max = y.max(dim=1, keepdim=True)[0]
    y_norm = (y - y_min) / (y_max - y_min + 1e-8)
    # if visualization is needed, set flip to True
    if flip:
        y_scaled = (1 - y_norm) * (height - 1)
    else:
        y_scaled = y_norm * (height - 1)
    y_idx = y_scaled.round().long().clamp(0, height - 1)  # [B, T]
    # y_idx = y.round().long().clamp(0, height - 1)  # [B, T]

    # plot scalars
    x_start, x_end = x_idx[:, :-1], x_idx[:, 1:]  # [B, T-1]
    y_start, y_end = y_idx[:, :-1], y_idx[:, 1:]

    # cal the variation of line segments in the x and y directions
    dx = x_end - x_start
    dy = y_end - y_start

    # determine the number of steps for interpolation
    line_lengths = torch.max(torch.abs(dx), torch.abs(dy))
    max_length = line_lengths.max().item()
    num_steps = max_length + 1 if max_length > 0 else 1  # handle zero-length lines

    # generate interpolation param t
    t = torch.linspace(0, 1, steps=num_steps, device=device)  # [num_steps]

    # cal all points on the line segments
    x_points = x_start.unsqueeze(-1) + dx.unsqueeze(-1) * t  # [B, T-1, num_steps]
    y_points = y_start.unsqueeze(-1) + dy.unsqueeze(-1) * t

    # round and clamp the coordinate range
    x_rounded = x_points.round().long().clamp(0, width - 1)
    y_rounded = y_points.round().long().clamp(0, height - 1)

    # generate batch index and flatten coordinates
    batch_indices = (
        torch.arange(B, device=device)[:, None, None]
        .expand(-1, dx.size(1), num_steps)
        .flatten()
    )
    x_flat = x_rounded.flatten()
    y_flat = y_rounded.flatten()

    # set the pixel values to 1.0
    images[batch_indices, y_flat, x_flat] = 1.0

    images = images.unsqueeze(1)  # [B, 1, height, width]
    # images = images.expand(-1, 3, -1, -1)  # [B, 3, height, width]
    return images  # [B, 1, height, width]
