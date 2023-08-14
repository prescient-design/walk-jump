from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ._activations import ACTIVATION_STR_TO_TYPE
from ._layers import PositionFeedForward


class MaskedConv1d(nn.Conv1d):
    """A masked 1-dimensional convolution layer.

    Takes the same arguments as torch.nn.Conv1D, except that the padding is set automatically.

         Shape:
            Input: (N, L, in_channels)
            input_mask: (N, L, 1), optional
            Output: (N, L, out_channels)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        """
        :param in_channels: input channels
        :param out_channels: output channels
        :param kernel_size: the kernel width
        :param stride: filter shift
        :param dilation: dilation factor
        :param groups: perform depth-wise convolutions
        :param bias: adds learnable bias to output
        """
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding=padding,
        )

    def forward(self, x, input_mask: Optional[torch.Tensor] = None):
        if input_mask is not None:
            # padding mask
            x.masked_fill_(input_mask[..., None], 0.0)
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class ByteNetBlock(nn.Module):
    """Residual block from ByteNet paper (https://arxiv.org/abs/1610.10099).

    Shape:
       Input: (N, L, d_in)
       input_mask: (N, L, 1), optional
       Output: (N, L, d_out)

    """

    def __init__(
        self, d_in, d_h, d_out, kernel_size, dilation=1, groups=1, activation="silu", rank=None
    ):
        super().__init__()
        self.conv = MaskedConv1d(
            d_h, d_h, kernel_size=kernel_size, dilation=dilation, groups=groups
        )

        self.res_connection = d_in == d_out
        act = ACTIVATION_STR_TO_TYPE[activation]

        layers1 = [
            nn.LayerNorm(d_in),
            act(),
            PositionFeedForward(d_in, d_h, rank=rank),
            nn.LayerNorm(d_h),
            act(),
        ]
        layers2 = [
            nn.LayerNorm(d_h),
            act(),
            PositionFeedForward(d_h, d_out, rank=rank),
        ]
        self.sequence1 = nn.Sequential(*layers1)
        self.sequence2 = nn.Sequential(*layers2)

    def forward(self, x, input_mask=None):
        """
        :param x: (batch, length, in_channels)
        :param input_mask: (batch, length, 1)
        :return: (batch, length, out_channels)
        """
        rep = self.sequence2(self.conv(self.sequence1(x), input_mask=input_mask))
        if self.res_connection:
            return x + rep
        return rep


class ByteNet(nn.Module):

    """Stacked residual blocks from ByteNet paper defined by n_layers

    Shape:
       Input: (N, L,)
       input_mask: (N, L, 1), optional
       Output: (N, L, d)

    """

    def __init__(
        self,
        n_tokens,
        d_model,
        n_layers,
        kernel_size,
        r,
        rank=None,
        dropout=0.0,
        slim=True,
        activation="silu",
        down_embed=True,
    ):
        """
        :param n_tokens: number of tokens in token dictionary
        :param d_model: dimension to use within ByteNet model, //2 every layer
        :param n_layers: number of layers of ByteNet block
        :param kernel_size: the kernel width
        :param r: used to calculate dilation factor
        :param rank: rank of compressed weight matrices
        :param slim: if True, use half as many dimensions in the NLP as in the CNN
        :param activation: 'relu', 'gelu', or 'silu'
        :param down_embed: if True, have lower dimension for initial embedding than in CNN layers
        """
        super().__init__()

        log2 = int(np.log2(r)) + 1
        dilations = [2 ** (n % log2) for n in range(n_layers)]
        d_h = d_model
        if slim:
            d_h = d_h // 2
        layers = [
            ByteNetBlock(
                d_model if i > 0 else n_tokens,
                d_h,
                d_model,
                kernel_size,
                dilation=d,
                rank=rank,
                activation=activation,
            )
            for i, d in enumerate(dilations)
        ]
        self.layers = nn.ModuleList(modules=layers)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, input_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, input_mask=input_mask)
            if self.dropout > 0.0:
                x = F.dropout(x, self.dropout)
        return x


class ByteNetArch(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        kernel_size: int,
        max_dilation: int,
        dropout: float = 0.0,
        slim: bool = True,
        activation: str = "silu",
        rank=None,
        n_tokens: int = 21,
        final_layernorm: bool = True,
    ):
        super().__init__()
        self.embedder = ByteNet(
            n_tokens,
            d_model,
            n_layers,
            kernel_size,
            max_dilation,
            dropout=dropout,
            slim=slim,
            activation=activation,
            rank=rank,
        )
        self.decoder: nn.Linear | PositionFeedForward
        self.last_norm: nn.LayerNorm | nn.Identity
        self.decoder = PositionFeedForward(d_model, n_tokens)

        if final_layernorm:
            self.last_norm = nn.LayerNorm(d_model)
        else:
            self.last_norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_mask = (x == 0).all(-1)
        e = self.embedder(x, input_mask=input_mask)
        e = self.last_norm(e)
        return self.decoder(e)
