from typing import Iterable

import torch
from torch import nn

from ._layers import SeqCNN


class LMSArch(nn.Module):
    def __init__(
        self,
        chain_len: int,
        noise_factor: float = 1.0,
        reg_l2_norm: int = 5,
        kernel_sizes: Iterable[int] = (15, 5, 3),
        hidden: int = 32,
        n_tokens: int = 21,
        friction: float = 1.0,
        activation: str = "relu",
    ):
        super().__init__()
        self.chain_len = chain_len
        self.noise_factor = noise_factor
        self.reg_l2_norm = reg_l2_norm
        self.kernel_sizes = list(kernel_sizes)
        self.hidden = hidden
        self.n_tokens = n_tokens
        self.friction = friction

        self.energy_model = SeqCNN(
            chain_len,
            vocab_size=n_tokens,
            hidden_features=hidden,
            kernel_sizes=kernel_sizes,
            activation=activation,
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.energy_model(y)
