from dataclasses import dataclass
from functools import cached_property

import torch

from walkjump.constants import TOKENS_AHO


@dataclass
class AbBatch:
    batch_tensor: torch.Tensor
    """(b, L)-shaped tensor of sequences"""
    vocab_size: int = len(TOKENS_AHO)

    @classmethod
    def from_tensor_pylist(
        cls, inputs: list[torch.Tensor], vocab_size: int = len(TOKENS_AHO)
    ) -> "AbBatch":

        packed_batch = torch.stack(inputs, dim=0)
        return cls(packed_batch, vocab_size=vocab_size)

    @cached_property
    def x(self) -> torch.Tensor:
        return torch.nn.functional.one_hot(self.batch_tensor, num_classes=self.vocab_size).float()
