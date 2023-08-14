import torch
from torch import nn

from walkjump.data import AbBatch
from walkjump.utils import isotropic_gaussian_noise_like

from ._base import TrainableScoreModel


class DenoiseModel(TrainableScoreModel):
    needs_gradients: bool = False

    def score(self, y: torch.Tensor) -> torch.Tensor:
        return (self.model(y) - y) / pow(self.sigma, 2)

    def compute_loss(self, batch: AbBatch) -> torch.Tensor:
        y = batch.x + isotropic_gaussian_noise_like(batch.x, self.sigma)
        xhat = self.xhat(y)
        return nn.MSELoss()(xhat, batch.x)
