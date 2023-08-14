import hydra
import torch
from omegaconf import DictConfig

from walkjump.data import AbBatch
from walkjump.sampling import walk
from walkjump.utils import random_discrete_seeds

from ._base import TrainableScoreModel


class _NullDenoiser:
    def xhat(self, y: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(y).scatter_(-1, y.argmax(-1).unsqueeze(-1), 1.0)


class NoiseEnergyModel(TrainableScoreModel):
    """Model learns to approximate the noise;
    Parameterized by an EBM with a pretrained Score-based Bayes estimator."""

    needs_gradients: bool = True

    def __init__(self, model_cfg: DictConfig):
        super().__init__(model_cfg)

        if model_cfg.get("pretrained"):
            self.denoise_model = hydra.utils.instantiate(model_cfg.pretrained)
        else:
            self.denoise_model = _NullDenoiser()

        self.training_sampler_fn = hydra.utils.instantiate(model_cfg.sampler)

        if not isinstance(self.denoise_model, _NullDenoiser):
            self.sigma_renorm = self.sigma / self.denoise_model.sigma
        else:
            self.sigma_renorm = self.sigma

    def score(self, y: torch.Tensor) -> torch.Tensor:
        """Gets called in langevin to get scores of noisy inputs."""
        # y = y.detach()  # make leaf variable
        if not y.requires_grad:
            y.requires_grad = True
        # y.requires_grad = True  # already true
        with torch.set_grad_enabled(True):
            energy, _h = self.model.energy_model(y)

            # eng.sum().backward()
            # score = y.grad.data
            score = torch.autograd.grad(-energy.sum(), y, create_graph=self.training)[
                0
            ]  # correct sign

        return score

    def apply_noise(self, xs: torch.Tensor) -> torch.Tensor:
        # use `sample_noise` method from denoise model, but update noise level to LMS settings
        noised = xs + self.renorm_noise_factor * self.sample_noise(xs)
        noised.requires_grad = True
        return noised

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.training_cfg.lr,
            betas=(self.training_cfg.beta1, 0.999),
            weight_decay=self.training_cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 1000, gamma=0.97
        )  # Exponential decay over epochs

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "frequency": 1,
                "interval": "step",
            },
        }

    def xhat(self, y: torch.Tensor) -> torch.Tensor:
        return self.denoise_model.xhat(y)

    def compute_loss(self, batch: AbBatch) -> torch.Tensor:
        random_seeds = random_discrete_seeds(
            batch.x.size(0),
            n_tokens=self.arch_cfg.n_tokens,
            seed_length=self.arch_cfg.chain_len,
            onehot=True,
        )

        y_fake = walk(random_seeds, self, self.training_sampler_fn)
        y_real = self.apply_noise(batch.x)

        energy_real, _ = self.model(y_real)
        energy_fake, _ = self.model(y_fake)

        cdiv_coeff = 1.0

        cdiv_loss = cdiv_coeff * energy_real.mean() - energy_fake.mean()

        reg_loss = (energy_real**2 + energy_fake**2).mean()

        loss = cdiv_loss + self.model.reg_l2_norm * reg_loss

        return loss
