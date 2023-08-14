import torch

from walkjump.constants import LENGTH_FV_HEAVY_AHO, LENGTH_FV_LIGHT_AHO, TOKENS_AHO


def isotropic_gaussian_noise_like(x: torch.Tensor, sigma: float) -> torch.Tensor:
    return sigma * torch.randn_like(x.float())


def random_discrete_seeds(
    n_seeds: int,
    n_tokens: int = len(TOKENS_AHO),
    seed_length: int = LENGTH_FV_LIGHT_AHO + LENGTH_FV_HEAVY_AHO,
    onehot: bool = False,
) -> torch.Tensor:
    random_seeds = torch.randint(0, n_tokens, (n_seeds, seed_length))
    if onehot:
        return torch.nn.functional.one_hot(random_seeds, num_classes=n_tokens)
    else:
        return random_seeds
