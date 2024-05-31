import math
from typing import Optional

import torch
from tqdm import trange

from walkjump.model import TrainableScoreModel

_DEFAULT_SAMPLING_OPTIONS = {"delta": 0.5, "friction": 1.0, "lipschitz": 1.0, "steps": 100}


def sachsetal(
    model: TrainableScoreModel,
    y: torch.Tensor,
    v: torch.Tensor,
    sampling_options: dict[str, float | int] = _DEFAULT_SAMPLING_OPTIONS,
    mask_idxs: Optional[list[int]] = None,
    save_trajectory: bool = False,
    verbose: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    options = _DEFAULT_SAMPLING_OPTIONS | sampling_options  # overwrite

    delta, gamma, lipschitz = options["delta"], options["friction"], options["lipschitz"]

    step_iterator = (
        trange(int(options["steps"]), desc="Sachs, et al", leave=False)
        if verbose
        else range(int(options["steps"]))
    )

    with torch.set_grad_enabled(model.needs_gradients):
        u = pow(lipschitz, -1)  # inverse mass
        zeta1 = math.exp(
            -gamma
        )  # gamma is effective friction here (originally 'zeta1 = math.exp(-gamma * delta)')
        zeta2 = math.exp(-2 * gamma)

        traj = []
        for _i in step_iterator:
            # y += delta * v / 2  # y_{t+1}
            y = y + delta * v / 2

            psi = model(y)
            noise_update = torch.randn_like(y)
            # prevent updates in masked positions
            if mask_idxs:
                psi[:, mask_idxs, :] = 0.0
                noise_update[:, mask_idxs, :] = 0.0
            v += u * delta * psi / 2  # v_{t+1}
            v = (
                zeta1 * v + u * delta * psi / 2 + math.sqrt(u * (1 - zeta2)) * noise_update
            )  # v_{t+1}
            # y += delta * v / 2  # y_{t+1}
            y = y + delta * v / 2
            # gc.collect()
            # torch.cuda.empty_cache()
            if save_trajectory:
                traj.append(y)
        return y, v, traj
