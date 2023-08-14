from typing import Iterable

import hydra
import pandas as pd
import torch
from lightning.pytorch.callbacks import Callback
from omegaconf import DictConfig

from walkjump.constants import ALPHABET_AHO, LENGTH_FV_HEAVY_AHO, LENGTH_FV_LIGHT_AHO, RANGES_AHO
from walkjump.data import AbDataset
from walkjump.model import DenoiseModel, NoiseEnergyModel
from walkjump.utils import random_discrete_seeds, token_string_from_tensor

model_typer = {
    "denoise": DenoiseModel,
    "noise_ebm": NoiseEnergyModel,
}

_ERR_MSG_UNRECOGNIZED_REGION = "Could not parse these regions to redesign: {regions}"
_LOG_MSG_INSTANTIATE_MODEL = "Loading {model_type} model from {checkpoint_path}"


def instantiate_redesign_mask(redesign_regions: Iterable[str]) -> list[int]:
    unrecognized_regions = set(redesign_regions) - set(RANGES_AHO.keys())
    assert not unrecognized_regions, _ERR_MSG_UNRECOGNIZED_REGION.format(
        regions=unrecognized_regions
    )

    redesign_accumulator = set()
    for region in redesign_regions:
        chain = region[0]
        aho_start, aho_end = RANGES_AHO[region]
        shift = LENGTH_FV_HEAVY_AHO * int(chain == "L")

        redesign_accumulator |= set(range(aho_start + shift, aho_end + shift))

    mask_idxs = sorted(
        set(range(0, LENGTH_FV_HEAVY_AHO + LENGTH_FV_LIGHT_AHO)) - redesign_accumulator
    )
    return mask_idxs


def instantiate_seeds(seeds_cfg: DictConfig) -> list[str]:
    if seeds_cfg.seeds == "denovo":
        seed_batch = random_discrete_seeds(seeds_cfg.limit_seeds, onehot=False)
    else:
        seed_df = pd.read_csv(seeds_cfg.seeds)
        dataset = AbDataset(seed_df)
        seed_batch = torch.stack(
            [dataset[i] for i in range(seeds_cfg.limit_seeds or len(dataset))], dim=0
        )

    return token_string_from_tensor(seed_batch, alphabet=ALPHABET_AHO, from_logits=False)


def instantiate_model_for_sample_mode(
    sample_mode_model_cfg: DictConfig,
) -> NoiseEnergyModel | DenoiseModel:
    print(
        "[instantiate_model_for_sample_mode]",
        _LOG_MSG_INSTANTIATE_MODEL.format(
            model_type=sample_mode_model_cfg.model_type,
            checkpoint_path=sample_mode_model_cfg.checkpoint_path,
        ),
    )
    model = model_typer[sample_mode_model_cfg.model_type].load_from_checkpoint(
        sample_mode_model_cfg.checkpoint_path
    )
    if isinstance(model, NoiseEnergyModel) and sample_mode_model_cfg.denoise_path is not None:

        print(
            "[instantiate_model_for_sample_mode] (model.denoise_model)",
            _LOG_MSG_INSTANTIATE_MODEL.format(
                model_type="denoise", checkpoint_path=sample_mode_model_cfg.denoise_path
            ),
        )
        model.denoise_model = DenoiseModel.load_from_checkpoint(sample_mode_model_cfg.denoise_path)
        model.denoise_model.eval()
        model.denoise_model.training = False

    return model


def instantiate_callbacks(callbacks_cfg: DictConfig) -> list[Callback]:
    """Instantiates callbacks from config."""
    callbacks: list[Callback] = []

    if not callbacks_cfg:
        print("[instantiate_callbacks] No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("[instantiate_callbacks] Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            print(f"[instantiate_callbacks] Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks
