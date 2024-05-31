from typing import Callable

import hydra
import pytest
from omegaconf import DictConfig, OmegaConf
from walkjump.cmdline import sample, train
from walkjump.cmdline.utils import instantiate_callbacks

from tests.constants import CONFIG_PATH, SAMPLER_OVERRIDES, TRAINER_OVERRIDES

COMMAND_TO_OVERRIDES = {"train": TRAINER_OVERRIDES, "sample": SAMPLER_OVERRIDES}


def test_instantiate_callbacks_and_trainer():
    with hydra.initialize(version_base=None, config_path=CONFIG_PATH):
        cfg = hydra.compose(config_name="train", overrides=TRAINER_OVERRIDES)
        callbacks = instantiate_callbacks(cfg.get("callbacks"))
        trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks)
        assert trainer


@pytest.mark.parametrize("cmd_name,cmd", [("train", train), ("sample", sample)])
def test_cmdline_dryruns(cmd_name: str, cmd: Callable[[DictConfig], bool]):
    with hydra.initialize(version_base=None, config_path=CONFIG_PATH):
        cfg = hydra.compose(config_name=cmd_name, overrides=COMMAND_TO_OVERRIDES.get(cmd_name))
        print(OmegaConf.to_yaml(cfg))
        assert cmd(cfg)
