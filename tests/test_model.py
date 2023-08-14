import hydra
import pytest
from omegaconf import OmegaConf

CONFIG_PATH = "../src/walkjump/hydra_config/model"
OVERRIDES = {"noise_ebm": ["~model_cfg/pretrained"]}


@pytest.mark.parametrize("model_name", ["denoise", "noise_ebm"])
def test_instantiate_models(model_name: str):
    with hydra.initialize(version_base=None, config_path=CONFIG_PATH):
        cfg = hydra.compose(config_name=model_name, overrides=OVERRIDES.get(model_name))
        print(OmegaConf.to_yaml(cfg))
        model = hydra.utils.instantiate(cfg, _recursive_=False)
        assert model
