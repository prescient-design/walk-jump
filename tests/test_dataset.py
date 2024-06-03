import hydra
from walkjump.constants import TOKENS_AHO
from walkjump.data import AbDataset

from tests.constants import CONFIG_PATH, TRAINER_OVERRIDES
from tests.fixtures import aho_sequence, mock_ab_dataframe  # noqa: F401


def test_abdataset(mock_ab_dataframe):  # noqa: F811
    dataset = AbDataset(mock_ab_dataframe, TOKENS_AHO)
    print(dataset)
    assert len(dataset[0]) == mock_ab_dataframe.loc[0].str.len().sum()


def test_instantiate_datamodule():
    with hydra.initialize(version_base=None, config_path=CONFIG_PATH):
        cfg = hydra.compose(config_name="train", overrides=TRAINER_OVERRIDES)
        datamodule = hydra.utils.instantiate(cfg.data)
        datamodule.setup(stage="fit")
