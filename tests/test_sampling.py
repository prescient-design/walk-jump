import torch
from omegaconf import DictConfig
from walkjump.constants import ALPHABET_AHO, TOKEN_GAP
from walkjump.model import TrainableScoreModel
from walkjump.sampling import stack_seed_sequences, walkjump
from walkjump.utils import token_string_from_tensor

dummy_score_model_cfg = DictConfig({"arch": {}})


class DummyScoreModel(TrainableScoreModel):
    def __init__(self):
        super().__init__(dummy_score_model_cfg)

    def score(self, y: torch.Tensor) -> torch.Tensor:
        # WARNING: This definition of the score function is valid only for the DenoiseModel!
        return (self.nu(y) - y) / pow(self.sigma, 2)

    def nu(self, ys: torch.Tensor) -> torch.Tensor:
        """Placeholder nu"""
        return ys + torch.randn_like(ys)

    def sample_noise(self, xs: torch.Tensor) -> torch.Tensor:
        return xs

    @property
    def device(self):
        return torch.device("cpu")


def test_stack_seed_sequences():
    seeds = ["EVQLV", "AARRRGGY", "MMMMSKITTLES"]
    stack = stack_seed_sequences(seeds, 10)

    assert stack.size(0) == 30

    returned = [
        x.replace(TOKEN_GAP, "")
        for x in token_string_from_tensor(stack, ALPHABET_AHO, from_logits=True)
    ]

    assert not (set(seeds) - set(returned))


def test_sample_sequences_1seed():
    seed = "EVQLV"
    model = DummyScoreModel()
    num_samples = 10

    seqs = walkjump(seed, model, steps=20, num_samples=num_samples)
    assert len(seqs) == num_samples


def test_masked_sampling():
    """Check if masked residues are preserved somewhere in the sample."""
    seed = "EVQLV"
    model = DummyScoreModel()
    num_samples = 10
    mask_idxs_list = [[0], [0, 1], [2, 4]]

    for mask_idxs in mask_idxs_list:
        seqs = walkjump(
            seed,
            model,
            steps=10,
            num_samples=10,
            mask_idxs=mask_idxs,
        )
        assert len(seqs) == num_samples

        for sample in seqs:
            seq_nogap = sample.replace(TOKEN_GAP, "")
            assert [list(seed)[idx] in list(seq_nogap) for idx in mask_idxs]


def test_sample_sequences_multiseed():
    seeds = ["EVQLV"] * 5
    model = DummyScoreModel()
    num_samples = 10

    seqs = walkjump(seeds, model, steps=20, num_samples=num_samples)
    assert len(seqs) == num_samples * 5
    assert {"fv_heavy_aho", "fv_light_aho", "fv_heavy_aho_seed", "fv_light_aho_seed"}.issubset(
        set(seqs.columns)
    )
