from walkjump.conformity import conformity_score

import pytest
import torch

def test_conformity_score():
    log_prob = torch.tensor([1,2,4])
    val_log_prob = torch.tensor([3,3,3,3])

    output = conformity_score(log_prob, val_log_prob)

    expected = torch.tensor([
        0.0,
        0.0,
        0.8
    ])
    assert torch.allclose(output, expected)