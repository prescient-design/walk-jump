import numpy as np
from walkjump.metrics import LargeMoleculeDescriptors


def test_large_molecule_descriptors():
    sequence = "MKKTAIAIAVALAGFATVAFA"
    lmd = LargeMoleculeDescriptors.from_sequence(sequence)
    assert lmd.length == 21
    assert np.isclose(lmd.molecular_weight, 2065, atol=1)
