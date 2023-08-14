from functools import partial
from typing import Callable, Optional

from ._langevin import _DEFAULT_SAMPLING_OPTIONS, sachsetal


def create_sampler_fn(
    verbose: bool = True, mask_idxs: Optional[list[int]] = None, **sampling_options
) -> Callable:
    options = _DEFAULT_SAMPLING_OPTIONS | sampling_options
    return partial(sachsetal, sampling_options=options, mask_idxs=mask_idxs, verbose=verbose)
