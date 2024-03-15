import warnings
from dataclasses import dataclass

import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import minmax_scale

from ._large_molecule_descriptors import LargeMoleculeDescriptors


class MetricWarning(RuntimeWarning):
    pass


LARGE_MOL_FIGSIZE = (3, 5)
NO_VALID_DESIGNS_WARNING = "There were no valid designs."

warnings.simplefilter("always", category=MetricWarning)


@dataclass
class MetricColumnInfo:
    feature_columns: list[str]
    sample_column: str
    figshape: tuple[int, int]

    def __post_init__(self):
        self.feature_columns = sorted(set(self.feature_columns))


def get_column_info(chain) -> MetricColumnInfo:
    match chain:
        case "fv_heavy":
            return MetricColumnInfo(
                [f"fv_heavy_{feature}" for feature in LargeMoleculeDescriptors.descriptor_names()],
                "fv_heavy_aho",
                LARGE_MOL_FIGSIZE,
            )
        case "fv_light":
            return MetricColumnInfo(
                [f"fv_light_{feature}" for feature in LargeMoleculeDescriptors.descriptor_names()],
                "fv_light_aho",
                LARGE_MOL_FIGSIZE,
            )


def get_batch_descriptors(
    sample_df: pd.DataFrame, ref_feats: pd.DataFrame, chain
) -> tuple[dict[str, float], float, float, float]:
    """
    Compute aggregate statistics for a collection of samples compared to reference.

    Parameters
    ----------
    sample_df: pd.DataFrame
        Collection of samples, generally this would be a return value of
        `walkjump.callbacks.sample_and_compute_metrics()`
    ref_feats: pd.DataFrame
        Pre-computed reference distributions
    chain: ReferenceChainType
        Type of input molecule. Behavior switches based on molecule type.

    Returns
    -------
    Tuple[Dict[str, float], float, float, float]
        Wasserstein distances per statistic column and the
            (average wass. dist., total wass. dist., proportion not NaN)
    """
    info = get_column_info(chain)

    try:
        prop_valid = float(sample_df[info.sample_column].notna().sum()) / len(sample_df)
    except ZeroDivisionError:
        warnings.warn(NO_VALID_DESIGNS_WARNING, category=MetricWarning, stacklevel=2)
        prop_valid = 0.0

    wasserstein_distances = {}
    for column in info.feature_columns:
        # filter out NaN rows for this column
        valid = sample_df.loc[sample_df[column].notna(), column]
        valid_ref = ref_feats.loc[ref_feats[column].notna(), column]

        # min/max norm the validated rows.
        try:
            normed = minmax_scale(valid)
            normed_ref = minmax_scale(valid_ref)
            # compute wasserstein
            wasserstein_distances[f"{column}_wd"] = wasserstein_distance(normed, normed_ref)
        except ValueError:
            wasserstein_distances[f"{column}_wd"] = float("inf")

    total_wd = sum(wasserstein_distances.values())
    avg_wd = total_wd / len(info.feature_columns)

    return wasserstein_distances, avg_wd, total_wd, prop_valid
