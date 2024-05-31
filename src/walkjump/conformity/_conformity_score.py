import torch
from torch import Tensor


def conformity_score(log_prob: Tensor, val_log_prob: Tensor) -> Tensor:
    """Returns the conformity test statistic.

    For each test likelihood value, compute the proportion of validation likelihoods
    that are less than or equal to it. Between 0 and 1, where:

    - > 0.5: higher conformity, more similar to training data than validation data
    - 0.5: optimal conformity, as similar to training data as validation data
    - < 0.5: lower conformity, validation is more similar to training data than test data


    References
    * [A Tutorial on Conformal Prediction](https://jmlr.csail.mit.edu/papers/volume9/shafer08a/shafer08a.pdf)
    * [Criteria of efficiency for conformal prediction](https://arxiv.org/pdf/1603.04416.pdf)


    Parameters
    ----------
    log_prob : Tensor
        The log probability of data to compute conformity scores for, shape (n)
    val_log_prob : Tensor
        The log probability of the validation data, shape (n_val)
    Returns
    -------
    Tensor
        The conformity test statistic, shape (n, )

    """
    n_val = val_log_prob.shape[0]

    p_value = Tensor(
        [
            float("nan") if torch.isnan(s) else (val_log_prob <= s).sum().item() / (n_val + 1)
            for s in log_prob
        ]
    )

    return p_value
