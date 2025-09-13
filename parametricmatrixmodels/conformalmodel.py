from __future__ import annotations

from typing import Callable, Tuple

import jax.numpy as np
import numpy as onp
from packaging.version import parse

import parametricmatrixmodels as pmm

from .model import Model


class ConformalModel(object):
    r"""
    A wrapper class for conformal prediction models.

    Confidence intervals (uncertainty quantification) is optionally provided by
    conformal prediction methods. A heuristic notion of uncertainty is
    transformed into a rigorous statistical guarantee on the coverage of the
    prediction intervals via a calibration dataset. See [1]_ for an overview of
    conformal prediction methods.

    References
    ----------
    .. [1] Angelopoulos, A. N., & Bates, S. (2022). A gentle introduction to
       conformal prediction and distribution-free uncertainty quantification.
       arXiv preprint arXiv:2107.07511. https://arxiv.org/abs/2107.07511
    """
