from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import jax
import jax.numpy as np
import numpy as onp
from beartype import beartype
from jaxtyping import jaxtyped
from packaging.version import parse
from scipy.spatial import KDTree

import parametricmatrixmodels as pmm

from .. import preprocessing, tree_util
from ..model import Model
from ..model_util import ModelState
from ..typing import (
    Any,
    Array,
    Data,
    Dict,
    Inexact,
    List,
    PyTree,
    Real,
    Serializable,
    Tuple,
)


@jaxtyped(typechecker=beartype)
class ConformalizedModel(object):
    __version__ = "0.0.0"
    r"""
    A wrapper class for conformalized prediction models.

    Unlike other models and modules, ConformalizedModel is not a module and
    cannot be nested inside other models. It is a wrapper around a trained
    Model only.

    Confidence intervals (uncertainty quantification) is optionally provided by
    conformal prediction methods. A heuristic notion of uncertainty is
    transformed into a rigorous statistical guarantee on the coverage of the
    prediction intervals via a calibration dataset. See [1]_ for an overview of
    conformal prediction methods.

    For now, the uncertainty heuristic is fixed and is the MAD (median absolute
    deviation) normalized sum of three sensitivities:

    The sensitivity of the model with respect to its trainable parameters,

    .. math::

        S_\Theta(x) = \left\Vert\frac{\partial f}{\partial \Theta} \cdot
            \left|\Theta\right| \right\Vert_2^2 / \mathrm{size}(\Theta)

    the sensitivity of the model with respect to the continuous input features
    via the training data,

    .. math::

        S_x(x) = \left\Vert\frac{\partial f}{\partial x_\mathrm{cont}} \cdot
            \mathrm{IQR}(X_\mathrm{cont,\,train}) \right\Vert_2^2 /
            n_\mathrm{cont\,features}

    and the distance of the evaluation point from the training data

    .. math::

        S_d(x) = d(x, X_\mathrm{train})

    where :math:`f` is the model function, :math:`\Theta` are the
    trainable parameters of the model, :math:`x` is the evaluation point,
    :math:`X_\mathrm{train}` is the training data,
    :math:`X_\mathrm{cont,\,train}` is the training data of the continuous
    features only, :math:`\mathrm{IQR}` is the interquartile range, and
    :math:`d` is a distance function, which we take to be the distance to be
    the mean `k`-nearest neighbor Grower distance to the training data.

    These three sensitivities are normalized over the calibration set by the
    MAD,

    .. math::
        \tilde{S}(x) = S(x) / \mathrm{MAD}(S(X_\mathrm{cal}))

    making the final uncertainty heuristic

    .. math::
        u(x) = \tilde{S}_\Theta(x) + \tilde{S}_x(x) + \tilde{S}_d(x)

    The normalization factors and the KDTree for the distance sensitivity are
    computed at calibration time and saved. So the evaluation time scales only
    with the cost of the forward pass and the backward pass for the gradients,
    as well as :math:`\mathcal{O}(\log n_\mathrm{train})` for the KDTree query.
    If training data is not provided during calibration, the input and distance
    sensitivities are not computed and the uncertainty heuristic is just the
    sensitivity to parameters.

    The process of conformalization takes this heuristic uncertainty and
    transforms it into a rigorous statistical guarantee on the coverage of the
    prediction intervals via a calibration dataset. This process is
    distribution-free, meaning that it makes no assumptions on the distribution
    of the data or the model.

    If the calibration labels have associated uncertainties, then the MAD
    normalization is replaced by a weighted MAD. This breaks the formal
    coverage guarantee of conformal prediction, but still results in
    approximate coverage.

    If the evaluation point(s) have associated uncertainties, then these are
    incorporated into the uncertainty heuristic as an additional sensitivity
    term in :math:`S_x`,

    .. math::
        S_\mathrm{input}(x) = \left\Vert\frac{\partial f}
            {\partial x_\mathrm{cont}}
            \cdot \sigma_{x_\mathrm{cont}} \right\Vert_2^2 /
            n_\mathrm{cont\,features}

    where :math:`\sigma_{x_\mathrm{cont}}` is the uncertainty in the continuous
    input features.

    References
    ----------
    .. [1] Angelopoulos, A. N., & Bates, S. (2022). A gentle introduction to
       conformal prediction and distribution-free uncertainty quantification.
       arXiv preprint arXiv:2107.07511. https://arxiv.org/abs/2107.07511

    See Also
    --------
    Model
        Abstract base class for models formed from modules that can be trained
        and evaluated.
    SequentialModel
        A simple sequential model that stacks modules linearly.
    NonsequentialModel
        A flexible model that allows for arbitrary module connections.
    """

    def __init__(
        self,
        model: Model,
        alpha: float = 1 - 0.682689492137,
        nn_dist_quantile: float = 0.5,
    ):
        r"""
        Parameters
        ----------
        model
            A trained ``Model`` instance.
        alpha
            The default miscoverage level for the conformal prediction
            intervals. Also known as the significance level. Must be in the
            range :math:`(0, 1)`. Default is :math:`1 - \Phi(1) \approx 0.32`,
            where :math:`\Phi` is the standard normal cumulative distribution
            function, which corresponds to a 68% prediction interval, or one
            standard deviation for normally distributed data. Can be specified
            when calling the model, this is just a default value.
        nn_dist_quantile
            The quantile of the nearest neighbor distances in the training data
            to use as the cutoff for the nearest-neightbor calculation in
            distance sensitivity. Points beyond this distance from any
            training point instead fall back to just the nearest neighbor
            distance.
        """

        # default alpha
        self.default_alpha = alpha
        self.nn_dist_quantile = nn_dist_quantile
        # backing model
        self.model = model
        # conformalized scores and quantiles
        self.scores = None
        self.qhats = {}  # dict of quantiles for different alphas
        # storaged precomputed data
        self._training_data = None
        self._kdtree = None
        self._iqr_X_train = None
        self._range_X_train = None
        self._norm_S_params = None
        self._norm_S_input = None
        self._norm_S_dist = None
        self._continuous_features = None
        self._distance_cutoff = None
        self._onehot_training_enc = None
        self._stratified_bias_corrections = None
        self._strata_unique_values = None
        self._strata_biases = None
        self._num_params_sensitivity = (
            None  # not the same as num_trainable_floats
        )
        self._num_input_sensitivity = None
        self._num_distance_sensitivity = None
        self._input_sensitivity = None
        self._parameter_sensitivity = None
        self._distance_sensitivity = None

    def calibrate(
        self,
        X_cal: PyTree[Inexact[Array, "cal_batch ?f0 ?*f"], " In"],
        Y_cal: (
            PyTree[Inexact[Array, "cal_batch ?t0 ?*t"], " Out"] | None
        ) = None,
        X_cal_unc: (
            PyTree[Inexact[Array, "cal_batch ?f0 ?*f"], " In"] | None
        ) = None,
        Y_cal_unc: (
            PyTree[Inexact[Array, "cal_batch ?t0 ?*t"], " Out"] | None
        ) = None,
        X_train: (
            PyTree[Inexact[Array, "train_batch ?f0 ?*f"], " In"] | None
        ) = None,
        parameter_sensitivity: bool | PyTree[bool, " Model"] = True,
        input_sensitivity: bool | PyTree[bool, " In"] = True,
        distance_sensitivity: bool | PyTree[bool, " In"] = True,
        continuous_features: PyTree[bool, " In"] | None = None,
        stratified_bias_corrections: PyTree[bool, " In"] | bool | None = None,
        dtype: jax.typing.DTypeLike = np.float64,
        max_batch_size: int | None = None,
        rng: Any | int | None = None,
        update_state: bool = False,
        fwd: bool | None = None,
        skip_sanity_checks: bool = False,
        verbose: bool = False,
    ) -> float:
        r"""
        Calibrate the conformal prediction model on a calibration dataset.
        Parameters
        ----------
        <TODO>

        """

        if Y_cal is not None:
            # supervised learning
            if not tree_util.uniform_leaf_shapes_equal(X_cal, Y_cal, axis=0):
                raise ValueError(
                    "X_cal and Y_cal must have the same number of samples. "
                    "I.e. the leading dimension (axis 0) across all leaves "
                    "of the pytrees must be the same."
                )
        if Y_cal_unc is not None and Y_cal is not None:
            if not tree_util.shapes_equal(Y_cal, Y_cal_unc):
                raise ValueError(
                    "Y_cal and Y_cal_unc must have the same shape."
                )
        if X_cal_unc is not None:
            if not tree_util.shapes_equal(X_cal, X_cal_unc):
                raise ValueError(
                    "X_cal and X_cal_unc must have the same shape."
                )
        if not tree_util.all_equal(
            self.model.input_shape, tree_util.get_shapes(X_cal, slice(1, None))
        ):
            raise ValueError(
                "The input shape of the model must match the shape of the"
                f" calibration features. Got {self.model.input_shape} and"
                f" {tree_util.get_shapes(X_cal, slice(1, None))}."
            )
        if X_train is not None and not tree_util.all_equal(
            self.model.input_shape,
            tree_util.get_shapes(X_train, slice(1, None)),
        ):
            raise ValueError(
                "The input shape of the model must match the shape of the"
                f" training features. Got {self.model.input_shape} and"
                f" {tree_util.get_shapes(X_train, slice(1, None))}."
            )
        if Y_cal is not None and not tree_util.all_equal(
            self.model.output_shape,
            tree_util.get_shapes(Y_cal, slice(1, None)),
        ):
            raise ValueError(
                "The output shape of the model must match the shape of the"
                f" calibration targets. Got {self.model.output_shape} and"
                f" {tree_util.get_shapes(Y_cal, slice(1, None))}."
            )
        elif Y_cal is None and not tree_util.all_equal(
            self.model.output_shape,
            tree_util.get_shapes(X_cal, slice(1, None)),
        ):
            raise ValueError(
                "For unsupervised calibration, the output shape of the model"
                " must match the shape of the calibration features. Got"
                f" {self.model.output_shape} and"
                f" {tree_util.get_shapes(X_cal, slice(1, None))}."
            )
        if continuous_features is not None:
            if jax.tree.structure(continuous_features) != jax.tree.structure(
                X_cal
            ):
                raise ValueError(
                    "The structure of continuous_features must match the "
                    "structure of X_cal. Got "
                    f"{jax.tree.structure(continuous_features)} and "
                    f"{jax.tree.structure(X_cal)}."
                )
        else:
            # if not provided, assume all features are continuous
            continuous_features = jax.tree.map(lambda _: True, X_cal)

        self._continuous_features = continuous_features

        if stratified_bias_corrections is not None:
            # stratified bias correction should be a bool or PyTree with the
            # same structure as the input features, and each True must be False
            # in the continuous features, since stratified bias correction only
            # makes sense across categorical features
            # if true, then a single global bias correction is applied
            if not isinstance(stratified_bias_corrections, bool):
                if jax.tree.structure(
                    stratified_bias_corrections
                ) != jax.tree.structure(X_cal):
                    raise ValueError(
                        "The structure of stratified_bias_corrections must"
                        " match the structure of X_cal. Got"
                        f" {jax.tree.structure(stratified_bias_corrections)}"
                        f" and {jax.tree.structure(X_cal)}."
                    )

                def check_stratified_bias_correction_validity(sbc, cont):
                    if sbc and cont:
                        raise ValueError(
                            "Stratified bias correction cannot be applied to"
                            " continuous features. Please set the value to"
                            " False for continuous features. Got True for a"
                            " continuous feature."
                        )

                jax.tree.map(
                    check_stratified_bias_correction_validity,
                    stratified_bias_corrections,
                    continuous_features,
                )
        else:
            # if not provided, assume no stratified bias correction
            stratified_bias_corrections = False

        self._stratified_bias_corrections = stratified_bias_corrections

        if not skip_sanity_checks:
            # check each feature marked as continuous has n_unique greater than
            # some absolute threshold and n_unique/n_total greater than some
            # relative threshold in the training data (or calibration
            # data if training data not provided)
            if X_train is not None:
                data_for_sanity_check = X_train
            else:
                data_for_sanity_check = X_cal

            # if the data for the sanity check has more than MAX_N samples or
            # MAX_K features, we subsample it to speed up the sanity check
            MAX_N = 100_000
            MAX_K = 100
            shuffle_rng = jax.random.key(42)

            data_for_sanity_check = jax.tree.map(
                lambda x: jax.random.permutation(shuffle_rng, x, axis=0)[
                    :MAX_N, :MAX_K
                ],
                data_for_sanity_check,
            )

            abs_thresh = 10
            rel_thresh = 0.05

            def sanity_check(feature, is_continuous):
                n_unique = len(np.unique(feature, axis=0))
                n_total = feature.shape[0]

                looks_continuous = (
                    n_unique > abs_thresh and (n_unique / n_total) > rel_thresh
                )

                return looks_continuous == is_continuous

            sanity_checks = jax.tree.map(
                sanity_check, data_for_sanity_check, self._continuous_features
            )

            if not jax.tree.all(sanity_checks):
                warnings.warn(
                    "Some features failed the continuous/categorical sanity"
                    " check and appear to disagree with the provided"
                    " continuous_features mask. This means that some features"
                    " marked as continuous have very few unique values in the"
                    " training/calibration data, which may indicate that they"
                    " are actually categorical. Conversely, some features"
                    " marked as non-continuous have many unique values in the"
                    " data, which may indicate that they are actually"
                    " continuous. This may lead to poor performance of the"
                    " conformal prediction intervals."
                    f"\nSanity checks:\n{sanity_checks}"
                    "\n\nThis warning can be suppressed by setting"
                    " skip_sanity_checks=True when calling the calibrate"
                    " method."
                )

        # make PyTrees of booleans for which sensitivities to compute, or check
        # their structure if already provided as PyTrees
        if isinstance(parameter_sensitivity, bool):
            self._parameter_sensitivity = jax.tree.map(
                lambda _: parameter_sensitivity, self.model.get_params()
            )
        else:
            if jax.tree.structure(parameter_sensitivity) != jax.tree.structure(
                self.model.get_params()
            ):
                raise ValueError(
                    "The structure of parameter_sensitivity must match the"
                    " structure of the model's parameters. Got "
                    f"{jax.tree.structure(parameter_sensitivity)} and "
                    f"{jax.tree.structure(self.model.get_params())}."
                )
            self._parameter_sensitivity = parameter_sensitivity

        if isinstance(input_sensitivity, bool):
            self._input_sensitivity = jax.tree.map(
                lambda _: input_sensitivity,
                self.model.input_shape,
                is_leaf=tree_util.is_shape_leaf,
            )
        else:
            if jax.tree.structure(input_sensitivity) != jax.tree.structure(
                self.model.input_shape, is_leaf=tree_util.is_shape_leaf
            ):
                raise ValueError(
                    "The structure of input_sensitivity must match the"
                    " structure of the model's input shape. Got "
                    f"{jax.tree.structure(input_sensitivity)} and "
                    f"{jax.tree.structure(self.model.input_shape)}."
                )
            self._input_sensitivity = input_sensitivity

        if isinstance(distance_sensitivity, bool):
            self._distance_sensitivity = jax.tree.map(
                lambda _: distance_sensitivity,
                self.model.input_shape,
                is_leaf=tree_util.is_shape_leaf,
            )
        else:
            if jax.tree.structure(distance_sensitivity) != jax.tree.structure(
                self.model.input_shape, is_leaf=tree_util.is_shape_leaf
            ):
                raise ValueError(
                    "The structure of distance_sensitivity must match the"
                    " structure of the model's input shape. Got "
                    f"{jax.tree.structure(distance_sensitivity)} and "
                    f"{jax.tree.structure(self.model.input_shape)}."
                )
            self._distance_sensitivity = distance_sensitivity

        # precompute and store relevant fields

        Y_cal_pred = None  # define now to potentially reuse later
        strata_idxs = None  # " "
        if not isinstance(
            self._stratified_bias_corrections, bool
        ) and tree_util.any(self._stratified_bias_corrections):

            # for all of the features that we are applying stratified bias
            # correction across, we get the unique values to identify each
            # strata
            X_cal_strata_features_flat = np.concatenate(
                jax.tree.leaves(
                    jax.tree.map(
                        lambda x, sbc: (
                            x.reshape(x.shape[0], -1)
                            if sbc
                            else np.empty((x.shape[0], 0), dtype=x.dtype)
                        ),
                        X_cal,
                        self._stratified_bias_corrections,
                    )
                ),
                axis=-1,
            )
            self._strata_unique_values = np.unique(
                X_cal_strata_features_flat, axis=0
            )

            # partition the calibration X and Y by the stratification features,
            # and compute the residuals for each partition
            matches = np.all(
                X_cal_strata_features_flat[:, None, :]
                == self._strata_unique_values[None, :, :],
                axis=-1,
            )
            found = np.any(matches, axis=-1)
            all_idxs = np.argmax(matches, axis=-1)
            # for k unique values, any point that doesn't match any of them
            # gets assigned to index k
            idxs = np.where(
                found, all_idxs, self._strata_unique_values.shape[0]
            )
            Y_cal_pred = self.model(
                X_cal,
                dtype=dtype,
                rng=rng,
                update_state=update_state,
                max_batch_size=max_batch_size,
                verbose=verbose,
            )
            Y_cal_residuals = tree_util.sub(
                Y_cal if Y_cal is not None else X_cal,
                Y_cal_pred,
            )

            # partition the residuals by the strata and take the median
            self._strata_biases = jax.tree.map(
                lambda y: np.array(
                    [
                        np.median(y[idxs == i], axis=0)
                        for i in range(self._strata_unique_values.shape[0])
                    ]
                    # bias of 0.0 for points that don't match any strata
                    + [[0.0]],
                    dtype=y.dtype,
                ),
                Y_cal_residuals,
            )

            # save for later
            strata_idxs = idxs

        elif (
            isinstance(self._stratified_bias_corrections, bool)
            and self._stratified_bias_corrections
        ):
            # no stratification, just a single global bias correction applied
            # over the leaves and the leading dimension
            Y_cal_pred = self.model(
                X_cal,
                dtype=dtype,
                rng=rng,
                update_state=update_state,
                max_batch_size=max_batch_size,
                verbose=verbose,
            )

            Y_cal_residuals = tree_util.sub(
                Y_cal if Y_cal is not None else X_cal,
                Y_cal_pred,
            )
            self._strata_biases = jax.tree.map(
                lambda y: np.median(y, axis=0), Y_cal_residuals
            )

        self._training_data = X_train

        self._num_input_sensitivity = sum(
            jax.tree.leaves(
                jax.tree.map(
                    lambda x, cont, s: (
                        np.prod(np.array(x.shape[1:])) if cont and s else 0
                    ),
                    X_cal,
                    continuous_features,
                    self._input_sensitivity,
                )
            )
        )

        # get number of trainable parameters subjected to sensitivity by
        # summing the sizes of the parameter arrays
        self._num_params_sensitivity = sum(
            jax.tree.leaves(
                jax.tree.map(
                    lambda p, s: p.size if s else 0,
                    self.model.get_params(),
                    self._parameter_sensitivity,
                )
            )
        )

        # get the number of features subject to distance sensitivity by summing
        # the sizes of the input arrays, this counts only the
        # pre-one-hot-encoded features, since the distance sensitivity is
        # computed on the original input space, not the one-hot-encoded space,
        # so each original feature contributes to the distance sensitivity by
        # its size (number of dimensions) if it is marked as continuous and
        # included in the distance sensitivity, and contributes 0 otherwise
        self._num_distance_sensitivity = sum(
            jax.tree.leaves(
                jax.tree.map(
                    lambda x, cont, s: (
                        np.prod(np.array(x.shape[1:]))
                        if cont and s
                        else 1 if s else 0
                    ),
                    X_cal,
                    continuous_features,
                    self._distance_sensitivity,
                )
            )
        )

        if self._training_data is not None:
            # compute IQR for continuous features
            def compute_iqr(
                feature: Inexact[Array, " n_samples f0 *f"],
                is_continuous: bool,
                input_sensitivity: bool,
            ) -> Inexact[Array, " f0 *f"]:
                if not is_continuous or not input_sensitivity:
                    # zero IQR for categorical features since we don't want
                    # them to contribute to the input sensitivity (their
                    # derivatives are not meaningful)
                    return np.zeros(feature.shape[1:], dtype=feature.dtype)
                q75, q25 = np.quantile(feature, np.array([0.75, 0.25]), axis=0)
                return q75 - q25

            # compute range for all features
            def compute_range(
                feature: Inexact[Array, " n_samples f0 *f"],
                is_continuous: bool,
            ) -> Real[Array, " f0 *f"]:
                # categorical features are scaled by 0.5, since we want
                # categorical features to have d = {0, 1} after one-hot
                # encoding which has a range of 1 and a max grower distance
                # of 2 (L1 distance)
                if not is_continuous:
                    return np.full(feature.shape[1:], 2.0, dtype=feature.dtype)
                return np.ptp(feature, axis=0)

            self._iqr_X_train = jax.tree.map(
                compute_iqr,
                self._training_data,
                self._continuous_features,
                self._input_sensitivity,
            )

            self._range_X_train = jax.tree.map(
                compute_range, self._training_data, self._continuous_features
            )

            # since we want to be able to use KDTree for fast kNN lookup, we
            # need to convert the non-continuous features to one-hot encoding.
            # that way the L1 distance between these features is the same as
            # the binary yes/no distance which is used in the Gower distance
            self._onehot_training_enc = preprocessing.MixedScaler(
                jax.tree.map(
                    lambda cont: (None if cont else preprocessing.OneHot()),
                    continuous_features,
                )
            )
            self._onehot_training_enc.fit(self._training_data)

            flat_data = self._prepare_data_for_kdtree(self._training_data)

            # build KDTree
            self._kdtree = KDTree(flat_data)

            # query the tree with the training data to get the distance cutoff
            # for the distance sensitivity
            distances, _ = self._kdtree.query(
                flat_data, k=2, p=1.0, workers=-1
            )
            # if the nearest neighbor distance is zero (the tree correctly
            # identified the point itself as the nearest neighbor), we take the
            # second nearest neighbor distance as the distance to the nearest
            # neighbor, otherwise (due to numerical error in KDTree) we take
            # the first nearest neighbor distance
            nn_distances = np.where(
                distances[:, 0] == 0, distances[:, 1], distances[:, 0]
            )
            self._distance_cutoff = np.quantile(
                nn_distances, self.nn_dist_quantile
            )

        weights = (
            jax.tree.map(lambda x: 1 / x, Y_cal_unc)
            if Y_cal_unc is not None
            else None
        )

        # compute each sensitivity for the calibration data, and then compute
        # the MAD normalization
        if tree_util.any(self._parameter_sensitivity):
            S_params_cal = self.parameter_sensitivity(
                X_cal,
                dtype=dtype,
                max_batch_size=max_batch_size,
                rng=rng,
                update_state=update_state,
                fwd=fwd,
                normalize=False,
                verbose=verbose,
            )
            self._norm_S_params = pytree_robust_normalization(
                S_params_cal, weights
            )
        if self._training_data is not None:
            if tree_util.any(self._input_sensitivity):
                S_input_cal = self.input_sensitivity(
                    X_cal,
                    X_unc=X_cal_unc,
                    dtype=dtype,
                    max_batch_size=max_batch_size,
                    rng=rng,
                    update_state=update_state,
                    fwd=fwd,
                    normalize=False,
                    verbose=verbose,
                )
                self._norm_S_input = pytree_robust_normalization(
                    S_input_cal, weights
                )

            S_dist_cal = self.distance_sensitivity(
                X_cal,
                dtype=dtype,
                normalize=False,
            )
            # no weights for distance sensitivity since it is independent of
            # the labels
            self._norm_S_dist = pytree_robust_normalization(S_dist_cal)

        # compute conformity scores
        u_cal = self.uncertainty_heuristic(
            X_cal,
            X_unc=X_cal_unc,
            dtype=dtype,
            max_batch_size=max_batch_size,
            rng=rng,
            update_state=update_state,
            fwd=fwd,
        )

        if Y_cal_pred is None:
            Y_cal_pred = self.model(
                X_cal,
                dtype=dtype,
                rng=rng,
                return_state=False,
                update_state=update_state,
                max_batch_size=max_batch_size,
                verbose=verbose,
            )

        # bias correction, if applicable
        if not isinstance(
            self._stratified_bias_corrections, bool
        ) and tree_util.any(self._stratified_bias_corrections):

            Y_cal_pred = tree_util.add(
                Y_cal_pred,
                jax.tree.map(
                    lambda bias: bias[strata_idxs],
                    self._strata_biases,
                ),
            )
        elif (
            isinstance(self._stratified_bias_corrections, bool)
            and self._stratified_bias_corrections
        ):
            Y_cal_pred = tree_util.add(
                Y_cal_pred,
                self._strata_biases,
            )

        Y_cal_residuals = tree_util.sub(
            Y_cal if Y_cal is not None else X_cal, Y_cal_pred
        )

        # s = |Y - Y_pred| / u

        self.scores = tree_util.div(tree_util.abs(Y_cal_residuals), u_cal)

    def parameter_sensitivity(
        self,
        X: Data,
        dtype: jax.typing.DTypeLike = np.float64,
        max_batch_size: int | None = None,
        rng: Any | int | None = None,
        update_state: bool = False,
        fwd: bool | None = None,
        normalize: bool = True,
        verbose: bool = False,
    ) -> Data | None:
        if not self._parameter_sensitivity:
            return None

        if normalize and self._norm_S_params is None:
            raise ValueError(
                "MAD normalization factor for parameter sensitivity is not"
                " computed. Please calibrate the model first to compute it."
            )

        n_samples = jax.tree.leaves(X)[0].shape[0]

        # df/dTheta
        # will have the composite structure where the upper level structure is
        # the same as the model's output structure and the lower level
        # is the structure of the model's parameters. Each leaf will have shape
        # (n_samples, <output_shape>, <param_shape>)
        df_dparams = self.model.grad_params(
            X,
            dtype=dtype,
            max_batch_size=max_batch_size,
            rng=rng,
            return_state=False,
            update_state=update_state,
            fwd=fwd,
            verbose=verbose,
        )

        # l2 norm over params, scaled by abs(params), divided by number of
        # params
        # dy_params = || df/dTheta * |Theta| ||_2^2
        # ie sum over all params of |df/dTheta_i * |Theta_i||^2 / size(Theta)
        params = self.model.get_params()
        # zero-out the parameters that we are not including in the sensitivity
        params = jax.tree.map(
            lambda p, s: p if s else np.zeros_like(p),
            params,
            self._parameter_sensitivity,
        )

        # TODO: u_dtype should match the output dtype of the model, not the
        # gradient
        u_dtype = dtype

        # total uncertainty from the sensitivity to parameters
        dy_params = tree_util.scalar_mul(
            jax.tree.map(
                lambda y, df: map_over_output_leaves(
                    n_samples, u_dtype, y, df, params
                ),
                self.model.output_shape,
                df_dparams,
                is_leaf=tree_util.is_shape_leaf,
            ),
            1.0 / self._num_params_sensitivity,
        )

        if normalize:
            dy_params = tree_util.div(dy_params, self._norm_S_params)

        return dy_params

    def input_sensitivity(
        self,
        X: Data,
        X_unc: Data | None = None,
        dtype: jax.typing.DTypeLike = np.float64,
        max_batch_size: int | None = None,
        rng: Any | int | None = None,
        update_state: bool = False,
        fwd: bool | None = None,
        normalize: bool = True,
        verbose: bool = False,
    ) -> Data | None:
        if not self._input_sensitivity:
            return None
        if normalize and self._norm_S_input is None:
            raise ValueError(
                "MAD normalization factor for input sensitivity is not"
                " computed. Please calibrate the model first to compute it."
            )

        n_samples = jax.tree.leaves(X)[0].shape[0]

        df_dx = self.model.grad_input(
            X,
            dtype=dtype,
            max_batch_size=max_batch_size,
            rng=rng,
            fwd=fwd,
            update_state=update_state,
            return_state=False,
            verbose=verbose,
        )

        # TODO: u_dtype should match the output dtype of the model, not the
        # gradient
        u_dtype = dtype

        # l2 norm of df/dx scaled by IQR of training inputs
        # i.e. sum over all input features of |df/dx_i * IQR(x_i)|^2
        # divided by number of continuous features
        # IQR will be 0.0 for non-continuous features and features not included
        # in the input sensitivity, so they won't contribute to the
        # sensitivity, as desired
        dy_input = tree_util.scalar_mul(
            jax.tree.map(
                lambda y, df: map_over_output_leaves(
                    n_samples, u_dtype, y, df, self._iqr_X_train
                ),
                self.model.output_shape,
                df_dx,
                is_leaf=tree_util.is_shape_leaf,
            ),
            1.0 / self._num_input_sensitivity,
        )

        if X_unc is not None:

            # zero-out the uncertainties for features that we are not including
            # in the sensitivity
            X_unc = jax.tree.map(
                lambda x_u, cont, s: x_u if cont and s else np.zeros_like(x_u),
                X_unc,
                self._continuous_features,
                self._input_sensitivity,
            )

            # l2 norm of df/dx scaled by uncertainty in inputs
            # divided by number of continuous features
            dy_input_unc = tree_util.scalar_mul(
                jax.tree.map(
                    lambda y, df: map_over_output_leaves(
                        n_samples, u_dtype, y, df, X_unc
                    ),
                    self.model.output_shape,
                    df_dx,
                    is_leaf=tree_util.is_shape_leaf,
                ),
                1.0 / self._num_input_sensitivity,
            )

            dy_input = tree_util.add(dy_input, dy_input_unc)

        if normalize:
            dy_input = tree_util.div(dy_input, self._norm_S_input)

        return dy_input

    def distance_sensitivity(
        self,
        X: Data,
        dtype: jax.typing.DTypeLike = np.float64,
        normalize: bool = True,
    ) -> Real[Array, "..."] | None:
        if not self._distance_sensitivity:
            return None
        if self._training_data is None:
            raise ValueError(
                "Distance sensitivity cannot be computed without training"
                " data. Please provide training data during calibration."
            )

        if (
            self._kdtree is None
            or self._range_X_train is None
            or self._onehot_training_enc is None
            or self._distance_cutoff is None
        ):
            raise ValueError(
                "KDTree, range of training data, one-hot encoder, or distance"
                " cutoff for distance sensitivity are not computed. Please"
                " calibrate the model first to compute them."
            )

        if normalize and self._norm_S_dist is None:
            raise ValueError(
                "MAD normalization factor for distance sensitivity is not"
                " computed. Please calibrate the model first to compute it."
            )

        flat_data = self._prepare_data_for_kdtree(X)

        # query the KDTree for all points within the distance cutoff of each
        # point in the batch
        idxs = self._kdtree.query_ball_point(
            flat_data, r=self._distance_cutoff, p=1.0, workers=-1
        )
        # index the data to get the points
        neighbors = jax.tree.map(
            lambda i: self._kdtree.data[i], [np.array(idxs_) for idxs_ in idxs]
        )
        # mean L1 distance to neighbors
        mean_distances = np.array(
            jax.tree.map(
                lambda x, nbrs: (
                    np.mean(np.linalg.norm(x - nbrs, ord=1, axis=-1))
                    if nbrs.shape[0] > 0
                    else 0.0
                ),
                [x for x in flat_data],
                neighbors,
            )
        )
        # now we also compute the k=1 nearest neighbor distance, and take the
        # max of that and the mean distance, so points outside of the cutoff
        # distance (which have no neighbors and thus 0.0 distance) instead get
        # their distance to the nearest neighbor as their distance sensitivity
        nn_distances, _ = self._kdtree.query(flat_data, k=1, p=1.0, workers=-1)

        distances = np.maximum(mean_distances, np.array(nn_distances))

        # normalize by the number of features (original number, before one-hot
        # encoding)
        distances = distances / self._num_distance_sensitivity

        if normalize:
            distances = distances / self._norm_S_dist

        return distances[:, None]

    def _prepare_data_for_kdtree(
        self, X: Data
    ) -> Inexact[Array, " n_samples flat_features"]:

        if self._onehot_training_enc is None or self._range_X_train is None:
            raise ValueError(
                "One-hot encoder or range of training data for KDTree is not"
                " computed. Please calibrate the model first to compute them."
            )

        # first we need to encode and scale the input data in the same way as
        # the training data, so that the KDTree query is valid
        X_encoded = self._onehot_training_enc.transform(X)
        X_scaled = tree_util.div(X_encoded, self._range_X_train)

        # then we flatten the PyTree and all feature axes into a single axis
        # to end up with a 2D array of shape (n_samples, n_features_total)
        flat_data = np.concatenate(
            jax.tree.leaves(
                jax.tree.map(
                    lambda x: x.reshape((x.shape[0], -1)),
                    X_scaled,
                )
            ),
            axis=-1,
        )

        return flat_data

    def get_qhat(
        self,
        alpha: (
            float | List[float] | Tuple[float, ...] | Real[Array, " d"] | None
        ) = None,
    ) -> Data | List[Data]:

        if alpha is None:
            alpha = self.default_alpha

        if np.isscalar(alpha):
            alpha = np.array([alpha])
        if (
            isinstance(alpha, onp.ndarray)
            or isinstance(alpha, list)
            or isinstance(alpha, tuple)
        ):
            alpha = np.array(alpha)

        if np.any((alpha <= 0) | (alpha >= 1)):
            raise ValueError("alpha must be in the range (0, 1).")

        if self.scores is None:
            raise ValueError(
                "The model must be calibrated before computing "
                "the quantile. Please call the `calibrate` method."
            )

        # separate alphas into those already cached and new ones
        # evaluate all the new ones at once as an array, add them to the cache
        # then put them back in the original order to return
        new = ~np.isin(alpha, np.array(list(self.qhats.keys())))
        new_idxs = np.where(new)[0]
        cached_idxs = np.where(~new)[0]

        cached_alphas = alpha[cached_idxs]
        new_alphas = alpha[new_idxs]

        res_qhats = [None for _ in range(alpha.size)]
        for idx, a in zip(cached_idxs, cached_alphas):
            res_qhats[idx] = self.qhats[a.item()]

        if new_alphas.size > 0:
            n = jax.tree.leaves(self.scores)[0].shape[0]
            new_qhats = jax.tree.map(
                # TODO: is there a generalization of quantile for complex
                # numbers?
                lambda s: np.quantile(
                    s,
                    np.ceil((n + 1) * (1 - new_alphas)) / n,
                    axis=0,
                ),
                self.scores,
            )
            # new_qhats is a PyTree where the leading dimension of each leaf is
            # the number of new alphas, so we need to split it into a list of
            # PyTrees, one for each new alpha
            new_qhats_list = [
                jax.tree.map(lambda x: x[i], new_qhats)
                for i in range(new_alphas.size)
            ]
            for idx, a, qhat in zip(new_idxs, new_alphas, new_qhats_list):
                self.qhats[a.item()] = qhat
                res_qhats[idx] = qhat

        # res_qhats is a list of qhats in the same order as the input alphas,
        # so we return it as a list if the input was a list or array, but if
        # the input was a single alpha, we return a single qhat instead of a
        # list of one qhat
        if len(res_qhats) == 1:
            return res_qhats[0]
        else:
            return res_qhats

    def __call__(
        self,
        X: Data,
        X_unc: Data | None = None,
        alpha: (
            float | List[float] | Tuple[float, ...] | Real[Array, " d"] | None
        ) = None,
        dtype: jax.typing.DTypeLike = np.float64,
        max_batch_size: int | None = None,
        rng: Any | int | None = None,
        update_state: bool = False,
        return_state: bool = False,
        fwd: bool | None = None,
        verbose: bool = False,
    ) -> (
        # single alpha case, with and without state
        Tuple[Data, Tuple[Data, Data]]
        | Tuple[Data, Tuple[Data, Data], ModelState]
        # multiple alpha case, with and without state
        | Tuple[Data, List[Tuple[Data, Data]]]
        | Tuple[Data, List[Tuple[Data, Data]], ModelState]
    ):
        r"""
        <DOC TODO>
        """

        if alpha is None:
            alpha = self.default_alpha

        if np.isscalar(alpha):
            alpha = np.array([alpha])
        if (
            isinstance(alpha, onp.ndarray)
            or isinstance(alpha, list)
            or isinstance(alpha, tuple)
        ):
            alpha = np.array(alpha)

        if np.any((alpha <= 0) | (alpha >= 1)):
            raise ValueError("alpha must be in the range (0, 1).")

        if not tree_util.all_equal(
            self.model.input_shape, tree_util.get_shapes(X, slice(1, None))
        ):
            raise ValueError(
                "The input shape of the model must match the shape of the"
                f" input features. Got {self.model.input_shape} and"
                f" {tree_util.get_shapes(X, slice(1, None))}."
            )
        if self.scores is None:
            raise ValueError(
                "The model must be calibrated before making "
                "predictions. Please call the `calibrate` method."
            )

        Y_pred, state = self.model(
            X,
            dtype=dtype,
            rng=rng,
            return_state=True,
            update_state=update_state,
            max_batch_size=max_batch_size,
            verbose=verbose,
        )

        if not isinstance(
            self._stratified_bias_corrections, bool
        ) and tree_util.any(self._stratified_bias_corrections):
            if (
                self._strata_biases is None
                or self._strata_unique_values is None
            ):
                raise ValueError(
                    "Stratified bias corrections are enabled, but strata"
                    " biases or unique values are not computed. Please"
                    " calibrate the model first to compute them."
                )

            # flatten the stratified features to match with the unique values
            # we computed during calibration
            X_strata_features_flat = np.concatenate(
                jax.tree.leaves(
                    jax.tree.map(
                        lambda x, sbc: (
                            x.reshape(x.shape[0], -1)
                            if sbc
                            else np.empty((x.shape[0], 0), dtype=x.dtype)
                        ),
                        X,
                        self._stratified_bias_corrections,
                    )
                ),
                axis=-1,
            )

            # partition X and Y by the stratification features
            matches = np.all(
                X_strata_features_flat[:, None, :]
                == self._strata_unique_values[None, :, :],
                axis=-1,
            )
            found = np.any(matches, axis=-1)
            all_idxs = np.argmax(matches, axis=-1)
            # for k unique values, any point that doesn't match any of them
            # gets assigned to index k, which has 0.0 bias
            idxs = np.where(
                found, all_idxs, self._strata_unique_values.shape[0]
            )
            biases = jax.tree.map(
                lambda bias: bias[idxs],
                self._strata_biases,
            )

            # add the bias (median residual)
            Y_pred = tree_util.add(Y_pred, biases)

        elif (
            isinstance(self._stratified_bias_corrections, bool)
            and self._stratified_bias_corrections
        ):
            if self._strata_biases is None:
                raise ValueError(
                    "Global bias correction is enabled, but bias is not"
                    " computed. Please calibrate the model first to compute"
                    " it."
                )

            Y_pred = tree_util.add(Y_pred, self._strata_biases)

        u_x = self.uncertainty_heuristic(
            X,
            X_unc=X_unc,
            dtype=dtype,
            max_batch_size=max_batch_size,
            rng=rng,
            update_state=update_state,
            fwd=fwd,
            verbose=verbose,
        )

        qhat = self.get_qhat(alpha=alpha)

        # lower = Y_pred - qhat * u_x
        # upper = Y_pred + qhat * u_x

        if alpha.size == 1:
            lower = tree_util.sub(Y_pred, tree_util.mul(qhat, u_x))
            upper = tree_util.add(Y_pred, tree_util.mul(qhat, u_x))

            if return_state:
                return Y_pred, (lower, upper), state
            else:
                return Y_pred, (lower, upper)

        else:
            lowers = []
            uppers = []
            for q in qhat:
                lowers.append(tree_util.sub(Y_pred, tree_util.mul(q, u_x)))
                uppers.append(tree_util.add(Y_pred, tree_util.mul(q, u_x)))

            if return_state:
                return Y_pred, list(zip(lowers, uppers)), state
            else:
                return Y_pred, list(zip(lowers, uppers))

    predict = __call__  # alias

    def uncertainty_heuristic(
        self,
        X: Data,
        X_unc: Data | None = None,
        dtype: jax.typing.DTypeLike = np.float64,
        max_batch_size: int | None = None,
        rng: Any | int | None = None,
        update_state: bool = False,
        fwd: bool | None = None,
        verbose: bool = False,
    ) -> Data:
        r"""
        The uncertainty heuristic used to compute conformity scores for
        conformal prediction.

        Parameters
        ----------
        X
            The input features where the uncertainty heuristic should be
            evaluated. Shape ``(n_samples, <input_shape>)``.
        X_unc
            The uncertainty in the input features. Shape
            ``(n_samples, <input_shape>)``.
            If ``None``, the uncertainty in the input features is assumed to be
            zero. Default is ``None``.
        dtype
            Data type to use for the forward pass. Default is
            ``jax.numpy.float64``. It is strongly recommended to perform
            training in single precision (``float32`` and ``complex64``) and
            inference with double precision inputs (``float64``, the default
            here) with single precision weights.
        max_batch_size
            The maximum batch size to use when computing the uncertainty
            heuristic. Used to avoid OOM errors. If ``None``, the entire
            dataset is processed at once. Default is ``None``.

        Returns
        -------
        The uncertainty heuristic evaluated at the input features. Shape
        ``(n_samples, <output_shape>)``.

        See Also
        --------
        Model.__call__
            Evaluation of the wrapped ``Model``.
        Model.grad_params
            Gradient of the wrapped ``Model`` with respect to its parameters.
        Model.grad_input
            Gradient of the wrapped ``Model`` with respect to its inputs.
        """

        # input validation
        # ensure X has uniform leading axis
        if not tree_util.has_uniform_leaf_shapes(X, axis=0):
            raise ValueError(
                "X must have uniform leading axis (axis 0) across all "
                "leaves of the pytree."
            )

        if X_unc is not None:
            if not tree_util.shapes_equal(X, X_unc):
                raise ValueError(
                    "X and X_unc must have the same shape. Got"
                    f" {tree_util.get_shapes(X)} and"
                    f" {tree_util.get_shapes(X_unc)}."
                )

        # ensure the input shape matches the model's expected input shape
        if not tree_util.all_equal(
            self.model.input_shape, tree_util.get_shapes(X, slice(1, None))
        ):
            raise ValueError(
                "The input shape of the model must match the shape of the"
                f" input features. Got {self.model.input_shape} and"
                f" {tree_util.get_shapes(X, slice(1, None))}."
            )

        # compute each of the sensitivities
        n_samples = jax.tree.leaves(X)[0].shape[0]

        u = jax.tree.map(
            lambda s: np.zeros((n_samples, *s), dtype=dtype),
            self.model.output_shape,
            is_leaf=tree_util.is_shape_leaf,
        )

        if tree_util.any(self._parameter_sensitivity):
            u_param = self.parameter_sensitivity(
                X,
                dtype=dtype,
                max_batch_size=max_batch_size,
                rng=rng,
                update_state=update_state,
                fwd=fwd,
                normalize=True,
                verbose=verbose,
            )
            u = tree_util.add(u, u_param)

        if self._training_data is not None:
            if tree_util.any(self._input_sensitivity):
                u_input = self.input_sensitivity(
                    X,
                    X_unc=X_unc,
                    dtype=dtype,
                    max_batch_size=max_batch_size,
                    rng=rng,
                    update_state=update_state,
                    fwd=fwd,
                    normalize=True,
                    verbose=verbose,
                )
                u = tree_util.add(u, u_input)

            if self._distance_sensitivity:
                u_dist = self.distance_sensitivity(
                    X,
                    dtype=dtype,
                    normalize=True,
                )

                u = jax.tree.map(lambda x: x + u_dist, u)

        return u

    def serialize(self) -> Dict[str, Serializable]:
        r"""
        Serialize the conformalized model to a dictionary.

        Returns
        -------
        A dictionary containing the serialized conformalized model.

        See Also
        --------
        Model.serialize
            Serialization of the wrapped ``Model``.
        BaseModule.serialize
            Serialization of the wrapped ``BaseModule``.
        ConformalizedModel.deserialize
            Deserialization of a conformalized model from a dictionary.
        """

        # save the package version, the serialized model, and the serialized
        # state of the conformalization
        package_version = pmm.__version__
        model_typename = self.model.__class__.__name__
        model_fulltypename = str(type(self.model))
        model_module = self.model.__module__
        serialized_model = self.model.serialize()

        # KDTree will have to be reconstructed during deserialization from the
        # saved training data

        conformalization_state = {
            "default_alpha": self.default_alpha,
            "nn_dist_quantile": self.nn_dist_quantile,
            "scores": self.scores,
            "qhats": self.qhats,
            "_training_data": self._training_data,
            "_iqr_X_train": self._iqr_X_train,
            "_range_X_train": self._range_X_train,
            "_norm_S_params": self._norm_S_params,
            "_norm_S_input": self._norm_S_input,
            "_norm_S_dist": self._norm_S_dist,
            "_continuous_features": self._continuous_features,
            "_distance_cutoff": self._distance_cutoff,
            "_onehot_training_enc": (
                self._onehot_training_enc.serialize()
                if self._onehot_training_enc is not None
                else None
            ),
            "_stratified_bias_corrections": self._stratified_bias_corrections,
            "_strata_unique_values": self._strata_unique_values,
            "_strata_biases": self._strata_biases,
            "_num_params_sensitivity": self._num_params_sensitivity,
            "_num_input_sensitivity": self._num_input_sensitivity,
            "_num_distance_sensitivity": self._num_distance_sensitivity,
            "_input_sensitivity": self._input_sensitivity,
            "_parameter_sensitivity": self._parameter_sensitivity,
            "_distance_sensitivity": self._distance_sensitivity,
        }

        return {
            "version": self.__version__,
            "package_version": package_version,
            "model_typename": model_typename,
            "model_fulltypename": model_fulltypename,
            "model_module": model_module,
            "model": serialized_model,
            "conformalization_state": conformalization_state,
        }

    @classmethod
    def deserialize(
        cls,
        data: Dict[str, Serializable],
        /,
        *,
        strict_package_version: bool = False,
    ) -> ConformalizedModel:
        r"""
        Deserialize a conformalized model from a dictionary.

        Parameters
        ----------
        data
            A dictionary containing the serialized conformalized model.

        Returns
        -------
        A ``ConformalizedModel`` instance.
        """

        # read the version of the package this model was serialized with
        current_version = parse(pmm.__version__)
        package_version = parse(str(data["package_version"]))

        if current_version != package_version:
            if strict_package_version:
                raise ValueError(
                    "Model was saved with package version "
                    f"{package_version}, but current package version is "
                    f"{current_version}."
                )

        # read the conformal mode version and handle and backward compatibility
        # if needed
        confmodel_version = parse(str(data["version"]))
        if confmodel_version != parse(cls.__version__):
            # nothing to do yet
            pass

        # deserialize the wrapped model
        model_module = data["model_module"]
        model_typename = data["model_typename"]
        model = getattr(sys.modules[model_module], model_typename)()
        model.deserialize(data["model"])

        # deserialize the conformalization state
        conf_state = data["conformalization_state"]
        default_alpha = conf_state["default_alpha"]
        nn_dist_quantile = conf_state["nn_dist_quantile"]

        conf_model = cls(
            model=model,
            alpha=default_alpha,
            nn_dist_quantile=nn_dist_quantile,
        )

        conf_model.scores = conf_state["scores"]
        conf_model.qhats = conf_state["qhats"]
        conf_model._training_data = conf_state["_training_data"]
        conf_model._iqr_X_train = conf_state["_iqr_X_train"]
        conf_model._range_X_train = conf_state["_range_X_train"]
        conf_model._norm_S_params = conf_state["_norm_S_params"]
        conf_model._norm_S_input = conf_state["_norm_S_input"]
        conf_model._norm_S_dist = conf_state["_norm_S_dist"]
        conf_model._continuous_features = conf_state["_continuous_features"]
        conf_model._distance_cutoff = conf_state["_distance_cutoff"]
        onehot_enc_data = conf_state["_onehot_training_enc"]
        conf_model._onehot_training_enc = (
            preprocessing.MixedScaler.deserialize(onehot_enc_data)
            if onehot_enc_data is not None
            else None
        )
        conf_model._stratified_bias_corrections = conf_state[
            "_stratified_bias_corrections"
        ]
        conf_model._strata_unique_values = conf_state["_strata_unique_values"]
        conf_model._strata_biases = conf_state["_strata_biases"]
        conf_model._num_params_sensitivity = conf_state[
            "_num_params_sensitivity"
        ]
        conf_model._num_input_sensitivity = conf_state[
            "_num_input_sensitivity"
        ]
        conf_model._num_distance_sensitivity = conf_state[
            "_num_distance_sensitivity"
        ]
        conf_model._input_sensitivity = conf_state["_input_sensitivity"]
        conf_model._parameter_sensitivity = conf_state[
            "_parameter_sensitivity"
        ]
        conf_model._distance_sensitivity = conf_state["_distance_sensitivity"]

        # if the training data is not None, rebuild the KDTree
        if conf_model._training_data is not None:
            flat_data = conf_model._prepare_data_for_kdtree(
                conf_model._training_data
            )
            conf_model._kdtree = KDTree(flat_data)

        return conf_model

    def save(self, file: str | IO | Path, /) -> None:
        r"""
        Save the conformalized model to a file.

        Parameters
        ----------
        file
            The name of the file to save the conformalized model to.
        """

        data: Dict[str, Serializable] = self.serialize()
        if isinstance(file, str):
            file = file if file.endswith(".npz") else file + ".npz"

        np.savez(file, **data)

    def save_compressed(self, file: str | IO | Path, /) -> None:
        r"""
        Save the conformalized model to a compressed file.

        Parameters
        ----------
        file
            The name of the file to save the conformalized model to.
        """

        data: Dict[str, Serializable] = self.serialize()
        if isinstance(file, str):
            file = file if file.endswith(".npz") else file + ".npz"

        # jax numpy doesn't have savez_compressed, so we use numpy
        onp.savez_compressed(file, **data)

    @classmethod
    def from_file(
        cls,
        file: str | IO | Path,
        /,
        *,
        strict_package_version: bool = False,
    ) -> ConformalizedModel:
        r"""
        Load a conformalized model from a file.

        Parameters
        ----------
        file
            The name of the file to load the conformalized model from.

        Returns
        -------
        A ``ConformalizedModel`` instance.
        """

        data = np.load(file, allow_pickle=True)

        # convert npz object to dict
        data = {key: data[key] for key in data.files}

        # convert certain items back to lists or bare values
        if isinstance(data["model"], (onp.ndarray, np.ndarray)):
            data["model"] = data["model"].item()
        if isinstance(data["model_typename"], (onp.ndarray, np.ndarray)):
            data["model_typename"] = data["model_typename"].item()
        if isinstance(data["model_fulltypename"], (onp.ndarray, np.ndarray)):
            data["model_fulltypename"] = data["model_fulltypename"].item()
        if isinstance(data["model_module"], (onp.ndarray, np.ndarray)):
            data["model_module"] = data["model_module"].item()
        if isinstance(
            data["conformalization_state"], (onp.ndarray, np.ndarray)
        ):
            data["conformalization_state"] = data[
                "conformalization_state"
            ].item()

        return cls.deserialize(
            data, strict_package_version=strict_package_version
        )


def weighted_quantile(
    data: Inexact[Array, "..."],
    weights: Inexact[Array, "..."],
    axis: int = 0,
    quantiles: Real[Array, " q"] | float = 0.5,
) -> Inexact[Array, "..."]:
    idxs = np.argsort(weights, axis=axis)
    cum_weights = np.cumsum(weights[idxs], axis=axis)
    w_qs = np.interp(quantiles * cum_weights[~0], cum_weights, data[idxs])
    return w_qs


def mad(
    data: Inexact[Array, "..."],
    weights: Inexact[Array, "..."] | None = None,
    axis: int = 0,
) -> Inexact[Array, "..."]:
    if weights is None:
        median = np.median(data, axis=axis)
        mad = np.median(np.abs(data - median), axis=axis)
    else:
        median = weighted_quantile(data, weights, axis=axis, quantiles=0.5)
        mad = weighted_quantile(
            np.abs(data - median), weights, axis=axis, quantiles=0.5
        )
    return mad


def robust_normalization(
    data: Inexact[Array, "..."],
    weights: Inexact[Array, "..."] | None = None,
    eps: float = 1e-12,
) -> Inexact[Array, "..."]:

    norm = mad(data, weights, axis=0)

    if np.any(norm < eps):
        warnings.warn(f"MAD < {eps}. Falling back to IQR.")

        if weights is None:
            qs = np.quantile(
                data,
                np.array([0.25, 0.75]),
                axis=0,
            )
        else:
            qs = weighted_quantile(
                data,
                weights,
                axis=0,
                quantiles=np.array([0.25, 0.75]),
            )
        norm = qs[1] - qs[0]

    if np.any(norm < eps):
        warnings.warn(f"IQR < {eps}. Falling back to 5-95 percentile range.")
        if weights is None:
            qs = np.quantile(
                data,
                np.array([0.05, 0.95]),
                axis=0,
            )
        else:
            qs = weighted_quantile(
                data,
                weights,
                axis=0,
                quantiles=np.array([0.05, 0.95]),
            )
        norm = qs[1] - qs[0]

    if np.any(norm < eps):
        warnings.warn(
            f"5-95 percentile range < {eps}. Falling back to max-min range."
        )
        norm = np.ptp(data, axis=0)

    if np.any(norm < eps):
        warnings.warn(f"Max-min range < {eps}. Falling back to inf.")
        norm = np.full_like(norm, np.inf)

    return norm


def pytree_robust_normalization(
    data: PyTree[Inexact[Array, "..."], " In"],
    weights: PyTree[Inexact[Array, "..."], " In"] | None = None,
    eps: float = 1e-12,
) -> PyTree[Inexact[Array, "..."], " Out"]:
    if weights is not None:
        if not tree_util.shapes_equal(data, weights):
            raise ValueError(
                "Data and weights must have the same shape. Got"
                f" {tree_util.get_shapes(data)} and"
                f" {tree_util.get_shapes(weights)}."
            )
        return jax.tree.map(
            lambda d, w: robust_normalization(d, w, eps), data, weights
        )
    else:
        return jax.tree.map(lambda d: robust_normalization(d, None, eps), data)


# data class for zipped grad and input (so that it becomes a leaf in PyTrees)
@dataclass
class GradAndInput:
    grad: Inexact[Array, "..."]
    x: Inexact[Array, "..."]

    def __iter__(self):
        yield self.grad
        yield self.x


# we map over the output leaves and reduce over the input leaves
@jaxtyped(typechecker=beartype)
def reduce_over_input_leaves(
    out_shape: Tuple[int, ...],
    df_leaf_carry: Inexact[Array, "..."],
    df_leaf_dx_and_x: GradAndInput,
) -> Inexact[Array, "..."]:
    # df_leaf_carry is the accumulated sum over x leaves which has
    # shape (n_samples, <output_shape>)
    # df_leaf_dx_and_x is a GradAndInput object containing
    # (df_leaf_dx, x) for a single x leaf
    df_leaf_dx, x = df_leaf_dx_and_x
    # df_leaf_dx has shape (n_samples, <output_shape>, <x_shape>)
    # x has shape (<x_shape>,)
    # we scale df_leaf_dx by abs(x)
    scaled = df_leaf_dx * np.abs(x)
    # reshape to (n_samples, <output_shape>, -1)
    scaled_reshaped = scaled.reshape((scaled.shape[0],) + out_shape + (-1,))
    # and compute the squared abs sum over the last axis (all xs)
    sum_squares = np.sum(np.abs(scaled_reshaped) ** 2, axis=-1)
    # accumulate
    return df_leaf_carry + sum_squares


@jaxtyped(typechecker=beartype)
def map_over_output_leaves(
    n_samples: int,
    u_dtype: jax.typing.DTypeLike,
    y_leaf_shape: Tuple[int, ...],
    df_leaf_dxs: PyTree[Inexact[Array, "..."], " In"],
    xs: PyTree[Inexact[Array, "..."], " In"],
) -> Data:
    # y_leaf_shape is the model output shape for the given leaf
    # which is <output_shape>
    # df_leaf_dxs is a PyTree with the same structure as xs
    # representing the gradient of this particular output leaf w.r.t.
    # each x leaf. each leaf of df_leaf_dxs has shape
    # (n_samples, <output_shape>, <x_shape>)
    # where <x_shape> varies per leaf
    # xs is a PyTree with the same structure as df_leaf_dxs
    # and each leaf has shape <x_shape>, matching the corresponding
    # leaf of df_leaf_dxs

    # now we zip (map) together the leaves of df_leaf_dxs and
    # xs so that we can call reduce over them (reduce only takes a
    # single tree, unlike map)
    df_leaf_dx_and_xs = jax.tree.map(
        lambda d, p: GradAndInput(d, p), df_leaf_dxs, xs
    )
    # now we reduce over the x leaves
    dy_leaf = jax.tree.reduce(
        lambda acc, dfdx: reduce_over_input_leaves(y_leaf_shape, acc, dfdx),
        df_leaf_dx_and_xs,
        initializer=np.zeros((n_samples,) + y_leaf_shape, dtype=u_dtype),
    )
    return dy_leaf
