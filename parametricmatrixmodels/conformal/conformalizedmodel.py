from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as np
from beartype import beartype
from jaxtyping import jaxtyped

from .. import tree_util
from ..model import Model
from ..model_util import ModelState
from ..tree_util import (
    all_equal,
    get_shapes,
    has_uniform_leaf_shapes,
    is_shape_leaf,
    shapes_equal,
    uniform_leaf_shapes_equal,
)
from ..typing import (
    Any,
    Array,
    Data,
    Dict,
    Inexact,
    PyTree,
    Tuple,
)


class ConformalizedModel(object):
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

    For now, the uncertainty heuristic is fixed and is:

    .. math::

        u(x)^2 &= \left\Vert\frac{\partial f}{\partial\Theta}(x)
                    \odot \left|\Theta\right|\right\Vert_2^2 \\
             &\quad + \left\Vert\frac{\partial f}{\partial x}(x)
                    \odot
                    \sigma\left(x_\mathrm{train}\right)\right\Vert_2^2\\
            &\quad + \left\Vert\frac{\partial f}{\partial x}(x)
                    \odot \Delta x\right\Vert_2^2

    where :math:`f` is the prediction function, :math:`\Theta` are the
    trainable parameters of the model, :math:`\sigma(x_\mathrm{train})` is the
    standard deviation of the training inputs, and :math:`\Delta x` is the
    uncertainty in the input features, if any. This heuristic accounts for the
    sensitivity of the model both to the trainable parameters, training data,
    and input uncertainty.

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
        additional_data: Dict[str, Any] | None = None,
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
        additional_data
            A dictionary of additional data required for the uncertainty
            heuristic. Currently, the standard deviation of the training inputs
            must be provided with the key ``"std_X_train"``. Default is
            ``None``.
        """

        self.model = model
        self.scores = None
        self.additional_data = additional_data or {}
        self.qhats = {}  # dict of quantiles for different alphas
        self.default_alpha = alpha

    def calibrate(
        self,
        X_cal: Data,
        Y_cal: Data | None = None,
        X_cal_unc: Data | None = None,
        alpha: float | None = None,
        dtype: jax.typing.DTypeLike = np.float64,
        max_batch_size: int | None = None,
        rng: Any | int | None = None,
        update_state: bool = False,
        fwd: bool | None = None,
    ) -> float:
        r"""
        Calibrate the conformal prediction model on a calibration dataset.
        Parameters
        ----------
        <TODO>

        """

        if alpha is None:
            alpha = self.default_alpha
        if not (0 < alpha < 1):
            raise ValueError("alpha must be in the range (0, 1).")
        if Y_cal is not None:
            # supervised learning
            if not uniform_leaf_shapes_equal(X_cal, Y_cal, axis=0):
                raise ValueError(
                    "X_cal and Y_cal must have the same number of samples. "
                    "I.e. the leading dimension (axis 0) across all leaves "
                    "of the pytrees must be the same."
                )
        if X_cal_unc is not None:
            if not shapes_equal(X_cal, X_cal_unc):
                raise ValueError(
                    "X_cal and X_cal_unc must have the same shape."
                )
        if not all_equal(
            self.model.input_shape, get_shapes(X_cal, slice(1, None))
        ):
            raise ValueError(
                "The input shape of the model must match the shape of the"
                f" calibration features. Got {self.model.input_shape} and"
                f" {get_shapes(X_cal, slice(1, None))}."
            )
        if Y_cal is not None and not all_equal(
            self.model.output_shape, get_shapes(Y_cal, slice(1, None))
        ):
            raise ValueError(
                "The output shape of the model must match the shape of the"
                f" calibration targets. Got {self.model.output_shape} and"
                f" {get_shapes(Y_cal, slice(1, None))}."
            )
        elif Y_cal is None and not all_equal(
            self.model.output_shape, get_shapes(X_cal, slice(1, None))
        ):
            raise ValueError(
                "For unsupervised calibration, the output shape of the model"
                " must match the shape of the calibration features. Got"
                f" {self.model.output_shape} and"
                f" {get_shapes(X_cal, slice(1, None))}."
            )

        # compute conformity scores
        u_cal = self._uncertainty_heuristic(
            X_cal,
            X_unc=X_cal_unc,
            dtype=dtype,
            max_batch_size=max_batch_size,
            rng=rng,
            update_state=update_state,
            fwd=fwd,
        )
        Y_cal_pred = self.model(
            X_cal,
            dtype=dtype,
            rng=rng,
            return_state=False,
            update_state=update_state,
            max_batch_size=max_batch_size,
        )

        # s = |Y - Y_pred| / u

        if Y_cal is None:
            self.scores = tree_util.div(
                tree_util.abs(tree_util.sub(X_cal, Y_cal_pred)), u_cal
            )
        else:
            self.scores = tree_util.div(
                tree_util.abs(tree_util.sub(Y_cal, Y_cal_pred)), u_cal
            )

    def get_qhat(
        self,
        alpha: float = None,
    ) -> float:

        alpha = alpha or self.default_alpha

        if not (0 < alpha < 1):
            raise ValueError("alpha must be in the range (0, 1).")

        if self.scores is None:
            raise ValueError(
                "The model must be calibrated before computing "
                "the quantile. Please call the `calibrate` method."
            )

        if self.qhats.get(alpha) is None:
            n = jax.tree.leaves(self.scores)[0].shape[0]
            qhat = jax.tree.map(
                # TODO: is there a generalization of quantile for complex
                # numbers?
                lambda s: np.quantile(
                    s,
                    np.ceil((n + 1) * (1 - alpha)) / n,
                    axis=0,
                ),
                self.scores,
            )
            self.qhats[alpha] = qhat

        return self.qhats[alpha]

    def __call__(
        self,
        X: Data,
        X_unc: Data | None = None,
        alpha: float | None = None,
        dtype: jax.typing.DTypeLike = np.float64,
        max_batch_size: int | None = None,
        rng: Any | int | None = None,
        update_state: bool = False,
        return_state: bool = False,
        fwd: bool | None = None,
    ) -> (
        Tuple[Data, Tuple[Data, Data]]
        | Tuple[Data, Tuple[Data, Data], ModelState]
    ):
        r"""
        <DOC TODO>
        """

        if alpha is None:
            alpha = self.default_alpha

        if not (0 < alpha < 1):
            raise ValueError("alpha must be in the range (0, 1).")

        if not all_equal(
            self.model.input_shape, get_shapes(X, slice(1, None))
        ):
            raise ValueError(
                "The input shape of the model must match the shape of the"
                f" input features. Got {self.model.input_shape} and"
                f" {get_shapes(X, slice(1, None))}."
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
        )

        u_x = self._uncertainty_heuristic(
            X,
            X_unc=X_unc,
            dtype=dtype,
            max_batch_size=max_batch_size,
            rng=rng,
            update_state=update_state,
            fwd=fwd,
        )

        qhat = self.get_qhat(alpha=alpha)

        # lower = Y_pred - qhat * u_x
        # upper = Y_pred + qhat * u_x
        lower = tree_util.sub(Y_pred, tree_util.mul(qhat, u_x))
        upper = tree_util.add(Y_pred, tree_util.mul(qhat, u_x))

        if return_state:
            return Y_pred, (lower, upper), state
        else:
            return Y_pred, (lower, upper)

    def _uncertainty_heuristic(
        self,
        X: Data,
        X_unc: Data | None = None,
        dtype: jax.typing.DTypeLike = np.float64,
        max_batch_size: int | None = None,
        rng: Any | int | None = None,
        update_state: bool = False,
        fwd: bool | None = None,
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
        if not has_uniform_leaf_shapes(X, axis=0):
            raise ValueError(
                "X must have uniform leading axis (axis 0) across all "
                "leaves of the pytree."
            )

        if X_unc is not None:
            if not shapes_equal(X, X_unc):
                raise ValueError(
                    "X and X_unc must have the same shape. "
                    f"Got {get_shapes(X)} and {get_shapes(X_unc)}."
                )

        if self.additional_data.get("std_X_train") is None:
            raise ValueError(
                "The standard deviation of the training inputs "
                "is required for the uncertainty heuristic. "
                "Please provide it via the `additional_data` "
                "attribute. Or via the `calibrate` method."
            )

        std_X_train = self.additional_data["std_X_train"]

        if not all_equal(
            get_shapes(std_X_train), get_shapes(X, slice(1, None))
        ):
            raise ValueError(
                "The standard deviation of the training inputs must have the"
                " same shape as the input features (excluding batch"
                f" dimension). Got {get_shapes(std_X_train)} and"
                f" {get_shapes(X, slice(1, None))}."
            )

        n_samples = jax.tree.leaves(X)[0].shape[0]
        output_shape = self.model.output_shape

        # to compute the sensitivity in the output to each of the input types
        # (features, inputs via training, and inputs via uncertainty), we need
        # map over the PyTree structure of the model's output and reduce over
        # the PyTree structure of the input (model params, input data, etc)

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
            scaled_reshaped = scaled.reshape(
                (scaled.shape[0],) + out_shape + (-1,)
            )
            # and compute the squared abs sum over the last axis (all xs)
            sum_squares = np.sum(np.abs(scaled_reshaped) ** 2, axis=-1)
            # accumulate
            return df_leaf_carry + sum_squares

        @jaxtyped(typechecker=beartype)
        def map_over_output_leaves(
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
                lambda acc, dfdx: reduce_over_input_leaves(
                    y_leaf_shape, acc, dfdx
                ),
                df_leaf_dx_and_xs,
                initializer=np.zeros(
                    (n_samples,) + y_leaf_shape, dtype=u_dtype
                ),
            )
            return dy_leaf

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
        )

        # l2 norm over params, scaled by abs(params)
        # dy_params = || df/dTheta * |Theta| ||_2^2
        # ie sum over all params of |df/dTheta_i * |Theta_i||^2
        params = self.model.get_params()

        # TODO: u_dtype should match the output dtype of the model, not the
        # gradient
        u_dtype = dtype

        # total uncertainty from the sensitivity to parameters
        dy_params = jax.tree.map(
            lambda y, df: map_over_output_leaves(u_dtype, y, df, params),
            output_shape,
            df_dparams,
            is_leaf=is_shape_leaf,
        )

        df_dx = self.model.grad_input(
            X,
            dtype=dtype,
            max_batch_size=max_batch_size,
            rng=rng,
            fwd=fwd,
            update_state=update_state,
            return_state=False,
        )

        # l2 norm of df/dx scaled by std of training inputs
        # i.e. sum over all input features of |df/dx_i * std_X_train_i|^2
        dy_train = jax.tree.map(
            lambda y, df: map_over_output_leaves(u_dtype, y, df, std_X_train),
            output_shape,
            df_dx,
            is_leaf=is_shape_leaf,
        )

        u_x = tree_util.add(dy_params, dy_train)

        if X_unc is not None:
            # l2 norm of df/dx scaled by uncertainty in inputs
            dy_input = jax.tree.map(
                lambda y, df: map_over_output_leaves(u_dtype, y, df, X_unc),
                output_shape,
                df_dx,
                is_leaf=is_shape_leaf,
            )

            u_x = tree_util.add(u_x, dy_input)

        u_x = tree_util.abs_sqrt(u_x)

        return u_x

    def serialize(self) -> dict:
        r"""
        Serialize the conformalized model to a dictionary.

        Returns
        -------
        A dictionary containing the serialized conformalized model.
        """

        raise NotImplementedError(
            "Serialization of ConformalizedModel is not yet implemented."
        )

    def deserialize(self, data: dict) -> "ConformalizedModel":
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

        raise NotImplementedError(
            "Deserialization of ConformalizedModel is not yet implemented."
        )

    def save(self, filename: str) -> None:
        r"""
        Save the conformalized model to a file.

        Parameters
        ----------
        filename
            The name of the file to save the conformalized model to.
        """

        raise NotImplementedError(
            "Saving of ConformalizedModel is not yet implemented."
        )

    def save_compressed(self, filename: str) -> None:
        r"""
        Save the conformalized model to a compressed file.

        Parameters
        ----------
        filename
            The name of the file to save the conformalized model to.
        """

        raise NotImplementedError(
            "Saving of ConformalizedModel is not yet implemented."
        )

    @classmethod
    def from_file(cls, filename: str) -> "ConformalizedModel":
        r"""
        Load a conformalized model from a file.

        Parameters
        ----------
        filename
            The name of the file to load the conformalized model from.

        Returns
        -------
        A ``ConformalizedModel`` instance.
        """

        raise NotImplementedError(
            "Loading of ConformalizedModel is not yet implemented."
        )


# data class for zipped grad and input (so that it becomes a leaf in PyTrees)
@dataclass
class GradAndInput:
    grad: Inexact[Array, "..."]
    x: Inexact[Array, "..."]

    def __iter__(self):
        yield self.grad
        yield self.x
