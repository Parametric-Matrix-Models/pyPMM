from __future__ import annotations

from typing import Tuple

import jax.numpy as np
import numpy as onp
from packaging.version import parse

import parametricmatrixmodels as pmm

from .model import Model


class ConformalizedModel(object):
    r"""
    A wrapper class for conformalized prediction models.

    Confidence intervals (uncertainty quantification) is optionally provided by
    conformal prediction methods. A heuristic notion of uncertainty is
    transformed into a rigorous statistical guarantee on the coverage of the
    prediction intervals via a calibration dataset. See [1]_ for an overview of
    conformal prediction methods.

    For now, the uncertainty heuristic is fixed and is:

    .. math::

        u(x) &= \left\Vert\frac{\partial f}{\partial\Theta}(x)
                    \otimes \left|\Theta\right|\right\Vert_2^2 \\
             &\quad + \left\Vert\frac{\partial f}{\partial x}(x)
                    \otimes
                    \sigma\left(x_\mathrm{train}\right)\right\Vert_2^2\\
            &\quad + \left\Vert\frac{\partial f}{\partial x}(x)
                    \otimes \Delta x\Vert_2^2

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
        A model formed from modules that can be trained and evaluated.
    """

    def __init__(
        self,
        model: Model,
        alpha: float = 1 - 0.682689492137,
        additional_data: dict = None,
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
        """

        self.model = model
        self.scores = None
        self.additional_data = additional_data or {}
        self.qhats = {}  # dict of quantiles for different alphas
        self.default_alpha = alpha

    def calibrate(
        self,
        X_cal: np.ndarray,
        Y_cal: np.ndarray,
        X_cal_unc: np.ndarray = None,
        alpha: float = None,
        dtype: np.dtype = np.float64,
        max_batch_size: int = None,
        rng: np.ndarray = None,
        update_state: bool = False,
        fwd: bool = None,
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
        if X_cal.shape[0] != Y_cal.shape[0]:
            raise ValueError(
                "X_cal and Y_cal must have the same number of samples. "
                f"Got {X_cal.shape[0]} and {Y_cal.shape[0]}."
            )
        if X_cal_unc is not None:
            if X_cal.shape != X_cal_unc.shape:
                raise ValueError(
                    "X_cal and X_cal_unc must have the same shape. "
                    f"Got {X_cal.shape} and {X_cal_unc.shape}."
                )
        if self.model.input_shape != X_cal.shape[1:]:
            raise ValueError(
                "The input shape of the model must match the shape of "
                "the calibration features. "
                f"Got {self.model.input_shape} and {X_cal.shape[1:]}."
            )
        if self.model.output_shape != Y_cal.shape[1:]:
            raise ValueError(
                "The output shape of the model must match the shape of "
                "the calibration targets. "
                f"Got {self.model.output_shape} and {Y_cal.shape[1:]}."
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

        self.scores = np.abs(Y_cal - Y_cal_pred) / u_cal

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
            n = self.scores.shape[0]
            qhat = np.quantile(
                self.scores,
                np.ceil((n + 1) * (1 - alpha)) / n,
            )
            self.qhats[alpha] = qhat

        return self.qhats[alpha]

    def __call__(
        self,
        X: np.ndarray,
        X_unc: np.ndarray = None,
        alpha: float = None,
        dtype: np.dtype = np.float64,
        max_batch_size: int = None,
        rng: np.ndarray = None,
        update_state: bool = False,
        return_state: bool = False,
        fwd: bool = None,
    ) -> (
        Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]
        | Tuple[
            np.ndarray, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, ...]
        ]
    ):
        r"""
        <DOC TODO>
        """

        if alpha is None:
            alpha = self.default_alpha

        if not (0 < alpha < 1):
            raise ValueError("alpha must be in the range (0, 1).")

        if self.model.input_shape != X.shape[1:]:
            raise ValueError(
                "The input shape of the model must match the shape of "
                "the input features. "
                f"Got {self.model.input_shape} and {X.shape[1:]}."
            )
        if self.scores is None:
            raise ValueError(
                "The model must be calibrated before making "
                "predictions. Please call the `calibrate` method."
            )

        if return_state:
            Y_pred, state = self.model(
                X,
                dtype=dtype,
                rng=rng,
                return_state=return_state,
                update_state=update_state,
                max_batch_size=max_batch_size,
            )
        else:
            Y_pred = self.model(
                X,
                dtype=dtype,
                rng=rng,
                return_state=return_state,
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

        lower = Y_pred - qhat * u_x
        upper = Y_pred + qhat * u_x

        if return_state:
            return Y_pred, (lower, upper), state
        else:
            return Y_pred, (lower, upper)

    def _uncertainty_heuristic(
        self,
        X: np.ndarray,
        X_unc: np.ndarray = None,
        dtype: np.dtype = np.float64,
        max_batch_size: int = None,
        rng: np.ndarray = None,
        update_state: bool = False,
        fwd: bool = None,
    ) -> np.ndarray:
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
        if X_unc is not None:
            if X.shape != X_unc.shape:
                raise ValueError(
                    "X and X_unc must have the same shape. "
                    f"Got {X.shape} and {X_unc.shape}."
                )

        if self.additional_data.get("std_X_train") is None:
            raise ValueError(
                "The standard deviation of the training inputs "
                "is required for the uncertainty heuristic. "
                "Please provide it via the `additional_data` "
                "attribute. Or via the `calibrate` method."
            )

        std_X_train = self.additional_data["std_X_train"]

        if std_X_train.shape != X.shape[1:]:
            raise ValueError(
                "The standard deviation of the training inputs "
                "must have the same shape as the input features. "
                f"Got {std_X_train.shape} and {X.shape[1:]}."
            )

        output_shape = self.model.output_shape

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
        params = self.model.get_params()
        dy_dparams = np.array(
            [
                np.linalg.norm(
                    (dp * np.abs(p)).reshape(
                        (dp.shape[0],) + output_shape + (-1,)
                    ),
                    ord=2,
                    axis=-1,
                )
                ** 2
                for dp, p in zip(df_dparams, params)
            ]
        ).sum(
            axis=0
        )  # shape (n_samples, <output_shape>)

        df_dx = self.model.grad_input(
            X,
            dtype=dtype,
            max_batch_size=max_batch_size,
            rng=rng,
            fwd=fwd,
            batched=(max_batch_size != 1),
            update_state=update_state,
            return_state=False,
        )

        # l2 norm of df/dx scaled by std of training inputs
        dy_dx_train = (
            np.linalg.norm(
                (df_dx * std_X_train).reshape(
                    (df_dx.shape[0],) + output_shape + (-1,)
                ),
                ord=2,
                axis=-1,
            )
            ** 2
        )  # shape (n_samples, <output_shape>)

        u_x = dy_dparams + dy_dx_train

        if X_unc is not None:
            # l2 norm of df/dx scaled by uncertainty in inputs
            dy_dx_unc = (
                np.linalg.norm(
                    (df_dx * np.abs(X_unc)).reshape(
                        (df_dx.shape[0],) + output_shape + (-1,)
                    ),
                    ord=2,
                    axis=-1,
                )
                ** 2
            )

            u_x = u_x + dy_dx_unc

        u_x = np.sqrt(u_x)

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
