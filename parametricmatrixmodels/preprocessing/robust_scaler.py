from __future__ import annotations

import jax
import jax.numpy as np
from jaxtyping import Array, PyTree, Real

from ..typing import (
    Any,
    BatchlessRealDataFixed,
    Dict,
    List,
    RealDataFixed,
    Tuple,
)
from .scaler import Scaler


class RobustScaler(Scaler):
    r"""
    Scale features using statistics that are robust to outliers.

    This scaler optionally removes the median and scales the data according to
    the provided quantile range. Per-feature, the formula is:

    .. math::
        X' = \frac{X - \text{median}(X)}{\text{quantile}(X, q_{high}) -
        \text{quantile}(X, q_{low})}

    where :math:`q_{low}` and :math:`q_{high}` are the lower and upper
    quantiles specified by the user. Centering and scaling are both optional,
    and can be controlled by the `with_centering` and `with_scaling`
    parameters. By default, both are set to True.

    See Also
    --------
        sklearn.preprocessing.RobustScaler
            Roughly equivalent implementation in scikit-learn.
    """

    def __init__(
        self,
        quantile_range: (
            Real[Array, "2"]
            | Tuple[
                None
                | float
                | int
                | BatchlessRealDataFixed
                | PyTree[None | float | int, "RealDataFixed"],
                float
                | None
                | int
                | BatchlessRealDataFixed
                | PyTree[None | float | int, "RealDataFixed"],
            ]
            | List[
                float
                | None
                | int
                | BatchlessRealDataFixed
                | PyTree[None | float | int, "RealDataFixed"]
            ]
        ) = (0.25, 0.75),
        with_centering: bool | PyTree[bool, "RealDataFixed"] = True,
        with_scaling: bool | PyTree[bool, "RealDataFixed"] = True,
        eps: float = 1e-12,
    ):
        """
        Initializes the RobustScaler with the specified quantile range and
        options for centering and scaling.

        Parameters
        ----------
            quantile_range
                A tuple specifying the desired quantile range of each feature
                in the form (q_low, q_high). All values must be in [0.0, 1.0]
                or None. Can
                be a tuple of floats or a tuple of arrays. If arrays are
                provided, they must match the PyTree structure and shapes of
                the input data, excluding the leading batch dimension. Entries
                of None denote that the corresponding feature should not be
                scaled. If None appears in either the lower or upper quantiles,
                both will be treated as None for that feature.
                Default is (0.25, 0.75).
            with_centering
                bool or PyTree of bools specifying whether to center the data
                before scaling. If True, the median of each feature will be
                subtracted from the data. If False, no centering will be
                applied. If a PyTree is provided, it must match the structure
                of the input data, and each entry specifies whether to center
                the corresponding feature.
            with_scaling
                bool or PyTree of bools specifying whether to scale the data
                to the specified quantile range. If True, the data will be
                scaled according to the quantile range. If False, no scaling
                will be applied. If a PyTree is provided, it must match the
                structure of the input data, and each entry specifies whether
                to scale the corresponding feature.
            eps
                Minimum scale value to avoid division by zero when scaling.
                Default is 1e-12.
        """
        self.q_low, self.q_high = quantile_range
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.eps = eps
        self.scale_: BatchlessRealDataFixed | None = None
        self.median_: BatchlessRealDataFixed | None = None

    def fit(
        self, X: RealDataFixed, y: RealDataFixed | None = None
    ) -> RobustScaler:
        """
        Computes the minimum and maximum values for each feature in the data.

        Parameters
        ----------
            X
                Input data to compute the min and max values from.
            y
                Ignored. Present for API consistency by convention.

        Returns
        -------
            self
                The fitted scaler instance.
        """
        data_qlow_ = jax.tree.map(
            lambda x, q_low, scale: (
                np.quantile(x, q_low, axis=0)
                if q_low is not None and scale
                else None
            ),
            X,
            self.q_low,
            self.with_scaling,
            is_leaf=lambda x: x is None,
        )
        data_qhigh_ = jax.tree.map(
            lambda x, q_high, scale: (
                np.quantile(x, q_high, axis=0)
                if q_high is not None and scale
                else None
            ),
            X,
            self.q_high,
            self.with_scaling,
            is_leaf=lambda x: x is None,
        )
        self.scale_ = jax.tree.map(
            lambda q_high, q_low, scale: (
                q_high - q_low
                if (q_high is not None and q_low is not None and scale)
                else 1.0
            ),
            data_qhigh_,
            data_qlow_,
            self.with_scaling,
            is_leaf=lambda x: x is None,
        )

        self.median_ = jax.tree.map(
            lambda x, center: np.median(x, axis=0) if center else 0.0,
            X,
            self.with_centering,
        )

        return self

    def transform(
        self, X: RealDataFixed, y: RealDataFixed | None = None
    ) -> RealDataFixed:
        """
        Scales the input data to the specified feature range using the
        previously computed min and max values.
        Parameters
        ----------
            X
                Input data to be scaled.
            y
                Ignored. Present for API consistency by convention.
        Returns
        -------
            X_scaled
                The scaled input data.
        """
        if self.scale_ is None or self.median_ is None:
            raise RuntimeError(
                "RobustScaler instance is not fitted yet. Call 'fit' with "
                "appropriate data before using 'transform'."
            )
        return jax.tree.map(
            lambda x, median, scale: (
                (x - median) / scale
                if np.abs(scale) > self.eps
                else x - median
            ),
            X,
            self.median_,
            self.scale_,
        )

    def inverse_transform(
        self, X_scaled: RealDataFixed, y_scaled: RealDataFixed | None = None
    ) -> RealDataFixed:
        """
        Reverts the scaling of the input data back to the original range.

        Parameters
        ----------
            X_scaled
                Scaled input data to be reverted.
            y_scaled
                Ignored. Present for API consistency by convention.

        Returns
        -------
            X_original
                The input data reverted back to its original range.
        """
        if self.scale_ is None or self.median_ is None:
            raise RuntimeError(
                "RobustScaler instance is not fitted yet. Call 'fit' with "
                "appropriate data before using 'inverse_transform'."
            )
        return jax.tree.map(
            lambda x_scaled, median, scale: (
                x_scaled * scale + median
                if np.abs(scale) > self.eps
                else x_scaled + median
            ),
            X_scaled,
            self.median_,
            self.scale_,
        )

    def serialize(
        self,
    ) -> Dict[str, Any]:
        """
        Serializes the scaler's state to a dictionary.

        Returns
        -------
            state
                A dictionary containing the scaler's parameters and state.
        """
        return {
            "quantile_range": (self.q_low, self.q_high),
            "with_centering": self.with_centering,
            "with_scaling": self.with_scaling,
            "eps": self.eps,
            "scale_": self.scale_,
            "median_": self.median_,
        }

    @classmethod
    def deserialize(
        cls,
        state: Dict[str, Any],
    ) -> RobustScaler:
        """
        Deserializes the scaler's state from a dictionary.

        Parameters
        ----------
            state
                A dictionary containing the scaler's parameters and state.

        Returns
        -------
            scaler
                An instance of RobustScaler with the restored state.
        """
        scaler = cls(
            quantile_range=state["quantile_range"],
            with_centering=state["with_centering"],
            with_scaling=state["with_scaling"],
            eps=state["eps"],
        )
        scaler.scale_ = state["scale_"]
        scaler.median_ = state["median_"]
        return scaler
