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


class MinMaxScaler(Scaler):
    r"""
    Scales data to a specified range [min_val, max_val] using min-max
    normalization. Valid for real-valued data.
    """

    def __init__(
        self,
        feature_range: (
            Real[Array, "2"]
            | Tuple[
                float
                | int
                | BatchlessRealDataFixed
                | PyTree[float | int, "RealDataFixed"],
                float
                | int
                | BatchlessRealDataFixed
                | PyTree[float | int, "RealDataFixed"],
            ]
            | List[float | int | BatchlessRealDataFixed]
        ) = (0.0, 1.0),
    ):
        """
        Initializes the MinMaxScaler with the desired feature range.

        Parameters
        ----------
            feature_range
                A tuple specifying the desired range of transformed data. Can
                be a tuple of floats or a tuple of arrays. If arrays are
                provided, they must match the PyTree structure and shapes of
                the input data, excluding the leading batch dimension.
                Default is (0.0, 1.0).
        """
        self.min_val, self.max_val = feature_range
        self.data_min_: BatchlessRealDataFixed | None = None
        self.data_max_: BatchlessRealDataFixed | None = None

    def fit(
        self, X: RealDataFixed, y: RealDataFixed | None = None
    ) -> MinMaxScaler:
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
        self.data_min_ = jax.tree.map(lambda x: np.min(x, axis=0), X)
        self.data_max_ = jax.tree.map(lambda x: np.max(x, axis=0), X)
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
        if self.data_min_ is None or self.data_max_ is None:
            raise RuntimeError(
                "MinMaxScaler instance is not fitted yet. Call 'fit' with "
                "appropriate data before using 'transform'."
            )
        if np.isscalar(self.min_val):
            min_val = jax.tree.map(lambda x: self.min_val, self.data_min_)
        else:
            min_val = self.min_val
        if np.isscalar(self.max_val):
            max_val = jax.tree.map(lambda x: self.max_val, self.data_max_)
        else:
            max_val = self.max_val

        return jax.tree.map(
            lambda x, data_min, data_max, min_v, max_v: (
                (x - data_min) / (data_max - data_min) * (max_v - min_v)
                + min_v
            ),
            X,
            self.data_min_,
            self.data_max_,
            min_val,
            max_val,
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
        if self.data_min_ is None or self.data_max_ is None:
            raise RuntimeError(
                "MinMaxScaler instance is not fitted yet. Call 'fit' with "
                "appropriate data before using 'inverse_transform'."
            )
        if np.isscalar(self.min_val):
            min_val = jax.tree.map(lambda x: self.min_val, self.data_min_)
        else:
            min_val = self.min_val
        if np.isscalar(self.max_val):
            max_val = jax.tree.map(lambda x: self.max_val, self.data_max_)
        else:
            max_val = self.max_val

        return jax.tree.map(
            lambda x_scaled, data_min, data_max, min_v, max_v: (
                (x_scaled - min_v) / (max_v - min_v) * (data_max - data_min)
                + data_min
            ),
            X_scaled,
            self.data_min_,
            self.data_max_,
            min_val,
            max_val,
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
            "feature_range": (self.min_val, self.max_val),
            "data_min_": self.data_min_,
            "data_max_": self.data_max_,
        }

    @classmethod
    def deserialize(
        cls,
        state: Dict[str, Any],
    ) -> MinMaxScaler:
        """
        Deserializes the scaler's state from a dictionary.

        Parameters
        ----------
            state
                A dictionary containing the scaler's parameters and state.

        Returns
        -------
            scaler
                An instance of MinMaxScaler with the restored state.
        """
        scaler = cls(feature_range=state["feature_range"])
        scaler.data_min_ = state["data_min_"]
        scaler.data_max_ = state["data_max_"]
        return scaler
