from __future__ import annotations

import jax
import jax.numpy as np

from ..typing import Any, BatchlessRealDataFixed, Dict, RealDataFixed
from .scaler import Scaler


class StandardScaler(Scaler):
    r"""
    Scales input data to have zero mean and unit variance for each feature.
    Valid for real-valued data.
    """

    def __init__(
        self,
    ):
        """
        Initializes the StandardScaler instance.
        """
        self.mean_: BatchlessRealDataFixed | None = None
        self.scale_: BatchlessRealDataFixed | None = None

    def fit(
        self, X: RealDataFixed, y: RealDataFixed | None = None
    ) -> StandardScaler:
        """
        Computes the mean and standard deviation for each feature in the input
        data.

        Parameters
        ----------
            X
                Input data to be scaled.
            y
                Ignored. Present for API consistency by convention.

        Returns
        -------
            self
                The fitted scaler instance.
        """
        self.mean_ = jax.tree.map(lambda x: np.mean(x, axis=0), X)
        self.scale_ = jax.tree.map(lambda x: np.std(x, axis=0), X)

        return self

    def transform(
        self, X: RealDataFixed, y: RealDataFixed | None = None
    ) -> RealDataFixed:
        """
        Scales the input data to have zero mean and unit variance based on the
        computed mean and standard deviation from the `fit` method.
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
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError(
                "StandardScaler instance is not fitted yet. Call 'fit' with "
                "appropriate data before using 'transform'."
            )
        return jax.tree.map(
            lambda x, mean, scale: (x - mean) / scale,
            X,
            self.mean_,
            self.scale_,
        )

    def inverse_transform(
        self, X_scaled: RealDataFixed, y_scaled: RealDataFixed | None = None
    ) -> RealDataFixed:
        """
        Reverts the scaling of the input data

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
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError(
                "StandardScaler instance is not fitted yet. Call 'fit' with "
                "appropriate data before using 'inverse_transform'."
            )
        return jax.tree.map(
            lambda x_scaled, mean, scale: x_scaled * scale + mean,
            X_scaled,
            self.mean_,
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
            "mean_": self.mean_,
            "scale_": self.scale_,
        }

    @classmethod
    def deserialize(
        cls,
        state: Dict[str, Any],
    ) -> StandardScaler:
        """
        Deserializes the scaler's state from a dictionary.

        Parameters
        ----------
            state
                A dictionary containing the scaler's parameters and state.

        Returns
        -------
            scaler
                An instance of StandardScaler with the restored state.
        """
        scaler = cls()
        scaler.mean_ = state["mean_"]
        scaler.scale_ = state["scale_"]
        return scaler
