from __future__ import annotations

import jax
import jax.numpy as np
from jaxtyping import Array, Inexact, Integer, PyTree

from ..typing import Any, Dict, Tuple
from .scaler import Scaler


class OneHot(Scaler):
    r"""
    Encodes arbitrary features as one-hot based on unique values found in
    fit(). Values not found are encoded as all zeros.
    """

    def __init__(
        self,
    ):
        """
        Initializes OneHot encoder.
        """
        self.unique_vals_: (
            PyTree[Inexact[Array, "n_unique ?*f"], " Data"] | None
        ) = None
        self.original_shapes_: PyTree[Tuple[int, ...], " Data"] | None = None

    def fit(
        self,
        X: PyTree[Inexact[Array, "n ?f0 ?*f"], " Data"],
        y: Any | None = None,
    ) -> OneHot:
        """
        Computes the onehot encoding based on the unique values found in the
        input data.

        Parameters
        ----------
            X
                Input data to compute the encoding from.
            y
                Ignored. Present for API consistency by convention.

        Returns
        -------
            self
                The fitted instance.
        """

        # flatten all feature axes and compute unique values for each leaf
        # sort the unique values to ensure consistent ordering across runs and
        # platforms
        self.unique_vals_ = jax.tree.map(
            lambda x: np.sort(
                np.unique(x.reshape(x.shape[0], -1), axis=0), axis=0
            ),
            X,
        )
        self.original_shapes_ = jax.tree.map(lambda x: x.shape[1:], X)

    def transform(
        self,
        X: PyTree[Inexact[Array, "n ?f0 ?*f"], " Data"],
        y: Any | None = None,
    ) -> PyTree[Integer[Array, "n num_unique"], " Data"]:
        """
        Encodes the input data as one-hot based on the unique values found
        during fitting. Values not found during fitting are encoded as all
        zeros.

        Parameters
        ----------
            X
                Input data to be encoded.
            y
                Ignored. Present for API consistency by convention.
        Returns
        -------
            X_encoded
                The encoded input data.
        """
        if self.unique_vals_ is None:
            raise RuntimeError(
                "OneHot instance is not fitted yet. Call 'fit' with "
                "appropriate data before using 'transform'."
            )

        return jax.tree.map(
            lambda x, unique_vals: np.all(
                x.reshape(x.shape[0], -1)[:, None, :]
                == unique_vals[None, :, :],
                axis=-1,
            ).astype(int),
            X,
            self.unique_vals_,
        )

    def inverse_transform(
        self,
        X_encoded: PyTree[Integer[Array, "n num_unique"], " Data"],
        y_encoded: Any | None = None,
    ) -> PyTree[Inexact[Array, "n ?f0 ?*f"], " Data"]:
        """
        Reverts the one-hot encoding back to the original values based on the
        unique values found during fitting. Encoded values that do not match
        any unique value found during fitting are reverted to zeros.

        Parameters
        ----------
            X_encoded
                The one-hot encoded data to be reverted back to original form.
            y_encoded
                Ignored. Present for API consistency by convention.

        Returns
        -------
            X_original
                The input data reverted back to its original form based on the
                unique values found during fitting. Encoded values that do not
                match any unique value found during fitting are reverted to
                zeros.
        """
        if self.unique_vals_ is None or self.original_shapes_ is None:
            raise RuntimeError(
                "OneHot instance is not fitted yet. Call 'fit' with "
                "appropriate data before using 'inverse_transform'."
            )

        return jax.tree.map(
            lambda x_encoded, unique_vals, original_shape: (
                x_encoded @ unique_vals
            ).reshape(x_encoded.shape[0], *original_shape),
            X_encoded,
            self.unique_vals_,
            self.original_shapes_,
        )

    def serialize(
        self,
    ) -> Dict[str, Any]:
        """
        Serializes the encoder's state to a dictionary.

        Returns
        -------
            state
                A dictionary containing the encoder's parameters and state.
        """
        return {
            "unique_vals_": self.unique_vals_,
            "original_shapes_": self.original_shapes_,
        }

    @classmethod
    def deserialize(
        cls,
        state: Dict[str, Any],
    ) -> OneHot:
        """
        Deserializes the encoder's state from a dictionary.

        Parameters
        ----------
            state
                A dictionary containing the encoder's parameters and state.

        Returns
        -------
            encoder
                An instance of OneHot with the restored state.
        """
        encoder = cls()
        encoder.unique_vals_ = state["unique_vals_"]
        encoder.original_shapes_ = state["original_shapes_"]
        return encoder
