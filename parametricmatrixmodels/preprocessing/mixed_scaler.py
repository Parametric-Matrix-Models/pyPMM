from __future__ import annotations

import sys

import jax
from jaxtyping import PyTree

from ..typing import (
    Any,
    Dict,
    RealDataFixed,
)
from .scaler import Scaler


class MixedScaler(Scaler):
    r"""
    Scaler made of other scalers: each feature is scaled with a different
    scaler.
    """

    def __init__(
        self,
        feature_scalers: PyTree[Scaler | None, "RealDataFixed"] | None = None,
    ):
        """
        Initializes the MixedScaler with the specified feature scalers.

        Parameters
        ----------
            feature_scalers
                A PyTree of scalers, where each scaler corresponds to a
                specific feature in the input data. If None, no scaling will
                be applied to any feature. The structure of the PyTree should
                match the structure of the input data. Features with a scaler
                set to None will not be scaled.
        """
        self.feature_scalers = feature_scalers

    def fit(
        self, X: RealDataFixed, y: RealDataFixed | None = None
    ) -> MixedScaler:
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
        if self.feature_scalers is None:
            return self

        def fit_individual(
            scaler: Scaler | None, feature_data: RealDataFixed
        ) -> Scaler | None:
            if scaler is not None:
                return scaler.fit(feature_data)
            else:
                return None

        self.feature_scalers = jax.tree.map(
            fit_individual,
            self.feature_scalers,
            X,
            is_leaf=lambda x: x is None,
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
        if self.feature_scalers is None:
            return X

        def transform_individual(
            scaler: Scaler | None, feature_data: RealDataFixed
        ) -> RealDataFixed:
            if scaler is not None:
                return scaler.transform(feature_data)
            else:
                return feature_data

        X_scaled = jax.tree.map(
            transform_individual,
            self.feature_scalers,
            X,
            is_leaf=lambda x: x is None,
        )

        return X_scaled

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
        if self.feature_scalers is None:
            return X_scaled

        def inverse_transform_individual(
            scaler: Scaler | None, feature_data: RealDataFixed
        ) -> RealDataFixed:
            if scaler is not None:
                return scaler.inverse_transform(feature_data)
            else:
                return feature_data

        X_original = jax.tree.map(
            inverse_transform_individual,
            self.feature_scalers,
            X_scaled,
            is_leaf=lambda x: x is None,
        )

        return X_original

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
        if self.feature_scalers is None:
            return {
                "feature_scalers": None,
                "scaler_fulltypenames": None,
                "scaler_typenames": None,
                "scaler_modules": None,
            }
        else:
            serialized_scalers = jax.tree.map(
                lambda scaler: (
                    scaler.serialize() if scaler is not None else None
                ),
                self.feature_scalers,
                is_leaf=lambda x: x is None,
            )
            scaler_fulltypenames = jax.tree.map(
                lambda s: str(type(s)) if s is not None else None,
                self.feature_scalers,
                is_leaf=lambda x: x is None,
            )
            scaler_typenames = jax.tree.map(
                lambda s: s.__class__.__name__ if s is not None else None,
                self.feature_scalers,
                is_leaf=lambda x: x is None,
            )
            scaler_modules = jax.tree.map(
                lambda s: s.__module__ if s is not None else None,
                self.feature_scalers,
                is_leaf=lambda x: x is None,
            )

            return {
                "feature_scalers": serialized_scalers,
                "scaler_fulltypenames": scaler_fulltypenames,
                "scaler_typenames": scaler_typenames,
                "scaler_modules": scaler_modules,
            }

    @classmethod
    def deserialize(
        cls,
        state: Dict[str, Any],
    ) -> MixedScaler:
        """
        Deserializes the scaler's state from a dictionary.

        Parameters
        ----------
            state
                A dictionary containing the scaler's parameters and state.

        Returns
        -------
            scaler
                An instance of MixedScaler with the restored state.
        """
        feature_scalers_serial = state.get("feature_scalers", None)
        scaler_typenames = state.get("scaler_typenames", None)
        scaler_modules = state.get("scaler_modules", None)

        if feature_scalers_serial is None:
            return cls(feature_scalers=None)
        else:
            feature_scalers = jax.tree.map(
                lambda scaler_name, scaler_module, scaler_serial: (
                    getattr(
                        sys.modules[scaler_module], scaler_name
                    ).deserialize(scaler_serial)
                    if scaler_serial is not None
                    else None
                ),
                scaler_typenames,
                scaler_modules,
                feature_scalers_serial,
                is_leaf=lambda x: x is None,
            )
            return cls(feature_scalers=feature_scalers)
