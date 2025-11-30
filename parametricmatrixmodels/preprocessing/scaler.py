from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import IO, final

import jax
import jax.numpy as np
import numpy as onp
from beartype import beartype
from jaxtyping import jaxtyped

from ..typing import Any, DataFixed, Dict, Tuple


@jaxtyped(typechecker=beartype)
class Scaler(ABC):
    """Abstract base class for data scalers."""

    def __init_subclass__(cls, **kwargs):
        r"""
        Ensures that all methods of all subclasses are also
        decorated with ``@jaxtyped(typechecker=beartype)``. This includes
        "private" methods (those starting with an underscore).
        """
        super().__init_subclass__(**kwargs)
        for name, method in cls.__dict__.items():
            if callable(method) and not hasattr(method, "__jaxtyped__"):
                setattr(cls, name, jaxtyped(typechecker=beartype)(method))
                # set the __jaxtyped__ attribute to avoid re-wrapping
                getattr(cls, name).__jaxtyped__ = True

    @abstractmethod
    def fit(self, X: DataFixed, y: DataFixed | None = None) -> Scaler:
        r"""
        Fit the scaler to the data.

        Parameters
        ----------
        X
            Input features.
        y
            Target values, by default None.
        Returns
        -------
        Scaler
            The fitted scaler.
        """
        pass

    @abstractmethod
    def transform(
        self, X: DataFixed, y: DataFixed | None = None
    ) -> DataFixed | Tuple[DataFixed, DataFixed]:
        r"""
        Transform the data using the fitted scaler.

        Parameters
        ----------
        X
            Input features to be transformed.
        y
            Target values to be transformed, by default None.
        Returns
        -------
        DataFixed | Tuple[DataFixed, DataFixed]
            Transformed input features and optionally transformed target values

        """
        pass

    @abstractmethod
    def inverse_transform(
        self, X: DataFixed, y: DataFixed | None = None
    ) -> DataFixed | Tuple[DataFixed, DataFixed]:
        r"""
        Inverse transform the data using the fitted scaler.

        Parameters
        ----------
        X
            Input features to be inverse transformed.
        y
            Target values to be inverse transformed, by default None.
        Returns
        -------
        DataFixed | Tuple[DataFixed, DataFixed]
            Inverse transformed input features and optionally inverse
            transformed target values

        """
        pass

    @final
    def fit_transform(
        self, X: DataFixed, y: DataFixed | None = None
    ) -> DataFixed | Tuple[DataFixed, DataFixed]:
        r"""
        Fit the scaler to the data and transform it.

        Parameters
        ----------
        X
            Input features.
        y
            Target values, by default None.
        Returns
        -------
        DataFixed | Tuple[DataFixed, DataFixed]
            Transformed input features and optionally transformed target values

        """
        self.fit(X, y)
        return self.transform(X, y)

    @abstractmethod
    def serialize(self) -> Dict[str, Any]:
        r"""
        Serialize the scaler to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Serialized scaler.
        """
        pass

    @classmethod
    @abstractmethod
    def deserialize(cls, data: Dict[str, Any]) -> Scaler:
        r"""
        Deserialize the scaler from a dictionary.

        Parameters
        ----------
        data
            Serialized scaler.
        Returns
        -------
        Scaler
            Deserialized scaler.
        """
        pass

    @final
    def save(self, filepath: IO | Path | str) -> None:
        r"""
        Save the scaler to a file.

        Parameters
        ----------
        filepath
            Path to the file where the scaler will be saved.
        """
        data = self.serialize()
        if isinstance(filepath, str) and not filepath.endswith(".npz"):
            filepath += ".npz"
        onp.savez(filepath, **data)

    @final
    def save_compressed(self, filepath: IO | Path | str) -> None:
        r"""
        Save the scaler to a compressed file.

        Parameters
        ----------
        filepath
            Path to the file where the scaler will be saved.
        """
        data = self.serialize()
        if isinstance(filepath, str) and not filepath.endswith(".npz"):
            filepath += ".npz"
        onp.savez_compressed(filepath, **data)

    @classmethod
    @final
    def load(cls, filepath: IO | Path | str) -> Scaler:
        r"""
        Load the scaler from a file.

        Parameters
        ----------
        filepath
            Path to the file where the scaler is saved.
        Returns
        -------
        Scaler
            Loaded scaler.
        """
        if isinstance(filepath, str) and not filepath.endswith(".npz"):
            filepath += ".npz"
        data = onp.load(filepath, allow_pickle=True)

        # convert all arrays to jax.numpy arrays, including tree leaves
        data_dict = {
            key: jax.tree.map(
                lambda x: np.array(x) if isinstance(x, onp.ndarray) else x,
                (
                    data[key].item()
                    if data[key].shape == ()
                    else (
                        data[key].tolist()
                        if data[key].dtype == object
                        else data[key]
                    )
                ),
            )
            for key in data.files
        }

        return cls.deserialize(data_dict)

    from_file = load
