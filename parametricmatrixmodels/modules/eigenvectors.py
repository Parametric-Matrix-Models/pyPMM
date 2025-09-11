from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as np

from ._eigensystems import select_eigenpairs_by_eigenvalue
from .basemodule import BaseModule


class Eigenvectors(BaseModule):
    r"""
    Module to compute selected eigenvectors of a symmetric (Hermitian) matrix.

    The output of this module for a single input sample is an array where each
    column is an eigenvector, with the columns ordered according to the
    specified `which` parameter.

    See Also
    --------
    Eigenvalues
        Module to compute only eigenvalues.
    Eigensystem
        Module to compute both eigenvalues and eigenvectors.
    jax.numpy.linalg.eigh
        JAX function to compute the eigensystem of a symmetric (Hermitian)
        matrix, which is used internally by this module.
    """

    def __init__(
        self,
        num_eig: int = 1,
        which: str = "SA",
    ) -> None:
        r"""
        Parameters
        ----------
        num_eig
            Number of eigenvectors to compute. Must be a positive integer.
            Default is 1.
        which
            Which eigenvectors to return based on associated eigenvalues,
            by default "SA".
            Options are:
            - 'SA' for smallest algebraic (default)
            - 'LA' for largest algebraic
            - 'SM' for smallest magnitude
            - 'LM' for largest magnitude
            - 'EA' for exterior algebraically
            - 'EM' for exterior by magnitude
            - 'IA' for interior algebraically
            - 'IM' for interior by magnitude

            For algebraic 'which' options, the eigenvectors are returned in
            ascending eigenvalue algebraic order.

            For magnitude 'which' options, the eigenvectors are returned in
            ascending eigenvalue magnitude order.
        """
        if num_eig is not None and (
            num_eig <= 0 or not isinstance(num_eig, int)
        ):
            raise ValueError(
                f"num_eig must be a positive integer, got {num_eig}"
            )
        if which.lower() not in [
            "sa",
            "la",
            "sm",
            "lm",
            "ea",
            "em",
            "ia",
            "im",
        ]:
            raise ValueError(
                "which must be one of: 'SA', 'LA', 'SM', 'LM', 'EA', 'EM', "
                f"'IA', 'IM'. Got: {which}"
            )

        self.num_eig = num_eig
        self.which = which.lower()
        self.input_shape = None
        self.output_shape = None

    def name(self) -> str:
        if self.num_eig == 1 and self.which == "sa":
            return "Eigenvectors(ground state)"
        else:
            return (
                f"Eigenvectors(num_eig={self.num_eig},"
                f" which={self.which.upper()})"
            )

    def is_ready(self) -> bool:
        return self.input_shape is not None and self.output_shape is not None

    def get_num_trainable_floats(self) -> int | None:
        return 0

    def _get_callable(self) -> Callable[
        [
            tuple[np.ndarray, ...],
            np.ndarray,
            bool,
            tuple[np.ndarray, ...],
            Any,
        ],
        tuple[np.ndarray, tuple[np.ndarray, ...]],
    ]:
        def _callable(
            params: tuple[np.ndarray, ...],
            input_NF: np.ndarray,
            training: bool,
            state: tuple[np.ndarray, ...],
            rng: Any,
        ) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
            E, V = jax.vmap(
                select_eigenpairs_by_eigenvalue,
                in_axes=(0, 0, None, None),
            )(*np.linalg.eigh(input_NF), self.num_eig, self.which)
            return V, state

        return _callable

    def compile(self, rng: Any, input_shape: tuple[int, ...]) -> None:
        # ensure input shape is valid
        if len(input_shape) != 2 or input_shape[0] != input_shape[1]:
            raise ValueError(
                f"Input shape must be a square matrix, got {input_shape}"
            )

        self.input_shape = input_shape
        self.output_shape = self.get_output_shape(input_shape)

    def get_output_shape(
        self, input_shape: tuple[int, ...]
    ) -> tuple[int, ...]:
        if len(input_shape) != 2 or input_shape[0] != input_shape[1]:
            raise ValueError(
                f"Input shape must be a square matrix, got {input_shape}"
            )

        return (input_shape[0], self.num_eig)

    def get_hyperparameters(self) -> dict[str, Any]:
        return {
            "num_eig": self.num_eig,
            "which": self.which,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
        }

    def set_hyperparameters(self, hyperparams: dict[str, Any]) -> None:
        super(Eigenvectors, self).set_hyperparameters(hyperparams)

    def get_params(self) -> tuple[np.ndarray, ...]:
        return ()

    def set_params(self, params: tuple[np.ndarray, ...]) -> None:
        return
