from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as np

from ._eigensystems import select_eigenpairs_by_eigenvalue
from .basemodule import BaseModule


class Eigensystem(BaseModule):
    r"""
    Module to compute selected eigenpairs of a symmetric (Hermitian) matrix.

    The output of this module for a single input sample is an array where each
    column is an eigenvalue followed by its corresponding eigenvector, e.g.

    .. math::

        \begin{bmatrix}
            \lambda_1 & \lambda_2 & \ldots \\
            v_{1,1} & v_{2,1} & \ldots \\
            v_{1,2} & v_{2,2} & \ldots \\
            \vdots & \vdots & \ddots
        \end{bmatrix}

    See Also
    --------
    Eigenvalues
        Module to compute only eigenvalues.
    Eigenvectors
        Module to compute only eigenvectors.
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
            Number of eigenpairs to compute. Must be a positive integer.
            Default is 1.
        which
            Which eigenpairs to return, by default "SA".
            Options are:
            - 'SA' for smallest algebraic (default)
            - 'LA' for largest algebraic
            - 'SM' for smallest magnitude
            - 'LM' for largest magnitude
            - 'EA' for exterior algebraically
            - 'EM' for exterior by magnitude
            - 'IA' for interior algebraically
            - 'IM' for interior by magnitude

            For algebraic 'which' options, the eigenpairs are returned in
            ascending eigenvalue algebraic order.

            For magnitude 'which' options, the eigenpairs are returned in
            ascending eigenvalue magnitude order.
        """
        if num_eig <= 0 or not isinstance(num_eig, int):
            raise ValueError("num_eig must be a positive integer")
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
            return "Eigensystem(ground state)"
        else:
            return (
                f"Eigensystem(num_eig={self.num_eig},"
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
            E, V = (
                jax.vmap(
                    select_eigenpairs_by_eigenvalue, in_axes=(0, 0, None, None)
                )(np.linalg.eigh(input_NF), self.num_eig, self.which),
            )

            # E is (N_batch, num_eig)
            # V is (N_batch, N_dim, num_eig)
            # we want to stack along axis 1 so that the output is
            # (N_batch, N_dim + 1, num_eig)

            out = np.concatenate([E[:, None, :], V], axis=1)

            return out, state

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

        # returns an array where each column is an eigenvalue followed by its
        # eigenvector, e.g.
        # [ [eigval1, eigval2, ...],
        # [ [eigvec1_comp1, eigvec2_comp1, ...],
        #   [eigvec1_comp2, eigvec2_comp2, ...],
        #   ...
        # ]
        # so out[0, :] are the eigenvalues and out[1:, :] are the eigenvectors

        return (input_shape[0] + 1, self.num_eig)

    def get_hyperparameters(self) -> dict[str, Any]:
        return {
            "num_eig": self.num_eig,
            "which": self.which,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
        }

    def set_hyperparameters(self, hyperparams: dict[str, Any]) -> None:
        super(Eigensystem, self).set_hyperparameters(hyperparams)

    def get_params(self) -> tuple[np.ndarray, ...]:
        return ()

    def set_params(self, params: tuple[np.ndarray, ...]) -> None:
        return
