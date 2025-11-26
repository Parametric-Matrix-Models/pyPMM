from __future__ import annotations

import jax
import jax.numpy as np
from beartype import beartype
from jaxtyping import jaxtyped

from ..eigen_util import (
    select_eigenpairs_by_eigenvalue,
    validate_eigensystem_input_shape,
)
from ..tree_util import is_shape_leaf
from ..typing import (
    Any,
    ArrayData,
    Data,
    DataShape,
    Dict,
    ModuleCallable,
    Params,
    State,
    Tuple,
)
from .basemodule import BaseModule


class Eigensystem(BaseModule):
    r"""
    Module to compute selected eigenpairs of a symmetric (Hermitian) matrix.
    Can be applied over PyTrees of matrices.

    The output of this module for a single input matrix is a Dictionary
    (PyTree) keyed by 'eigenvalues' and 'eigenvectors'.

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
        num_eig: int | None = 1,
        which: str = "SA",
    ) -> None:
        r"""
        Parameters
        ----------
        num_eig
            Number of eigenpairs to compute. Must be a positive integer or
            None. If None, all pairs are returned. Default is 1.
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
        if num_eig is not None and (
            num_eig <= 0 or not isinstance(num_eig, int)
        ):
            raise ValueError("num_eig must be a positive integer or None.")
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

    def name(self) -> str:
        if self.num_eig == 1 and self.which == "sa":
            return "Eigensystem(ground state)"
        elif self.num_eig is None:
            return "Eigensystem(ALL, which={self.which.upper()})"
        else:
            return (
                f"Eigensystem(num_eig={self.num_eig},"
                f" which={self.which.upper()})"
            )

    def is_ready(self) -> bool:
        return True

    def get_num_trainable_floats(self) -> int | None:
        return 0

    def _get_callable(self) -> ModuleCallable:

        @jaxtyped(typechecker=beartype)
        def get_eigensystem(data: ArrayData) -> Dict[str, ArrayData]:
            # compute all eigensystems over the batch dimension, then vmap to
            # select the desired eigenpairs
            E, V = jax.vmap(
                select_eigenpairs_by_eigenvalue, in_axes=(0, 0, None, None)
            )(*np.linalg.eigh(data), self.num_eig, self.which)
            return {"eigenvalues": E, "eigenvectors": V}

        @jaxtyped(typechecker=beartype)
        def tree_map_get_eigensystem(
            params: Params,
            data: Data,
            training: bool,
            state: State,
            rng: Any,
        ) -> Tuple[Data, State]:
            # tree map over the data PyTree
            return (
                jax.tree.map(
                    get_eigensystem,
                    data,
                ),
                state,
            )

        return tree_map_get_eigensystem

    def compile(self, rng: Any, input_shape: DataShape) -> None:
        # ensure input shape is valid
        validate_eigensystem_input_shape(input_shape, self.num_eig)

    def get_output_shape(self, input_shape: DataShape) -> DataShape:
        validate_eigensystem_input_shape(input_shape, self.num_eig)

        # the output is a PyTree with the original structure as a prefix to a
        # two-keyed dictionary with 'eigenvalues' and 'eigenvectors'
        def get_eigensystem_shape(
            matrix_shape: Tuple[int, int],
        ) -> Dict[str, Tuple[int, ...]]:
            n = matrix_shape[-1]
            k = self.num_eig if self.num_eig is not None else n
            return {
                "eigenvalues": (k,),
                "eigenvectors": (n, k),
            }

        return jax.tree.map(
            get_eigensystem_shape,
            input_shape,
            is_leaf=is_shape_leaf,
        )

    def get_hyperparameters(self) -> dict[str, Any]:
        return {
            "num_eig": self.num_eig,
            "which": self.which,
        }

    def set_hyperparameters(self, hyperparams: dict[str, Any]) -> None:
        super(Eigensystem, self).set_hyperparameters(hyperparams)

    def get_params(self) -> tuple[np.ndarray, ...]:
        return ()

    def set_params(self, params: tuple[np.ndarray, ...]) -> None:
        return
