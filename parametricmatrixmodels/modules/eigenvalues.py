from __future__ import annotations

import jax
import jax.numpy as np
from beartype import beartype
from jaxtyping import jaxtyped

from ..eigen_util import select_eigenvalues, validate_eigensystem_input_shape
from ..tree_util import is_shape_leaf
from ..typing import (
    Any,
    ArrayData,
    Data,
    DataShape,
    HyperParams,
    ModuleCallable,
    Params,
    State,
    Tuple,
)
from .basemodule import BaseModule


class Eigenvalues(BaseModule):
    r"""
    Module to compute selected eigenvalues of a symmetric (Hermitian) matrix.
    Can be applied over PyTrees of matrices.

    See Also
    --------
    Eigenvectors
        Module to compute only eigenvectors.
    Eigensystem
        Module to compute both eigenvalues and eigenvectors. Although
        JAX/NumPy's `np.linalg.eigvalsh` just calls `np.linalg.eigh` and
        discards the eigenvectors, it is still more efficient here so that an
        entire batch of eigenvectors aren't passed around needlessly.
    jax.numpy.linalg.eigvalsh
        JAX function to compute the eigenvalues of a symmetric (Hermitian)
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
            Number of eigenvalues to compute. Must be a positive integer or
            None. If None, all eigenvalues are returned. Default is 1.
        which
            Which eigenvalues to return, by default "SA".
            Options are:
            - 'SA' for smallest algebraic (default)
            - 'LA' for largest algebraic
            - 'SM' for smallest magnitude
            - 'LM' for largest magnitude
            - 'EA' for exterior algebraically
            - 'EM' for exterior by magnitude
            - 'IA' for interior algebraically
            - 'IM' for interior by magnitude

            For algebraic 'which' options, the eigenvalues are returned in
            ascending algebraic order.

            For magnitude 'which' options, the eigenvalues are returned in
            ascending magnitude order.
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

    @property
    def name(self) -> str:
        if self.num_eig == 1 and self.which == "sa":
            return "Eigenvalues(ground state)"
        elif self.num_eig is None:
            return "Eigenvalues(ALL, which={self.which.upper()})"
        else:
            return (
                f"Eigenvalues(num_eig={self.num_eig},"
                f" which={self.which.upper()})"
            )

    def is_ready(self) -> bool:
        return True

    def get_num_trainable_floats(self) -> int | None:
        return 0

    def _get_callable(self) -> ModuleCallable:

        @jaxtyped(typechecker=beartype)
        def get_eigenvalues(data: ArrayData) -> ArrayData:
            # compute all eigenvalues over the batch dimension, then vmap
            # to select the desired eigenvalues
            return jax.vmap(select_eigenvalues, in_axes=(0, None, None))(
                np.linalg.eigvalsh(data),
                self.num_eig,
                self.which,
            )

        @jaxtyped(typechecker=beartype)
        def tree_map_get_eigenvalues(
            params: Params,
            data: Data,
            training: bool,
            state: State,
            rng: Any,
        ) -> Tuple[Data, State]:
            # tree map over the data PyTree
            return (
                jax.tree.map(
                    get_eigenvalues,
                    data,
                ),
                state,  # state is not used in this module, return it unchanged
            )

        return tree_map_get_eigenvalues

    def compile(self, rng: Any, input_shape: DataShape) -> None:
        # ensure input shape is valid
        validate_eigensystem_input_shape(input_shape, self.num_eig)

    def get_output_shape(self, input_shape: DataShape) -> DataShape:
        validate_eigensystem_input_shape(input_shape, self.num_eig)
        return jax.tree.map(
            lambda s: (
                self.num_eig if self.num_eig is not None else input_shape[0],
            ),
            input_shape,
            is_leaf=is_shape_leaf,
        )

    def get_hyperparameters(self) -> HyperParams:
        return {
            "num_eig": self.num_eig,
            "which": self.which,
        }

    def set_hyperparameters(self, hyperparams: HyperParams) -> None:
        super().set_hyperparameters(hyperparams)

    def get_params(self) -> Params:
        return ()

    def set_params(self, params: Params) -> None:
        return
