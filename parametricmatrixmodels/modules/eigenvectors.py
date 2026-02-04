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
    HyperParams,
    ModuleCallable,
    Params,
    State,
    Tuple,
)
from .basemodule import BaseModule


class Eigenvectors(BaseModule):
    r"""
    Module to compute selected eigenvectors of a symmetric (Hermitian) matrix.
    Can be applied over PyTrees of matrices.

    The output of this module for a single input matrix is an array where each
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

    __version__: str = "0.0.0"

    def __init__(
        self,
        num_eig: int | None = 1,
        which: str = "SA",
    ) -> None:
        r"""
        Parameters
        ----------
        num_eig
            Number of eigenvectors to compute. Must be a positive integer or
            None. If None, all eigenvectors are returned. Default is 1.
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
                f"num_eig must be a positive integer or None, got {num_eig}"
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

    @property
    def name(self) -> str:
        if self.num_eig == 1 and self.which == "sa":
            return "Eigenvectors(ground state)"
        elif self.num_eig is None:
            return f"Eigenvectors(ALL, which={self.which.upper()})"
        else:
            return (
                f"Eigenvectors(num_eig={self.num_eig},"
                f" which={self.which.upper()})"
            )

    def is_ready(self) -> bool:
        return True

    def get_num_trainable_floats(self) -> int | None:
        return 0

    def _get_callable(self) -> ModuleCallable:

        @jaxtyped(typechecker=beartype)
        def get_eigenvectors(data: ArrayData) -> ArrayData:
            # compute all eigenvectors over the batch dimension, then vmap
            # to select the desired eigenvectors
            return jax.vmap(
                select_eigenpairs_by_eigenvalue, in_axes=(0, 0, None, None)
            )(*np.linalg.eigh(data), self.num_eig, self.which,)[1]

        @jaxtyped(typechecker=beartype)
        def tree_map_get_eigenvectors(
            params: Params,
            data: Data,
            training: bool,
            state: State,
            rng: Any,
        ) -> Tuple[Data, State]:
            # tree map over the data PyTree
            return (
                jax.tree.map(
                    get_eigenvectors,
                    data,
                ),
                state,
            )

        return tree_map_get_eigenvectors

    def compile(self, rng: Any, input_shape: DataShape) -> None:
        # ensure input shape is valid
        validate_eigensystem_input_shape(input_shape, self.num_eig)

    def get_output_shape(self, input_shape: DataShape) -> DataShape:
        validate_eigensystem_input_shape(input_shape, self.num_eig)
        return jax.tree.map(
            lambda s: (
                s[0],
                self.num_eig if self.num_eig is not None else s[0],
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
