from __future__ import annotations

import jax
import jax.numpy as np
from beartype import beartype
from jaxtyping import Array, Inexact, jaxtyped

from ..tree_util import is_shape_leaf, is_single_leaf
from ..typing import (
    Any,
    Data,
    DataShape,
    HyperParams,
    ModuleCallable,
    Params,
    State,
    Tuple,
)
from ._smoothing import exact_smoothing_matrix
from .basemodule import BaseModule


class LowRankAffineHermitianMatrix(BaseModule):
    r"""
    Module that builds a parametric hermitian matrix that is affine in the
    input features, composed of low-rank Hermitian matrices, with optional
    smoothing.

    :math:`M(x) = M_0 + x_1 M_1 + ... + x_p M_p + s C`
    where :math:`M_0, M_1, ..., M_p` are (trainable) Hermitian matrices,
    :math:`x_1, ..., x_p` are the input features, :math:`s` is the
    smoothing hyperparameter, and :math:`C` is a matrix that is computed
    as the imaginary unit times the sum of the commutators of all the
    :math:`M_i` matrices, in an efficient way using cumulative sums and the
    linearity of the commutator:

    .. math::

        C &= i\sum_{\substack{i,j\\i\neq j}} \left[M_i, M_j\right] \\
          &= i\sum_{\substack{i\\i\neq j}}
             \left[M_i, \sum_k^j M_k\right]

    Only accepts PyTree data that has a single leaf array that is 1D, excluding
    the batch dimension. The PyTree structure is preserved in the output.

    See Also
    --------
    AffineEigenvaluePMM
        Module that builds a parametric matrix that is affine in the input
        features, the same as this module, but returns the eigenvalues of said
        matrix.
    AffineObservablePMM
        Module that builds a parametric matrix that is affine in the input
        features, the same as this module, but returns the sum of trainable
        observables and transition probabilities of eigenstates of said matrix.
    Eigenvalues
        Module that computes the eigenvalues of a given Hermitian matrix. Can
        be applied after this module to effectively re-create the
        ``AffineEigenvaluePMM`` module.
    """

    def __init__(
        self,
        matrix_size: int | None = None,
        matrix_rank: int | None = None,
        smoothing: float | None = None,
        init_magnitude: float = 1e-2,
        bias_term: bool = True,
    ) -> None:
        """
        Create an ``AffineHermitianMatrix`` module.

        Parameters
        ----------
            matrix_size
                Size of the PMM matrices (square), shorthand :math:`n`.
            matrix_rank
                Rank of the PMM matrices, shorthand :math:`r`.
            smoothing
                Optional smoothing parameter. Set to ``0.0`` to disable
                smoothing. Default is ``None``/``0.0`` (no smoothing).
            init_magnitude
                Optional initial magnitude of the random matrices, used when
                initializing the module. Default is ``1e-2``.
            bias_term
                If ``True``, include the bias term :math:`M_0` in the equation
                for :math:`M(x)`. Default is ``True``.
        """

        # input validation
        if matrix_size is not None and (
            matrix_size <= 0 or not isinstance(matrix_size, int)
        ):
            raise ValueError("matrix_size must be a positive integer")
        if matrix_rank is not None and (
            matrix_rank <= 0 or not isinstance(matrix_rank, int)
        ):
            raise ValueError("matrix_rank must be a positive integer")
        if matrix_size is not None and matrix_rank is not None:
            if matrix_rank > matrix_size:
                raise ValueError(
                    "matrix_rank cannot be larger than matrix_size"
                )

        self.matrix_size = matrix_size
        self.matrix_rank = matrix_rank
        self.smoothing = smoothing if smoothing is not None else 0.0
        self.bias_term = bias_term
        self.us = None  # vectors for low-rank representation
        self.ls = None  # real scalars for low-rank representation
        self.init_magnitude = init_magnitude

    @property
    def name(self) -> str:
        return (
            f"AffineHermitianMatrix({self.matrix_size}x{self.matrix_size}"
            f"{'' if self.smoothing == 0.0 else f', smooth={self.smoothing}'}"
            f"{'' if self.bias_term else ', no bias'})"
        )

    def is_ready(self) -> bool:
        return self.us is not None and self.ls is not None

    def get_num_trainable_floats(self) -> int | None:
        if not self.is_ready():
            return None

        return 2 * self.us.size + self.ls.size

    def _get_callable(self) -> ModuleCallable:

        if self.bias_term:

            @jaxtyped(typechecker=beartype)
            def make_affine_hermitian_matrix(
                Ms: Inexact[Array, "p n n"],
                features: Inexact[Array, "b p-1"],
            ) -> Inexact[Array, "b n n"]:

                # convert to common dtype, this should be traced out
                dtype = np.result_type(Ms.dtype, features.dtype)
                Ms = Ms.astype(dtype)
                features = features.astype(dtype)

                M = Ms[0][None, :, :] + np.einsum(
                    "ni,ijk->njk", features, Ms[1:]
                )

                if self.smoothing != 0.0:
                    M += (
                        self.smoothing * exact_smoothing_matrix(Ms)[None, :, :]
                    )
                return M

        else:

            @jaxtyped(typechecker=beartype)
            def make_affine_hermitian_matrix(
                Ms: Inexact[Array, "p n n"],
                features: Inexact[Array, "b p"],
            ) -> Inexact[Array, "b n n"]:
                # convert to common dtype, this should be traced out
                dtype = np.result_type(Ms.dtype, features.dtype)
                Ms = Ms.astype(dtype)
                features = features.astype(dtype)

                M = np.einsum("ni,ijk->njk", features, Ms)

                if self.smoothing != 0.0:
                    M += (
                        self.smoothing * exact_smoothing_matrix(Ms)[None, :, :]
                    )
                return M

        @jaxtyped(typechecker=beartype)
        def make_low_rank_matrices(
            us: Inexact[Array, "p n r"],
            ls: Inexact[Array, "p r"],
        ) -> Inexact[Array, "p n n"]:
            # build low-rank Hermitian matrices from us and ls
            # Ms[i] = sum_k^r ls[i,k] * us[i,:,k] @ us[i,:,k].conj().T

            Ms = np.einsum("pr,pnr,pmr->pnm", ls, us, us.conj())
            return Ms

        @jaxtyped(typechecker=beartype)
        def low_rank_affine_hermitian_matrix(
            params: Params,
            data: Data,
            training: bool,
            state: State,
            rng: Any,
        ) -> Tuple[Data, State]:
            ls, us = params
            Ms = make_low_rank_matrices(us, ls)

            # enforce Hermitian matrices (this shouldn't be necessary)
            Ms = (Ms + Ms.conj().transpose((0, 2, 1))) / 2.0

            # tree map over the data to preserve the PyTree structure

            # compile will have validated that there is only one leaf that is
            # a 1D array, excluding the batch dimension
            M = jax.tree.map(
                lambda x: make_affine_hermitian_matrix(Ms, x),
                data,
            )

            return (M, state)

        return low_rank_affine_hermitian_matrix

    def compile(self, rng: Any, input_shape: DataShape) -> None:

        valid, leaf = is_single_leaf(
            input_shape, ndim=1, is_leaf=is_shape_leaf
        )

        # input shape must be a PyTree with a single leaf that is 1D
        if not valid:
            raise ValueError(
                "Input shape must be a PyTree with a single leaf consisting of"
                " a 1D array, got: {input_shape}"
            )

        # number of matrices is number of features + 1 (bias) if bias is used
        p = leaf[0] + 1 if self.bias_term else leaf[0]

        # if the module is already ready, just verify the input shape
        if self.is_ready():
            if self.us.shape[0] != p:
                raise ValueError(
                    f"Input shape {leaf} does not match the expected "
                    f"number of features {self.us.shape[0] - 1} "
                )
            return

        rng_ureal, rng_uimag, rng_l = jax.random.split(rng, 3)

        self.us = self.init_magnitude * (
            jax.random.normal(
                rng_ureal,
                (p, self.matrix_size, self.matrix_rank),
                dtype=np.complex64,
            )
            + 1j
            * jax.random.normal(
                rng_uimag,
                (p, self.matrix_size, self.matrix_rank),
                dtype=np.complex64,
            )
        )
        self.ls = self.init_magnitude * jax.random.normal(
            rng_l,
            (p, self.matrix_rank),
            dtype=np.float32,
        )

    def get_output_shape(self, input_shape: DataShape) -> DataShape:
        # return (n,n) with the same PyTree structure as input_shape, so long
        # as the input shape is valid
        valid, _ = is_single_leaf(input_shape, ndim=1, is_leaf=is_shape_leaf)
        if not valid:
            raise ValueError(
                "Input shape must be a PyTree with a single leaf consisting of"
                " a 1D array, got: {input_shape}"
            )
        return jax.tree.map(
            lambda x: (self.matrix_size, self.matrix_size),
            input_shape,
            is_leaf=is_shape_leaf,
        )

    def get_hyperparameters(self) -> HyperParams:
        return {
            "matrix_size": self.matrix_size,
            "smoothing": self.smoothing,
            "init_magnitude": self.init_magnitude,
            "bias_term": self.bias_term,
        }

    def set_hyperparameters(self, hyperparams: HyperParams) -> None:
        if self.us is not None or self.ls is not None:
            raise ValueError(
                "Cannot set hyperparameters after the module has parameters"
            )

        super().set_hyperparameters(hyperparams)

    def get_params(self) -> Params:
        return (self.ls, self.us)

    def set_params(self, params: Params) -> None:
        if (
            not isinstance(params, tuple)
            or len(params) != 2
            or not all(isinstance(p, np.ndarray) for p in params)
        ):
            raise ValueError(
                "Params must be a tuple of two numpy arrays (ls, us)"
            )

        ls, us = params

        expected_shape = (
            us.shape[0] if self.us is None else self.us.shape[0],
            self.matrix_size,
            self.matrix_rank,
        )

        if us.shape != expected_shape:
            raise ValueError(
                "us must be a 3D array of shape (input_size"
                f"{'+1' if self.bias_term else ''}, matrix_size,"
                f" matrix_rank) [{expected_shape}], got {us.shape}"
            )

        expected_shape = (
            ls.shape[0] if self.ls is None else self.ls.shape[0],
            self.matrix_rank,
        )
        if ls.shape != expected_shape:
            raise ValueError(
                "ls must be a 2D array of shape (input_size"
                f"{'+1' if self.bias_term else ''}, matrix_rank) "
                f"[{expected_shape}], got {ls.shape}"
            )

        self.ls = ls
        self.us = us
