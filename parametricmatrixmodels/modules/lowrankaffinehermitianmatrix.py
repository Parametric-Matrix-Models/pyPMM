from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as np

from ._smoothing import exact_smoothing_matrix
from .basemodule import BaseModule


class LowRankAffineHermitianMatrix(BaseModule):
    r"""
    Module that builds a parametric hermitian matrix from an affine function of
    the input features with low-rank matrices.

    :math:`M(x) = M_0 + x_1 M_1 + ... + x_p M_p + s C`
    where :math:`M_0, M_1, ..., M_p` are (trainable) low-rank Hermitian
    matrices, :math:`x_1, ..., x_p` are the input features, :math:`s` is the
    smoothing hyperparameter, and :math:`C` is a matrix that is computed
    as the imaginary unit times the sum of the commutators of all the
    :math:`M_i` matrices, in an efficient way using cumulative sums and the
    linearity of the commutator:

    .. math::

        C &= i\sum_{\substack{i,j\\i\neq j}} \left[M_i, M_j\right] \\
          &= i\sum_{\substack{i\\i\neq j}}
             \left[M_i, \sum_k^j M_k\right]

    Each :math:`M_i` is a low-rank Hermitian matrix, which can be parametrized
    as :math:`M_i = sum_k^r u_k^i (u_k^i)^H` where :math:`u_k^i` are a set of
    :math:`r` complex vectors of size :math:`n`, and :math:`r` is the rank of
    the matrix.

    See Also
    --------
    AffineHermitianMatrix
        Full-rank version of this module that uses full-rank Hermitian matrices
        instead of low-rank ones.
    """

    def __init__(
        self,
        matrix_size: int = None,
        rank: int = None,
        smoothing: float = None,
        us: np.ndarray = None,
        init_magnitude: float = 1e-2,
        bias_term: bool = True,
        flatten: bool = False,
    ) -> None:
        """
        Create an ``LowRankAffineHermitianMatrix`` module.

        Parameters
        ----------
            matrix_size
                Size of the PMM matrices (square), shorthand :math:`n`.
            rank
                Rank of the low-rank Hermitian matrices, shorthand :math:`r`.
                Must be a positive integer less than or equal to
                ``matrix_size``.
            smoothing
                Optional smoothing parameter. Set to ``0.0`` to disable
                smoothing. Default is ``None``/``0.0`` (no smoothing).
            us
                Optional array of shape `(input_size+1, rank, matrix_size)` (if
                ``bias_term`` is ``True``) or `(input_size, rank, matrix_size)`
                (if ``bias_term`` is ``False``), containing the `u_k^i` complex
                vectors used to construct the low-rank Hermitian matrices. If
                not provided, the vectors will be initialized randomly when the
                module is compiled.
            init_magnitude
                Optional initial magnitude of the random matrices, used when
                initializing the module. Default is ``1e-2``.
            bias_term
                If ``True``, include the bias term :math:`M_0` in the affine
                matrix. Default is ``True``.
            flatten
                If ``True``, the *output* will be flattened to a 1D array.
                Useful when combining with ``SubsetModule`` or other modules in
                order to avoid ragged arrays. Default is ``False``.

        """

        # input validation
        if matrix_size is not None and (
            matrix_size <= 0 or not isinstance(matrix_size, int)
        ):
            raise ValueError("matrix_size must be a positive integer")
        if us is not None:
            if not isinstance(us, np.ndarray):
                raise ValueError("us must be a numpy array")
            matrix_size = matrix_size or us.shape[2]
            rank = rank or us.shape[1]
            if us.shape != (us.shape[0], rank, matrix_size):
                raise ValueError(
                    "us must be a 3D array of shape (input_size"
                    f" {'+1' if bias_term else ''}, rank, matrix_size)"
                    f" [{(us.shape[0], rank, matrix_size)}], got {us.shape}"
                )

        self.matrix_size = matrix_size
        self.rank = rank
        self.smoothing = smoothing if smoothing is not None else 0.0
        self.bias_term = bias_term
        self.us = us
        self.init_magnitude = init_magnitude
        self.flatten = flatten

    def name(self) -> str:
        return (
            "LowRankAffineHermitianMatrix("
            f"{self.matrix_size}x{self.matrix_size},"
            f" rank={self.rank},"
            f" smoothing={self.smoothing},"
            f"{'' if self.bias_term else ' no bias,'}"
            f"{' FLATTENED' if self.flatten else ''})"
        )

    def is_ready(self) -> bool:
        return self.us is not None

    def get_num_trainable_floats(self) -> int | None:
        if not self.is_ready():
            return None

        return 2 * self.us.size

    def _get_callable(self) -> Callable:
        def lr_affine_hermitian_matrix(
            params: tuple[np.ndarray, ...],
            input_NF: np.ndarray,
            training: bool,
            state: tuple[np.ndarray, ...],
            rng: Any,
        ) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:

            us = params[0]

            # compute Ms from us: M_i = sum_k^r u_k^i (u_k^i)^H
            Ms = np.einsum("irk,irl->ikl", us, us.conj())

            # Hermiticity is guaranteed by the construction

            if self.bias_term:
                M = Ms[0][None, :, :] + np.einsum(
                    "ni,ijk->njk", input_NF.astype(Ms.dtype), Ms[1:]
                )
            else:
                M = np.einsum("ni,ijk->njk", input_NF.astype(Ms.dtype), Ms)

            if self.smoothing != 0.0:
                M += self.smoothing * exact_smoothing_matrix(Ms)[None, :, :]

            if self.flatten:
                # preserve batch dimension
                return (M.reshape(M.shape[0], -1), state)
            else:
                return (M, state)

        return lr_affine_hermitian_matrix

    def compile(self, rng: Any, input_shape: tuple[int, ...]) -> None:
        # input shape must be 1D
        if len(input_shape) != 1:
            raise ValueError(
                f"Input shape must be 1D, got {len(input_shape)}D shape: "
                f"{input_shape}"
            )

        # number of matrices is number of features + 1 (bias) if bias is used
        p = input_shape[0] + 1 if self.bias_term else input_shape[0]

        # if the module is already ready, just verify the input shape
        if self.is_ready():
            if self.us.shape[0] != p:
                raise ValueError(
                    f"Input shape {input_shape} does not match the expected "
                    f"number of features {self.us.shape[0] - 1} "
                )
            return

        rng_ureal, rng_uimag = jax.random.split(rng, 2)

        self.us = self.init_magnitude * (
            jax.random.normal(
                rng_ureal,
                (p, self.rank, self.matrix_size),
                dtype=np.complex64,
            )
            + 1j
            * jax.random.normal(
                rng_uimag,
                (p, self.rank, self.matrix_size),
                dtype=np.complex64,
            )
        )

    def get_output_shape(
        self, input_shape: tuple[int, ...]
    ) -> tuple[int, ...]:
        if self.flatten:
            return (self.matrix_size**2,)
        else:
            return (self.matrix_size, self.matrix_size)

    def get_hyperparameters(self) -> dict[str, Any]:
        return {
            "matrix_size": self.matrix_size,
            "rank": self.rank,
            "smoothing": self.smoothing,
            "init_magnitude": self.init_magnitude,
            "flatten": self.flatten,
            "bias_term": self.bias_term,
        }

    def set_hyperparameters(self, hyperparams: dict[str, Any]) -> None:
        if self.us is not None:
            raise ValueError(
                "Cannot set hyperparameters after the module has parameters"
            )

        super(LowRankAffineHermitianMatrix, self).set_hyperparameters(
            hyperparams
        )

    def get_params(self) -> tuple[np.ndarray, ...]:
        return (self.us,)

    def set_params(self, params: tuple[np.ndarray, ...]) -> None:
        if not isinstance(params, tuple) or not all(
            isinstance(p, np.ndarray) for p in params
        ):
            raise ValueError("params must be a tuple of numpy arrays")
        if len(params) != 1:
            raise ValueError(f"Expected 1 parameter arrays, got {len(params)}")

        us = params[0]

        expected_shape = (
            us.shape[0] if self.us is None else self.us.shape[0],
            self.rank,
            self.matrix_size,
        )

        if us.shape != expected_shape:
            raise ValueError(
                "us must be a 3D array of shape (input_size"
                f"{'+1' if self.bias_term else ''}, matrix_size,"
                f" matrix_size) [{expected_shape}], got {us.shape}"
            )

        self.us = us
