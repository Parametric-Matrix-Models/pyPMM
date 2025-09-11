from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as np

from ._smoothing import exact_smoothing_matrix
from .basemodule import BaseModule


class AffineHermitianMatrix(BaseModule):
    """
    Module that builds a parametric hermitian matrix that is affine in the
    input features.

    :math:`M(x) = M_0 + x_1 M_1 + ... + x_p M_p + s C`
    where :math:`M_0, M_1, ..., M_p` are (trainable) Hermitian matrices,
    :math:`x_1, ..., x_p` are the input features, :math:`s` is the
    smoothing hyperparameter, and :math:`C` is a matrix that is computed
    as the imaginary unit times the sum of the commutators of all the
    :math:`M_i` matrices, in an efficient way using cumulative sums and the
    linearity of the commutator:

    .. math::

        C &= i\\sum_{\\substack{i,j\\\\i\\neq j}} \\left[M_i, M_j\\right] \\\\
          &= i\\sum_{\\substack{i\\\\i\\neq j}}
             \\left[M_i, \\sum_k^j M_k\\right]

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
        matrix_size: int = None,
        smoothing: float = None,
        Ms: np.ndarray = None,
        init_magnitude: float = 1e-2,
        bias_term: bool = True,
        flatten: bool = False,
    ) -> None:
        """
        Create an ``AffineHermitianMatrix`` module.

        Parameters
        ----------
            matrix_size
                Size of the PMM matrices (square), shorthand :math:`n`.
            smoothing
                Optional smoothing parameter. Set to ``0.0`` to disable
                smoothing. Default is ``None``/``0.0`` (no smoothing).
            Ms
                Optional array of matrices :math:`M_0, M_1, ..., M_p` that
                define the parametric affine matrix. Each :math:`M_i` must be
                Hermitian. If not provided, the matrices will be randomly
                initialized when the module is compiled. Default is ``None``.
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
        if Ms is not None:
            if not isinstance(Ms, np.ndarray):
                raise ValueError("Ms must be a numpy array")
            matrix_size = matrix_size or Ms.shape[1]
            if Ms.shape != (Ms.shape[0], matrix_size, matrix_size):
                raise ValueError(
                    "Ms must be a 3D array of shape (input_size+1,"
                    f" matrix_size, matrix_size) [({Ms.shape[0]},"
                    f" {matrix_size}, {matrix_size})], got {Ms.shape}"
                )
            # ensure Ms are Hermitian
            if not np.allclose(Ms, Ms.conj().transpose((0, 2, 1))):
                raise ValueError("Ms must be Hermitian matrices")

        self.matrix_size = matrix_size
        self.smoothing = smoothing if smoothing is not None else 0.0
        self.bias_term = bias_term
        self.Ms = Ms  # matrices M0, M1, ..., Mp
        self.init_magnitude = init_magnitude
        self.flatten = flatten

    def name(self) -> str:
        return (
            f"AffineHermitianMatrix({self.matrix_size}x{self.matrix_size},"
            f" smoothing={self.smoothing}"
            f"{'' if self.bias_term else ', no bias'}"
            f"{', FLATTENED' if self.flatten else ''})"
        )

    def is_ready(self) -> bool:
        return self.Ms is not None

    def get_num_trainable_floats(self) -> int | None:
        if not self.is_ready():
            return None

        # each matrix M is Hermitian, and so contains n * (n - 1) / 2 distinct
        # complex numbers and n distinct real numbers on the diagonal
        # the total number of trainable floats is then just n^2 per matrix
        # so Ms contributes (p + 1) * n^2 floats

        return self.Ms.size

    def _get_callable(self) -> Callable:
        def affine_hermitian_matrix(
            params: tuple[np.ndarray, ...],
            input_NF: np.ndarray,
            training: bool,
            state: tuple[np.ndarray, ...],
            rng: Any,
        ) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:

            Ms = params[0]

            # enforce Hermitian matrices
            Ms = (Ms + Ms.conj().transpose((0, 2, 1))) / 2.0

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

        return affine_hermitian_matrix

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
            if self.Ms.shape[0] != p:
                raise ValueError(
                    f"Input shape {input_shape} does not match the expected "
                    f"number of features {self.Ms.shape[0] - 1} "
                )
            return

        rng_Mreal, rng_Mimag = jax.random.split(rng, 2)

        self.Ms = self.init_magnitude * (
            jax.random.normal(
                rng_Mreal,
                (p, self.matrix_size, self.matrix_size),
                dtype=np.complex64,
            )
            + 1j
            * jax.random.normal(
                rng_Mimag,
                (p, self.matrix_size, self.matrix_size),
                dtype=np.complex64,
            )
        )
        # ensure the matrices are Hermitian
        self.Ms = (self.Ms + self.Ms.conj().transpose((0, 2, 1))) / 2.0

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
            "smoothing": self.smoothing,
            "init_magnitude": self.init_magnitude,
            "flatten": self.flatten,
            "bias_term": self.bias_term,
        }

    def set_hyperparameters(self, hyperparams: dict[str, Any]) -> None:
        if self.Ms is not None:
            raise ValueError(
                "Cannot set hyperparameters after the module has parameters"
            )

        super(AffineHermitianMatrix, self).set_hyperparameters(hyperparams)

    def get_params(self) -> tuple[np.ndarray, ...]:
        return (self.Ms,)

    def set_params(self, params: tuple[np.ndarray, ...]) -> None:
        if not isinstance(params, tuple) or not all(
            isinstance(p, np.ndarray) for p in params
        ):
            raise ValueError("params must be a tuple of numpy arrays")
        if len(params) != 1:
            raise ValueError(f"Expected 1 parameter arrays, got {len(params)}")

        Ms = params[0]

        expected_shape = (
            Ms.shape[0] if self.Ms is None else self.Ms.shape[0],
            self.matrix_size,
            self.matrix_size,
        )

        if Ms.shape != expected_shape:
            raise ValueError(
                "Ms must be a 3D array of shape (input_size"
                f"{'+1' if self.bias_term else ''}, matrix_size,"
                f" matrix_size) [{expected_shape}], got {Ms.shape}"
            )
        # ensure Ms are Hermitian
        if not np.allclose(Ms, Ms.conj().transpose((0, 2, 1))):
            raise ValueError("Ms must be Hermitian matrices")

        self.Ms = Ms
