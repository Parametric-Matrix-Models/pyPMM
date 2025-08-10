from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as np

from ._regression_backing_funcs import reg_pmm_predict_func
from .basemodule import BaseModule


class AffineObservablePMM(BaseModule):
    """
    AffineObservablePMM is a module that implements the affine observable
    Parametric Matrix Model (PMM), which is useful for generalized regression
    """

    def __init__(
        self,
        matrix_size: int = None,
        num_eig: int = None,
        num_secondaries: int = None,
        output_size: int = None,
        smoothing: float = None,
        Ms: np.ndarray = None,
        Ds: np.ndarray = None,
        gs: np.ndarray = None,
        init_magnitude: float = 1e-2,
    ) -> None:
        """
        Initialize the AffineObservablePMM module. Represents a PMM which
        evaluates the affine observable PMM given by the process:

        M(x) = M0 + x1 * M1 + ... + xp * Mp
             -> (E, V) = eig(M(x)) [sorted in decreasing eigenvalue magnitude]

        z_k = g_k
            + sum_{ij}^r (
                sum_l [ |v_i^H D_{kl} v_j|^2 - 0.5 * ||D_{kl}||^2_2 ]
                )

            = g_k
            - 0.5 * r^2 * sum_l ||D_{kl}||^2_2
            + sum_{ij}^r sum_l |v_i^H D_{kl} v_j|^2

        By default this module is initialized to compute the smallest algebraic
        eigenvalue (ground state).

        Parameters
        ----------
            matrix_size
                Size of the PMM matrices (square), shorthand `n`.
            num_eig
                Number of eigenpairs to use. Shorthand `r`.
            num_secondaries
                Number of secondary (observable) matrices to use in the PMM.
                Shorthand `l`.
            output_size
                Number of output features. Shorthand `k`.
            smoothing
                Smoothing parameter, set to 0.0 to disable smoothing.
                Defaults to None/0.0.
            Ms
                Optional array of matrices M0, M1, ..., Mp that define the
                parametric Hamiltonian. Each M must be Hermitian. If not
                provided, the matrices will be randomly initialized when the
                module is compiled.
            Ds
                Optional 4D array of matrices D_{kl} that define the
                observables. Each D must be Hermitian. If not provided, the
                observables will be randomly initialized when the module is
                compiled.
            gs
                Optional vector of length ``output_size`` that defines the real
                biases for the observable. If not provided, the biases will be
                randomly initialized when the module is compiled.
            init_magnitude
                Initial magnitude for the random matrices if Ms is not
                provided, by default ``1e-2``.
        """

        # input validation
        if (
            matrix_size is None
            or num_eig is None
            or num_secondaries is None
            or output_size is None
        ):
            # module will be configured later, hopefully
            self.matrix_size = None
            self.num_eig = None
            self.num_secondaries = None
            self.output_size = None
            self.smoothing = None
            self.input_size = None
            self.Ms = None
            self.Ds = None
            self.gs = None
            self.init_magnitude = None
            return
        if matrix_size <= 0 or not isinstance(matrix_size, int):
            raise ValueError("matrix_size must be a positive integer")
        if num_eig <= 0 or not isinstance(num_eig, int):
            raise ValueError("num_eig must be a positive integer")
        if num_eig > matrix_size:
            raise ValueError(
                "num_eig must be less than or equal to matrix_size"
            )
        if num_secondaries <= 0 or not isinstance(num_secondaries, int):
            raise ValueError("num_secondaries must be a positive integer")
        if output_size <= 0 or not isinstance(output_size, int):
            raise ValueError("output_size must be a positive integer")

        # if any of the parameters are provided, they must all be provided
        if Ms is None and (Ds is not None or gs is not None):
            raise ValueError(
                "If Ds or gs are provided, Ms must also be provided"
            )
        if Ds is None and (gs is not None or Ms is not None):
            raise ValueError(
                "If gs or Ms are provided, Ds must also be provided"
            )
        if gs is None and (Ms is not None or Ds is not None):
            raise ValueError(
                "If Ms or Ds are provided, gs must also be provided"
            )
        if Ms is not None and Ds is not None and gs is not None:
            if not isinstance(Ms, np.ndarray):
                raise ValueError("Ms must be a numpy array")
            if not isinstance(Ds, np.ndarray):
                raise ValueError("Ds must be a numpy array")
            if not isinstance(gs, np.ndarray):
                raise ValueError("gs must be a numpy array")
            if Ms.shape != (Ms.shape[0], matrix_size, matrix_size):
                raise ValueError(
                    "Ms must be a 3D array of shape (input_size+1,"
                    f" matrix_size, matrix_size) [({Ms.shape[0]},"
                    f" {matrix_size}, {matrix_size})], got {Ms.shape}"
                )
            if Ds.shape != (
                output_size,
                num_secondaries,
                matrix_size,
                matrix_size,
            ):
                raise ValueError(
                    "Ds must be a 4D array of shape (output_size,"
                    " num_secondaries, matrix_size, matrix_size)"
                    f" [({output_size}, {num_secondaries}, {matrix_size},"
                    f" {matrix_size})], got {Ds.shape}"
                )
            if gs.shape != (output_size,):
                raise ValueError(
                    f"gs must be a 1D array of length {output_size}, got"
                    f" {gs.shape}"
                )
            # ensure Ms are Hermitian
            if not np.allclose(Ms, Ms.conj().transpose((0, 2, 1))):
                raise ValueError("Ms must be Hermitian matrices")
            # ensure Ds are Hermitian
            if not np.allclose(Ds, Ds.conj().transpose((0, 1, 3, 2))):
                raise ValueError("Ds must be Hermitian matrices")
            # ensure gs is real
            if not np.isreal(gs).all():
                raise ValueError("gs must be a real-valued array")

        self.matrix_size = matrix_size
        self.num_eig = num_eig
        self.num_secondaries = num_secondaries
        self.output_size = output_size
        self.smoothing = smoothing if smoothing is not None else 0.0
        self.input_size = (
            (Ms.shape[0] - 1) if Ms is not None else None
        )  # number of input features
        self.Ms = Ms  # matrices M0, M1, ..., Mp
        self.Ds = Ds
        self.gs = gs
        self.init_magnitude = init_magnitude

    def name(self) -> str:
        return (
            f"AffineObservablePMM({self.matrix_size}x{self.matrix_size},"
            f" num_eig={self.num_eig}, num_secondaries={self.num_secondaries},"
            f" output_size={self.output_size}, smoothing={self.smoothing})"
        )

    def is_ready(self) -> bool:
        return (
            self.input_size is not None
            and self.Ms is not None
            and self.Ds is not None
            and self.gs is not None
        )

    def get_num_trainable_floats(self) -> int | None:
        if not self.is_ready():
            return None

        # each matrix M is Hermitian, and so contains n * (n - 1) / 2 distinct
        # complex numbers and n distinct real numbers on the diagonal
        # the total number of trainable floats is then just n^2 per matrix
        # so Ms contributes (p + 1) * n^2 floats

        # each matrix D is also Hermitian, so it contributes k * l * n^2 floats

        # gs contributes k floats
        return (
            (self.input_size + 1) * self.matrix_size**2
            + self.output_size * self.num_secondaries * self.matrix_size**2
            + self.output_size
        )

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
        return lambda params, input_NF, training, state, rng: (
            reg_pmm_predict_func(
                params[0][0],  # A or M0
                np.array(params[0][1:]),  # Bs or M1, ..., Mp
                np.array(params[1]),  # Ds
                params[2],  # gs
                self.num_eig,  # r
                input_NF,  # X
                self.smoothing,  # smoothing
            ),
            state,  # state is not used in this module, return it unchanged
        )

    def compile(self, rng: Any, input_shape: tuple[int, ...]) -> None:
        # input shape must be 1D
        if len(input_shape) != 1:
            raise ValueError(
                f"Input shape must be 1D, got {len(input_shape)}D shape: "
                f"{input_shape}"
            )

        # if the module is already ready, just verify the input shape
        if self.is_ready():
            if self.input_size != input_shape[0]:
                raise ValueError(
                    f"Input shape {input_shape} does not match the expected "
                    f"number of features {self.input_size}"
                )
            return

        # otherwise, initialize the matrices
        self.input_size = input_shape[0]  # number of input features

        rng_Mreal, rng_Mimag, rng_Dreal, rng_Dimag, rng_g = jax.random.split(
            rng, 5
        )

        self.Ms = self.init_magnitude * (
            jax.random.normal(
                rng_Mreal,
                (self.input_size + 1, self.matrix_size, self.matrix_size),
                dtype=np.complex64,
            )
            + 1j
            * jax.random.normal(
                rng_Mimag,
                (self.input_size + 1, self.matrix_size, self.matrix_size),
                dtype=np.complex64,
            )
        )
        # ensure the matrices are Hermitian
        self.Ms = (self.Ms + self.Ms.conj().transpose((0, 2, 1))) / 2.0
        # initialize Ds
        self.Ds = self.init_magnitude * (
            jax.random.normal(
                rng_Dreal,
                (
                    self.output_size,
                    self.num_secondaries,
                    self.matrix_size,
                    self.matrix_size,
                ),
                dtype=np.complex64,
            )
            + 1j
            * jax.random.normal(
                rng_Dimag,
                (
                    self.output_size,
                    self.num_secondaries,
                    self.matrix_size,
                    self.matrix_size,
                ),
                dtype=np.complex64,
            )
        )
        # ensure the Ds are Hermitian
        self.Ds = (self.Ds + self.Ds.conj().transpose((0, 1, 3, 2))) / 2.0

        # initialize gs
        self.gs = self.init_magnitude * jax.random.normal(
            rng_g, (self.output_size,), dtype=np.float32
        )

    def get_output_shape(
        self, input_shape: tuple[int, ...]
    ) -> tuple[int, ...]:
        return (self.output_size,)

    def get_hyperparameters(self) -> dict[str, Any]:
        return {
            "matrix_size": self.matrix_size,
            "num_eig": self.num_eig,
            "num_secondaries": self.num_secondaries,
            "output_size": self.output_size,
            "input_size": self.input_size,
            "smoothing": self.smoothing,
            "init_magnitude": self.init_magnitude,
        }

    def set_hyperparameters(self, hyperparams: dict[str, Any]) -> None:
        if self.Ms is not None or self.Ds is not None or self.gs is not None:
            raise ValueError(
                "Cannot set hyperparameters after the module has parameters"
            )

        super(AffineObservablePMM, self).set_hyperparameters(hyperparams)

    def get_params(self) -> tuple[np.ndarray, ...]:
        return (self.Ms, self.Ds, self.gs)

    def set_params(self, params: tuple[np.ndarray, ...]) -> None:
        if not isinstance(params, tuple) or not all(
            isinstance(p, np.ndarray) for p in params
        ):
            raise ValueError("params must be a tuple of numpy arrays")
        if len(params) != 3:
            raise ValueError(f"Expected 3 parameter arrays, got {len(params)}")

        Ms, Ds, gs = params

        if Ms.shape != (
            self.input_size + 1,
            self.matrix_size,
            self.matrix_size,
        ):
            raise ValueError(
                "Ms must be a 3D array of shape (input_size+1, matrix_size,"
                f" matrix_size) [({self.input_size + 1}, {self.matrix_size},"
                f" {self.matrix_size})], got {Ms.shape}"
            )
        if Ds.shape != (
            self.output_size,
            self.num_secondaries,
            self.matrix_size,
            self.matrix_size,
        ):
            raise ValueError(
                "Ds must be a 4D array of shape (output_size,"
                " num_secondaries, matrix_size, matrix_size)"
                f" [({self.output_size}, {self.num_secondaries},"
                f" {self.matrix_size}, {self.matrix_size})], got {Ds.shape}"
            )
        if gs.shape != (self.output_size,):
            raise ValueError(
                f"gs must be a 1D array of length {self.output_size}, got"
                f" {gs.shape}"
            )
        # ensure Ms are Hermitian
        if not np.allclose(Ms, Ms.conj().transpose((0, 2, 1))):
            raise ValueError("Ms must be Hermitian matrices")
        # ensure Ds are Hermitian
        if not np.allclose(Ds, Ds.conj().transpose((0, 1, 3, 2))):
            raise ValueError("Ds must be Hermitian matrices")
        # ensure gs is real
        if not np.isreal(gs).all():
            raise ValueError("gs must be a real-valued array")

        self.Ms = Ms
        self.Ds = Ds
        self.gs = gs
