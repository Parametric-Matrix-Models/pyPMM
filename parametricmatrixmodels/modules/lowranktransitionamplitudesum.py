from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as np

from .basemodule import BaseModule


class LowRankTransitionAmplitudeSum(BaseModule):
    r"""
    A module that computes the sum of transition amplitudes of low-rank
    trainable observables given an input of state vectors. The output can be
    centered by subtracting half the operator norm squared of each observable.

    To produce :math:`q` output values, given a single input of :math:`r`
    state vectors of size :math:`n` (shape ``(n, r)``), denoted by :math:`v_i`,
    :math:`i=1, \ldots, r`,
    this module uses :math:`q\times l` low-rank trainable observables
    :math:`D_11, D_12, \ldots, D_1l, D_21, \ldots, D_ql` (each of shape ``(n,
    n)``), parameterized by the sum of self-outer products of :math:`h \leq n`
    complex vectors :math:`u_i^j`, :math:`i=1, \ldots, h`, scaled by real
    values :math:`\lambda_i`,
    :math:`j=1, \ldots, q\times l` (shape ``(h, n)``) to compute the output:

    .. math::

        z_k = \sum_{i,j=1}^r &\left( \sum_{m=1}^l |v_i^H D_{km} v_j|^2\\
                             &\quad - \frac{1}{2} ||D_{km}||^2_2 \right)

    for :math:`k=1, \ldots, q`. This is equivalent to

    .. math::

        z_k &= \sum_{m=1}^l \left(
                \sum_{i,j=1}^r \left( |v_i^H D_{km} v_j|^2 \right)\\
                &\quad - \frac{r^2}{2} ||D_{km}||^2_2 \right)

    where :math:`||\cdot||_2` is the operator 2-norm (largest singular value)
    so for Hermitian :math:`D`, :math:`||D||_2` is the largest absolute
    eigenvalue.

    The :math:`-\frac{1}{2} ||D_{km}||^2_2` term centers the value of each term
    and can be disabled by setting the ``centered`` parameter to ``False``.

    Each observable :math:`D_{km}` is defined as

    .. math::

        D_{km} = \sum_{i=1}^h \lambda_i u_i^{km} (u_i^{km})^H

    .. warning::
        This module assumes that the input state vectors are normalized. If
        they are not, the output values will be scaled by the square of the
        norm of the input vectors.

    .. warning::
        Even though the math shows that the centering term should be multiplied
        by :math:`r^2`, in practice this doesn't work well and instead setting
        the centering term to :math:`\frac{1}{2} ||D_{km}||^2_2` works much
        better. This non-:math:`r^2` scaling is used here.

    See Also
    --------

    TransitionAmplitudeSum
        A similar module that uses full-rank observables.

    """

    def __init__(
        self,
        rank: int = None,
        num_observables: int = None,
        output_size: int = None,
        lambdas: np.ndarray = None,
        us: np.ndarray = None,
        init_magnitude: float = 1e-2,
        centered: bool = True,
    ) -> None:
        r"""
        Initialize the module.

        Parameters
        ----------
            rank
                Rank of each observable matrix, shorthand :math:`h`.
            num_observables
                Number of observable matrices, shorthand :math:`l`.
            output_size
                Number of output features, shorthand :math:`q`.
            lambdas
                Optional 3D array of real values :math:`\lambda_{qlh}` that
                scale the self-outer product sums that define the
                observables. If not provided, the values will be
                initialized randomly when the module is compiled.
            us
                Optional 4D array of complex vectors :math:`u_{qlh}` that
                define the observables via self-outer product sums. If not
                provided, the vectors will be randomly initialized when the
                module is compiled.
            init_magnitude
                Initial magnitude for the random matrices if Ms is not
                provided. Default ``1e-2``.
            centered
                Whether to center the output by subtracting half the operator
                norm squared of each observable. Default ``True``.
        """

        if lambdas is not None:
            if not isinstance(lambdas, np.ndarray):
                raise ValueError("lambdas must be a numpy array")
            if lambdas.ndim != 3:
                raise ValueError(
                    f"lambdas must be a 3D array, got {lambdas.ndim}D array"
                )
            if rank is not None and lambdas.shape[2] != rank:
                raise ValueError(
                    "If provided, rank must match the shape of axis 2 of"
                    f" lambdas (got {rank} and {lambdas.shape[2]})"
                )
            if (
                num_observables is not None
                and lambdas.shape[1] != num_observables
            ):
                raise ValueError(
                    "If provided, num_observables must match the shape of"
                    f" axis 1 of lambdas (got {num_observables} and"
                    f" {lambdas.shape[1]})"
                )
            if output_size is not None and lambdas.shape[0] != output_size:
                raise ValueError(
                    "If provided, output_size must match the shape of axis 0"
                    f" of lambdas (got {output_size} and {lambdas.shape[0]})"
                )

        if us is not None:
            if not isinstance(us, np.ndarray):
                raise ValueError("us must be a numpy array")
            if us.ndim != 4:
                raise ValueError(
                    f"us must be a 4D array, got {us.ndim}D array"
                )
            if rank is not None and us.shape[2] != rank:
                raise ValueError(
                    "If provided, rank must match the shape of axis 2 of us"
                    f" (got {rank} and {us.shape[2]})"
                )

            if num_observables is not None and us.shape[1] != num_observables:
                raise ValueError(
                    "If provided, num_observables must match the shape of"
                    f" axis 1 of us (got {num_observables} and {us.shape[1]})"
                )

            if output_size is not None and us.shape[0] != output_size:
                raise ValueError(
                    "If provided, output_size must match the shape of axis 0"
                    f" of us (got {output_size} and {us.shape[0]})"
                )

        self.num_observables = (
            us.shape[1]
            if us is not None
            else lambdas.shape[1] if lambdas is not None else num_observables
        )
        self.rank = (
            us.shape[2]
            if us is not None
            else lambdas.shape[2] if lambdas is not None else rank
        )
        self.output_size = (
            us.shape[0]
            if us is not None
            else lambdas.shape[0] if lambdas is not None else output_size
        )
        self.lambdas = lambdas
        self.us = us
        self.init_magnitude = init_magnitude
        self.centered = centered

    def name(self) -> str:
        return (
            f"LowRankTransitionAmplitudeSum(output_size={self.output_size},"
            f" rank={self.rank}, num_observables={self.num_observables},"
            f" centered={self.centered})"
        )

    def is_ready(self) -> bool:
        return self.us is not None and self.lambdas is not None

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

        # function for a single input, which will be vmapped over for the batch
        def _single(Ds: np.ndarray, V: np.ndarray) -> np.ndarray:

            Z = np.einsum("ai,klab,bj->klij", V.conj(), Ds, V)
            Z = np.sum(np.abs(Z) ** 2, axis=(1, 2, 3))

            if self.centered:
                # TODO: this doesn't use the predicted r^2 scaling, which
                # doesn't work well in practice, why is this?
                norm_term = 0.5 * np.sum(
                    np.linalg.norm(Ds, axis=(2, 3), ord=2) ** 2, axis=1
                )

                return Z - norm_term
            else:
                return Z

        def _callable(
            params: tuple[np.ndarray, ...],
            inputs: np.ndarray,
            training: bool,
            states: tuple[np.ndarray, ...],
            rng: Any,
        ) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:

            (
                lambdas,
                us,
            ) = params

            # construct Ds from us
            Ds = np.einsum("qlh,qlhi,qlhj->qlij", lambdas.real, us, us.conj())

            outputs = jax.vmap(_single, in_axes=(None, 0))(Ds, inputs)
            return outputs, states

        return _callable

    def compile(self, rng: Any, input_shape: tuple[int, ...]) -> None:
        if self.num_observables is None or self.output_size is None:
            raise ValueError(
                "num_observables and output_size must be set before"
                " compiling the module"
            )

        # input shape must be 2D
        if len(input_shape) != 2:
            raise ValueError(
                f"Input shape must be 2D, got {len(input_shape)}D shape: "
                f"{input_shape}"
            )

        # if the module is already ready, just verify the input shape, which
        # should be (n, r) where n is the number of components in the state
        # vector
        if self.is_ready():
            if input_shape[0] != self.us.shape[3]:
                raise ValueError(
                    f"Input shape {input_shape} does not match the expected "
                    "shape based on the provided us array of shape "
                    f"{self.us.shape}."
                )
            return

        # otherwise, initialize the matrices
        n, _ = input_shape

        rng_lambdas, rng_ureal, rng_uimag = jax.random.split(rng, 3)

        # initialize lambdas
        self.lambdas = self.init_magnitude * jax.random.normal(
            rng_lambdas,
            (self.output_size, self.num_observables, self.rank),
            dtype=np.float32,
        )

        # initialize us
        self.us = self.init_magnitude * (
            jax.random.normal(
                rng_ureal,
                (
                    self.output_size,
                    self.num_observables,
                    self.rank,
                    n,
                ),
                dtype=np.complex64,
            )
            + 1j
            * jax.random.normal(
                rng_uimag,
                (
                    self.output_size,
                    self.num_observables,
                    self.rank,
                    n,
                ),
                dtype=np.complex64,
            )
        )

    def get_output_shape(
        self, input_shape: tuple[int, ...]
    ) -> tuple[int, ...]:
        return (self.output_size,)

    def get_hyperparameters(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "num_observables": self.num_observables,
            "output_size": self.output_size,
            "init_magnitude": self.init_magnitude,
            "centered": self.centered,
        }

    def set_hyperparameters(self, hyperparams: dict[str, Any]) -> None:
        if self.us is not None:
            raise ValueError(
                "Cannot set hyperparameters after the module has parameters"
            )

        super(LowRankTransitionAmplitudeSum, self).set_hyperparameters(
            hyperparams
        )

    def get_params(self) -> tuple[np.ndarray, ...]:
        return (
            self.lambdas,
            self.us,
        )

    def set_params(self, params: tuple[np.ndarray, ...]) -> None:
        if not isinstance(params, tuple) or not all(
            isinstance(p, np.ndarray) for p in params
        ):
            raise ValueError("params must be a tuple of numpy arrays")
        if len(params) != 2:
            raise ValueError(f"Expected 2 parameter array, got {len(params)}")

        lambdas = params[0]
        us = params[1]

        if lambdas.ndim != 3:
            raise ValueError(
                f"lambdas must be a 3D array, got {lambdas.ndim}D array"
            )
        if us.ndim != 4:
            raise ValueError(f"us must be a 4D array, got {us.ndim}D array")

        _, _, _, matrix_size = us.shape

        if us.shape != (
            self.output_size,
            self.num_observables,
            self.rank,
            matrix_size,
        ):
            raise ValueError(
                "us must be a 4D array of shape (output_size,"
                " num_observables, rank, matrix_size)"
                f" [({self.output_size}, {self.num_observables},"
                f" {self.rank}, {matrix_size})], got {us.shape}"
            )
        if lambdas.shape != (
            self.output_size,
            self.num_observables,
            self.rank,
        ):
            raise ValueError(
                "lambdas must be a 3D array of shape (output_size,"
                " num_observables, rank)"
                f" [({self.output_size}, {self.num_observables},"
                f" {self.rank})], got {lambdas.shape}"
            )

        self.lambdas = lambdas
        self.us = us
