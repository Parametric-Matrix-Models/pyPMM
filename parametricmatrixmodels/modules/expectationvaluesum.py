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
from .basemodule import BaseModule


class ExpectationValueSum(BaseModule):
    r"""
    A module that computes the sum of expectation values of trainable
    observables given an input of state vectors. The output can be centered by
    subtracting half the operator norm squared of each observable.

    To produce :math:`q` output values, given a single input of :math:`r`
    state vectors of size :math:`n` (shape ``(n, r)``), denoted by :math:`v_i`,
    :math:`i=1, \ldots, r`,
    this module uses :math:`q\times l` trainable observables
    :math:`D_{11}, D_{12}, \ldots, D_{1l}, D_{21}, \ldots, D_{ql}` (each of
    shape ``(n, n)``) to compute the output:

    .. math::

        z_k = \sum_{i=1}^r &\left( \sum_{m=1}^l v_i^H D_{km} v_i\\
                             &\quad - \frac{1}{2} ||D_{km}||^2_2 \right)

    for :math:`k=1, \ldots, q`. This is equivalent to

    .. math::

        z_k &= \sum_{m=1}^l \left(
                \sum_{i=1}^r \left( v_i^H D_{km} v_j^2 \right)\\
                &\quad - \frac{r}{2} ||D_{km}||^2_2 \right)

    where :math:`||\cdot||_2` is the operator 2-norm (largest singular value)
    so for Hermitian :math:`D`, :math:`||D||_2` is the largest absolute
    eigenvalue.

    The :math:`-\frac{1}{2} ||D_{km}||^2_2` term centers the value of each term
    and can be disabled by setting the ``centered`` parameter to ``False``.

    .. warning::
        This module assumes that the input state vectors are normalized. If
        they are not, the output values will be scaled by the square of the
        norm of the input vectors.

    .. warning::
        Even though the math shows that the centering term should be multiplied
        by :math:`r`, in practice this doesn't work well and instead setting
        the centering term to :math:`\frac{1}{2} ||D_{km}||^2_2` works much
        better. This non-:math:`r` scaling is used here.

    See Also
    --------

    TransitionAmplitudeSum
        A similar module that computes the sum of transition amplitudes
        (instead of expectation values) of trainable observables.

    LowRankExpectationValueSum
        A similar module that uses low-rank observables to reduce the number of
        trainable parameters.

    """

    def __init__(
        self,
        num_observables: int | None = None,
        output_size: int | None = None,
        Ds: Inexact[Array, "q l n n"] | None = None,
        init_magnitude: float = 1e-2,
        centered: bool = True,
    ) -> None:
        """
        Initialize the module.

        Parameters
        ----------
            num_observables
                Number of observable matrices, shorthand :math:`l`.
            output_size
                Number of output features, shorthand :math:`q`.
            Ds
                Optional 4D array of matrices :math:`D_{ql}` that define the
                observables. Each :math:`D` must be Hermitian. If not provided,
                the observables will be randomly initialized when the module is
                compiled.
            init_magnitude
                Initial magnitude for the random matrices if Ms is not
                provided. Default ``1e-2``.
            centered
                Whether to center the output by subtracting half the operator
                norm squared of each observable. Default ``True``.
        """

        if Ds is not None:
            if not isinstance(Ds, np.ndarray):
                raise ValueError("Ds must be a numpy array")
            if Ds.ndim != 4:
                raise ValueError(
                    f"Ds must be a 4D array, got {Ds.ndim}D array"
                )
            if Ds.shape[2] != Ds.shape[3]:
                raise ValueError(
                    "The last two dimensions of Ds must be equal"
                    f" (got {Ds.shape[2]} and {Ds.shape[3]})"
                )
            # ensure Ds are Hermitian
            if not np.allclose(Ds, Ds.conj().transpose((0, 1, 3, 2))):
                raise ValueError("Ds must be Hermitian matrices")

            if num_observables is not None and Ds.shape[1] != num_observables:
                raise ValueError(
                    "If provided, num_observables must match the shape of"
                    f" axis 1 of Ds (got {num_observables} and {Ds.shape[1]})"
                )

            if output_size is not None and Ds.shape[0] != output_size:
                raise ValueError(
                    "If provided, output_size must match the shape of axis 0"
                    f" of Ds (got {output_size} and {Ds.shape[0]})"
                )

        self.num_observables = (
            Ds.shape[1] if Ds is not None else num_observables
        )
        self.output_size = Ds.shape[0] if Ds is not None else output_size
        self.Ds = Ds
        self.init_magnitude = init_magnitude
        self.centered = centered

    @property
    def name(self) -> str:
        return (
            f"ExpectationValueSum(output_size={self.output_size},"
            f" num_observables={self.num_observables},"
            f" centered={self.centered})"
        )

    def is_ready(self) -> bool:
        return self.Ds is not None

    def get_num_trainable_floats(self) -> int | None:
        if not self.is_ready():
            return None

        # each matrix D is Hermitian, so it contributes q * l * n^2 floats

        q, l, n, _ = self.Ds.shape

        return q * l * n * n

    def _get_callable(self) -> ModuleCallable:

        # function for a single input, which will be vmapped over for the batch
        def _single(Ds: np.ndarray, V: np.ndarray) -> np.ndarray:

            Z = np.einsum("ai,klab,bi->k", V.conj(), Ds, V).real

            if self.centered:
                # TODO: this doesn't use the predicted r scaling, which
                # doesn't work well in practice, why is this?
                norm_term = 0.5 * np.sum(
                    np.linalg.norm(Ds, axis=(2, 3), ord=2) ** 2, axis=1
                )

                return Z - norm_term
            else:
                return Z

        @jaxtyped(typechecker=beartype)
        def _callable(
            params: Params,
            data: Data,
            training: bool,
            state: State,
            rng: Any,
        ) -> Tuple[Data, State]:
            Ds = params

            # force Hermiticity
            Ds = (Ds + Ds.conj().transpose((0, 1, 3, 2))) / 2.0

            # preserve the tree structure of the input
            # compile will have validated that there is only one leaf, and that
            # leaf is a 2D array of shape (n, r)

            outputs = jax.tree.map(
                lambda x: jax.vmap(_single, in_axes=(None, 0))(Ds, x), data
            )
            return outputs, state

        return _callable

    def compile(self, rng: Any, input_shape: DataShape) -> None:
        if self.num_observables is None or self.output_size is None:
            raise ValueError(
                "num_observables and output_size must be set before"
                " compiling the module"
            )

        n = self.Ds.shape[2] if self.Ds is not None else None

        valid, leaf = is_single_leaf(
            input_shape, ndim=2, shape=(n, None), is_leaf=is_shape_leaf
        )
        if not valid:
            raise ValueError(
                "Input shape must be a PyTree with a single leaf consisting "
                "of a 2D array with leading dimension size matching the "
                f"observable matrices. Got input shape: {input_shape}"
            )

        # otherwise, initialize the matrices
        n, _ = leaf

        rng_Dreal, rng_Dimag = jax.random.split(rng, 2)

        # initialize Ds
        self.Ds = self.init_magnitude * (
            jax.random.normal(
                rng_Dreal,
                (
                    self.output_size,
                    self.num_observables,
                    n,
                    n,
                ),
                dtype=np.complex64,
            )
            + 1j
            * jax.random.normal(
                rng_Dimag,
                (
                    self.output_size,
                    self.num_observables,
                    n,
                    n,
                ),
                dtype=np.complex64,
            )
        )
        # ensure the Ds are Hermitian
        self.Ds = (self.Ds + self.Ds.conj().transpose((0, 1, 3, 2))) / 2.0

    def get_output_shape(self, input_shape: DataShape) -> DataShape:
        n = self.Ds.shape[2] if self.Ds is not None else None
        valid, _ = is_single_leaf(
            input_shape, ndim=2, shape=(n, None), is_leaf=is_shape_leaf
        )
        if not valid:
            raise ValueError(
                "Input shape must be a PyTree with a single leaf consisting "
                "of a 2D array with leading dimension size matching the "
                f"observable matrices. Got input shape: {input_shape}"
            )
        # preserve the tree structure of the input
        return jax.tree.map(
            lambda x: (self.output_size,), input_shape, is_leaf=is_shape_leaf
        )

    def get_hyperparameters(self) -> HyperParams:
        return {
            "num_observables": self.num_observables,
            "output_size": self.output_size,
            "init_magnitude": self.init_magnitude,
            "centered": self.centered,
        }

    def set_hyperparameters(self, hyperparams: HyperParams) -> None:
        if self.Ds is not None:
            raise ValueError(
                "Cannot set hyperparameters after the module has parameters"
            )

        super().set_hyperparameters(hyperparams)

    def get_params(self) -> Params:
        return self.Ds

    def set_params(self, params: Params) -> None:
        if not isinstance(params, np.ndarray):
            raise ValueError("params must be an array")

        if params.ndim != 4:
            raise ValueError(
                f"Ds must be a 4D array, got {params.ndim}D array"
            )

        _, _, matrix_size, _ = params.shape

        if params.shape != (
            self.output_size,
            self.num_observables,
            matrix_size,
            matrix_size,
        ):
            raise ValueError(
                "Ds must be a 4D array of shape (output_size,"
                " num_observables, matrix_size, matrix_size)"
                f" [({self.output_size}, {self.num_observables},"
                f" {matrix_size}, {matrix_size})], got {params.shape}"
            )
        # ensure Ds are Hermitian
        if not np.allclose(params, params.conj().transpose((0, 1, 3, 2))):
            raise ValueError("Ds must be Hermitian matrices")

        self.Ds = params
