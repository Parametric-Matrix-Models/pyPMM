from __future__ import annotations

import jax
import jax.numpy as np
from beartype import beartype
from jaxtyping import Array, Inexact, PyTree, jaxtyped

from parametricmatrixmodels.typing import (
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


class Constant(BaseModule):
    r"""
    Module that always returns a constant value which is optionally trainable.
    """

    def __init__(
        self,
        constant: (
            PyTree[Inexact[Array, "..."], " Const"]
            | Inexact[Array, "..."]
            | float
            | complex
            | None
        ) = None,
        trainable: bool = False,
        shape: PyTree[Tuple[int, ...], " Const"] | None = None,
        init_magnitude: float = 1e-2,
        real: PyTree[bool, " Const"] | bool | None = None,
        name: str = "Constant",
    ) -> None:
        """
        Parameters
        ----------
        constant
            The constant value to return. If ``None`` and ``trainable`` is
            ``True``, the constant will be initialized randomly during
            compilation. If ``None`` and ``trainable`` is ``False``, the
            constant must be set with ``set_hyperparameters`` before use.
        trainable
            Whether the constant is trainable.
        shape
            The shape of the constant if it is trainable and not provided.
            Otherwise ignored.
        init_magnitude
            The magnitude of the random initialization if the constant is
            trainable and not provided.
        real
            Whether the constant is real-valued if it is trainable and not
            provided. If ``None``, the type is inferred from the ``constant``
            parameter if provided. If no ``constant`` is provided, the default
            is ``True``.
        name
            Custom name for this instance of the module.
        """
        # check if constant is a scalar
        if constant is not None and np.isscalar(constant):
            constant = np.array(constant)

        if constant is None and trainable and shape is None:
            raise ValueError(
                "If 'constant' is None and 'trainable' is True, "
                "'shape' must be provided."
            )

        if constant is None and real is None and trainable:
            real = jax.tree.map(lambda _: True, shape)

        if constant is not None:
            if shape is not None:
                expected_shape = jax.tree.map(lambda x: x.shape, constant)
                if expected_shape != shape:
                    raise ValueError(
                        "'shape' must match the shape of 'constant' "
                        f"if both are provided. Got {shape} and "
                        f"{expected_shape}."
                    )
            else:
                shape = jax.tree.map(lambda x: x.shape, constant)

            if real is None:
                real = jax.tree.map(lambda x: np.isrealobj(x), constant)
            elif isinstance(real, bool):
                real = jax.tree.map(lambda _: real, constant)

            # check that real matches constant
            def check_real(c, r):
                if r and not np.isrealobj(c):
                    raise ValueError(
                        "'real' must match the type of 'constant'. "
                        f"Got real={r} and constant={c}."
                    )

            jax.tree.map(check_real, constant, real)

        self.constant = constant
        self.trainable = trainable
        self.shape = shape
        self.init_magnitude = init_magnitude
        self.real = real
        self._name = name

    @property
    def name(self) -> str:
        return (
            f"{self._name}({self.shape}, real={self.real},"
            f" {'trainable' if self.trainable else 'fixed'})"
        )

    def is_ready(self) -> bool:
        return self.constant is not None

    def _get_callable(self) -> ModuleCallable:

        @jaxtyped(typechecker=beartype)
        def const_callable(
            params: Params,
            data: Data,
            training: bool,
            state: State,
            rng: Any,
        ) -> Tuple[Data, State]:
            # data is either ArrayData or a PyTree with ArrayData leaves
            # get the batch dimension in either case, which is the leading
            # dimension of any of the ArrayData leaves
            sample_leaf = jax.tree.leaves(data)[0]
            batch_size = sample_leaf.shape[0]

            if self.trainable:
                constant = params
            else:
                constant = self.constant

            constant_broadcasted = jax.tree.map(
                lambda c: np.broadcast_to(
                    c,
                    (batch_size,) + c.shape,
                ),
                constant,
            )
            return constant_broadcasted, state

        return const_callable

    def compile(self, rng: Any, input_shape: DataShape) -> None:
        if not self.trainable and not self.is_ready():
            raise ValueError(
                "Constant module is not trainable and 'constant' "
                "is not set. Please set 'constant' with "
                "'set_hyperparameters' before compiling."
            )

        if self.trainable and not self.is_ready():
            if self.real is None:
                raise ValueError(
                    "'real' must be set if 'constant' is None "
                    "and 'trainable' is True."
                )

            if isinstance(self.real, bool):
                self.real = jax.tree.map(lambda _: self.real, self.shape)

            def init_constant(cur_key, sr):
                shape, real = sr
                if real:
                    return self.init_magnitude * jax.random.normal(
                        cur_key, shape, dtype=np.float32
                    )
                else:
                    rekey, imkey = jax.random.split(cur_key)
                    return self.init_magnitude * (
                        jax.random.normal(rekey, shape, dtype=np.complex64)
                        + 1j
                        * jax.random.normal(imkey, shape, dtype=np.complex64)
                    )

            keys = jax.random.split(
                rng,
                len(
                    jax.tree.leaves(
                        self.shape,
                        is_leaf=lambda x: isinstance(x, tuple)
                        and all(isinstance(i, int) for i in x),
                    )
                ),
            )
            # give keys the same structure as shape
            keys = jax.tree.unflatten(
                jax.tree.structure(
                    self.shape,
                    is_leaf=lambda x: isinstance(x, tuple)
                    and all(isinstance(i, int) for i in x),
                ),
                keys,
            )

            shape_and_real = jax.tree.map(
                lambda s, r: (s, r),
                self.shape,
                self.real,
                is_leaf=lambda x: isinstance(x, tuple)
                and all(isinstance(i, int) for i in x),
            )

            self.constant = jax.tree.map(
                init_constant,
                keys,
                shape_and_real,
                is_leaf=lambda x: isinstance(x, tuple)
                and len(x) == 2
                and isinstance(x[0], tuple)
                and isinstance(x[1], bool),
            )

    def get_output_shape(self, input_shape: DataShape) -> DataShape:
        return self.shape

    def get_hyperparameters(self) -> HyperParams:
        return {
            **(
                {
                    "constant": self.constant,
                }
                if not self.trainable
                else {}
            ),
            "trainable": self.trainable,
            "shape": self.shape,
            "init_magnitude": self.init_magnitude,
            "real": self.real,
            "_name": self._name,
        }

    def set_hyperparameters(self, hyperparams: HyperParams) -> None:
        super(Constant, self).set_hyperparameters(hyperparams)

    def get_params(self) -> Params:
        if self.trainable:
            return self.constant
        else:
            return ()

    def set_params(self, params: Params) -> None:
        if self.trainable:
            self.constant = params
        else:
            if len(params) != 0:
                raise ValueError(
                    "Cannot set parameters for non-trainable Constant module."
                )
