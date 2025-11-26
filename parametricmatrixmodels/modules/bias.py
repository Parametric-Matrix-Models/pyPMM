from __future__ import annotations

import jax
import jax.numpy as np
from beartype import beartype
from jaxtyping import Array, Inexact, PyTree, jaxtyped

from ..tree_util import is_shape_leaf
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


class Bias(BaseModule):
    r"""
    A simple bias module that adds a (trainable by default) bias array
    (default) or scalar to the input. Can be real (default) or complex-valued.

    If the input is a ``PyTree`` of arrays, the same bias will be added to
    each leaf array and therefore the bias shape must match the shape of each
    leaf array (or be a scalar).
    """

    def __init__(
        self,
        bias: (
            PyTree[Inexact[Array, "..."]]
            | Inexact[Array, "..."]
            | float
            | complex
            | None
        ) = None,
        init_magnitude: float = 1e-2,
        real: bool = True,
        scalar: bool = False,
        trainable: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        bias
            Bias array, scalar, or PyTree of arrays. If None, the shape will be
            inferred at compile time and values will be initialized randomly
        init_magnitude
            Magnitude for the random initialization of the bias.
            Default is ``1e-2``.
        real
            If ``True``, the biases will be real-valued. If
            ``False``, they will be complex-valued. Default is ``True``.
        scalar
            If ``True`` the bias will be a scalar shared across all input
            features. If ``False``, the bias will be a array with one entry
            per input feature. Default is ``False``.
        trainable
            If ``True``, the bias will be trainable. Default is ``True``.
        """
        self.bias = bias
        self.init_magnitude = init_magnitude
        self.real = real
        self.scalar = scalar
        self.trainable = trainable

        if self.bias is not None:
            # input validation
            if self.scalar and not np.isscalar(self.bias):
                raise ValueError(
                    "If scalar is True, bias must be a scalar or None"
                )
            if not self.scalar and not isinstance(
                jax.tree.leaves(self.bias)[0], np.ndarray
            ):
                raise ValueError(
                    "If scalar is False, bias must be a PyTree of numpy "
                    "arrays, a numpy array, or None"
                )
            if self.real and not jax.tree.all(
                jax.tree.map(np.isrealobj, self.bias)
            ):
                raise ValueError("Bias must be real-valued for a real module")
            if not self.real and jax.tree.all(
                jax.tree.map(np.isrealobj, self.bias)
            ):
                raise ValueError(
                    "Bias must have at least one complex-valued leaf for a"
                    " complex module"
                )

            if self.scalar:
                self.bias = np.array(self.bias).reshape((1,))

    @property
    def name(self) -> str:
        return f"Bias(real={self.real})"

    def is_ready(self) -> bool:
        return self.bias is not None

    def _get_callable(self) -> ModuleCallable:

        if isinstance(self.bias, np.ndarray):

            @jaxtyped(typechecker=beartype)
            def bias_callable(
                params: Params,
                data: Data,
                training: bool,
                state: State,
                rng: Any,
            ) -> Tuple[Data, State]:
                # tree map over data to add bias
                bias = params

                def add_bias(x: np.ndarray) -> np.ndarray:
                    return x + bias

                output = jax.tree.map(add_bias, data)
                return output, state

        else:

            @jaxtyped(typechecker=beartype)
            def bias_callable(
                params: Params,
                data: Data,
                training: bool,
                state: State,
                rng: Any,
            ) -> Tuple[Data, State]:
                biases = params

                def add_bias(x: np.ndarray, b: np.ndarray) -> np.ndarray:
                    return x + b

                output = jax.tree.map(add_bias, data, biases)
                return output, state

        return bias_callable

    def compile(self, rng: Any, input_shape: DataShape) -> None:
        # if the module is already ready, just verify the input shape
        if self.is_ready():
            if (
                isinstance(self.bias, np.ndarray)
                and self.bias.shape != (1,)
                and self.bias.shape != ()
            ):
                # check if input shape is a single tuple (no PyTree)
                if (
                    isinstance(input_shape, tuple)
                    and all(isinstance(i, int) for i in input_shape)
                    and self.bias.shape != input_shape
                ):
                    raise ValueError(
                        f"Bias shape {self.bias.shape} does not match input "
                        f"shape {input_shape}"
                    )
                # else if the input shape is a PyTree of tuples,
                # all shapes must match
                elif any(
                    [
                        shape != self.bias.shape
                        for shape in jax.tree.leaves(
                            input_shape, is_leaf=is_shape_leaf
                        )
                    ]
                ):
                    input_shapes_list = jax.tree.leaves(
                        input_shape, is_leaf=is_shape_leaf
                    )
                    raise ValueError(
                        f"Bias shape {self.bias.shape} does not match all "
                        f"input leaf shapes {input_shapes_list}"
                    )
            elif isinstance(self.bias, np.ndarray) and (
                self.bias.shape == (1,) or self.bias.shape == ()
            ):
                pass
            else:
                # pytree case
                def check_shape(x: np.ndarray, s: Tuple[int, ...]) -> None:
                    if x.shape != s:
                        raise ValueError(
                            f"Bias leaf shape {x.shape} does not match input "
                            f"leaf shape {s}"
                        )

                jax.tree.map(
                    check_shape,
                    self.bias,
                    input_shape,
                    is_leaf=is_shape_leaf,
                )

        if self.bias is None:
            shape = (1,) if self.scalar else input_shape
            # otherwise, initialize the bias
            # bare array case
            if isinstance(shape, tuple) and all(
                isinstance(i, int) for i in shape
            ):
                if self.real:
                    self.bias = self.init_magnitude * jax.random.normal(
                        rng, shape
                    )
                else:
                    rkey, ikey = jax.random.split(rng)
                    self.bias = self.init_magnitude * (
                        jax.random.normal(rkey, shape)
                        + 1j * jax.random.normal(ikey, shape)
                    )
            else:

                def init_bias(s: Tuple[int, ...], k: Any) -> np.ndarray:
                    if self.real:
                        return self.init_magnitude * jax.random.normal(k, s)
                    else:
                        rkey, ikey = jax.random.split(k)
                        return self.init_magnitude * (
                            jax.random.normal(rkey, s)
                            + 1j * jax.random.normal(ikey, s)
                        )

                keys = jax.random.split(
                    rng, len(jax.tree.leaves(shape, is_leaf=is_shape_leaf))
                )
                keys = jax.tree.unflatten(
                    jax.tree.structure(shape, is_leaf=is_shape_leaf), keys
                )
                self.bias = jax.tree.map(
                    init_bias,
                    shape,
                    keys,
                    is_leaf=is_shape_leaf,
                )

    def get_output_shape(self, input_shape: DataShape) -> DataShape:
        return input_shape

    def get_hyperparameters(self) -> HyperParams:
        return {
            "init_magnitude": self.init_magnitude,
            "real": self.real,
            "scalar": self.scalar,
            "trainable": self.trainable,
        }

    def set_hyperparameters(self, hyperparams: HyperParams) -> None:
        super(Bias, self).set_hyperparameters(hyperparams)

    def get_params(self) -> Params:
        return self.bias

    def set_params(self, params: Params) -> None:
        if self.bias is not None:
            # ensure structure matches
            param_struct = jax.tree.structure(params)
            bias_struct = jax.tree.structure(self.bias)
            if param_struct != bias_struct:
                raise ValueError(
                    "Structure of params does not match existing parameters. "
                    f"Expected structure {bias_struct}, got {param_struct}"
                )

        if not jax.tree.all(
            jax.tree.map(lambda p: isinstance(p, np.ndarray), params)
        ):
            raise ValueError("All leaves must be numpy arrays")

        if self.scalar and not (
            isinstance(params, np.ndarray) and params.shape == (1,)
        ):
            raise ValueError(
                "For scalar bias, params must be a single numpy array with "
                f"shape (1,), got {params}"
            )

        # check real/complex consistency
        if self.real and not jax.tree.all(jax.tree.map(np.isrealobj, params)):
            raise ValueError("Params must be real-valued for a real module")
        if not self.real and jax.tree.all(jax.tree.map(np.isrealobj, params)):
            raise ValueError(
                "Params must have at least one complex-valued leaf for a "
                "complex module"
            )

        self.bias = params
