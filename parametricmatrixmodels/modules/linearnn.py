from __future__ import annotations

import random

import jax

from ..sequentialmodel import SequentialModel

# direct import to avoid circular import
from ..tree_util import is_shape_leaf
from ..typing import (
    Any,
    DataShape,
    List,
    PyTree,
)
from .basemodule import BaseModule
from .bias import Bias
from .flatten import Flatten
from .matmul import MatMul


class LinearNN(SequentialModel):
    r"""
    A Module (SequentialModel) representing a single linear neural network
    layer.

    This module first flattens the input data, then applies a linear
    transformation using a weight matrix and bias vector. Optionally, an
    element-wise activation function can be applied after the linear
    transformation.

    This module accepts both bare arrays and PyTrees with leaf arrays. In the
    case of PyTrees, all operations are applied to each leaf array
    independently.
    """

    def __init__(
        self,
        out_features: PyTree[int] | int | None = None,
        bias: bool = True,
        activation: BaseModule | None = None,
        init_magnitude: float = 1e-2,
        real: bool = True,
    ):
        r"""
        Initialize the LinearNN module.

        Parameters
        ----------
        out_features
            The number of output features. If compiled for PyTrees, this must
            be a PyTree with the same structure as the input data where each
            leaf is an integer specifying the number of output features for
            the corresponding input leaf. If None, this must be set later using
            ``set_hyperparameters``.
        """

        self.modules = None
        self.out_features = out_features
        self.bias = bias
        self.activation = activation
        self.init_magnitude = init_magnitude
        self.real = real

    def compile(
        self,
        rng: Any | int | None,
        input_shape: DataShape,
        verbose: bool = False,
    ) -> None:
        r"""
        Compile the LinearNN module by initializing its sub-modules, then
        calling the compile method of the parent SequentialModel class.

        Parameters
        ----------
            rng
                Random key for initializing the model parameters. JAX PRNGKey
                or integer seed.
            input_shape
                Shape of the input array, excluding the batch size.
                For example, (input_features,) for a 1D input or
                (input_height, input_width, input_channels) for a 3D input.
            verbose
                Print debug information during compilation. Default is False.
        """

        if self.out_features is None:
            raise ValueError(
                "out_features must be specified before compiling the module."
            )

        _out_features = self.out_features

        if isinstance(_out_features, int):
            _out_features = jax.tree.map(
                lambda _: _out_features,
                input_shape,
                is_leaf=is_shape_leaf,
            )

        # can't check for shapes as tuples of ints with bare output_dims since
        # the entire thing would be a leaf
        # so a bit of a hack, but we turn each output dim into a tuple of a
        # single int
        _out_features = jax.tree.map(
            lambda dim: (dim,) if isinstance(dim, int) else dim,
            _out_features,
        )

        # assert that the structure of output_dims matches input_shape
        output_features_structure = jax.tree.structure(
            _out_features, is_leaf=is_shape_leaf
        )
        input_shape_structure = jax.tree.structure(
            input_shape, is_leaf=is_shape_leaf
        )
        if output_features_structure != input_shape_structure:
            raise ValueError(
                "The structure of input_shape and out_features must match. "
                f"Got input_shape structure {input_shape_structure} and "
                f"out_features structure {output_features_structure}."
            )

        if rng is None:
            rng = random.randint(0, 2**32 - 1)
        if isinstance(rng, int):
            rng = jax.random.key(rng)

        modules: List[BaseModule] = [
            Flatten(),
            MatMul(
                output_dims=self.out_features,  # revert to original for MatMul
                init_magnitude=self.init_magnitude,
                real=self.real,
            ),
        ]

        if self.bias:
            modules.append(
                Bias(
                    init_magnitude=self.init_magnitude,
                    real=self.real,
                    scalar=False,
                    trainable=True,
                )
            )

        if self.activation is not None:
            modules.append(self.activation)

        self.modules = modules

        super().compile(rng, input_shape, verbose)
