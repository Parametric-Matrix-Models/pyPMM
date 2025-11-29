from __future__ import annotations

import jax

from ..sequentialmodel import SequentialModel

# direct import to avoid circular import
from ..tree_util import is_shape_leaf
from ..typing import (
    Any,
    DataShape,
    HyperParams,
    List,
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

    This module accepts only bare arrays or PyTrees with only a single leaf
    array.
    """

    def __init__(
        self,
        out_features: int | None = None,
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
            The number of output features. If None, this must be set later
            using ``set_hyperparameters``.
        """

        self.modules = None
        self.out_features = out_features
        self.bias = bias
        self.activation = activation
        self.init_magnitude = init_magnitude
        self.real = real
        super().__init__()

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

        if len(jax.tree.leaves(input_shape, is_leaf=is_shape_leaf)) != 1:
            raise ValueError(
                "LinearNN only supports input shapes with a single leaf array."
            )

        if input_shape != self.input_shape or self.modules is None:
            # don't overwrite modules if input shape hasn't changed
            modules: List[BaseModule] = [
                Flatten(),
                MatMul(
                    output_shape=self.out_features,
                    trainable=True,
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

    def get_output_shape(self, input_shape: DataShape) -> DataShape:
        r"""
        Get the output shape of the LinearNN module given the input shape.

        Parameters
        ----------
        input_shape
            Shape of the input array, excluding the batch size.

        Returns
        -------
        DataShape
            Shape of the output array, excluding the batch size.
        """

        if self.out_features is None:
            raise ValueError(
                "out_features must be specified before getting output shape."
            )

        return (self.out_features,)

    def get_hyperparameters(self) -> HyperParams:
        r"""
        Get the hyperparameters of the LinearNN module.

        Returns
        -------
        HyperParams
            A dictionary containing the hyperparameters of the module.
        """

        return {
            "out_features": self.out_features,
            "bias": self.bias,
            "activation": self.activation,
            "init_magnitude": self.init_magnitude,
            "real": self.real,
            **super().get_hyperparameters(),
        }

    def set_hyperparameters(self, hyperparams: HyperParams) -> None:
        r"""
        Set the hyperparameters of the LinearNN module.

        Parameters
        ----------
        hyperparams
            A dictionary containing the hyperparameters to set.
        """

        self.out_features = hyperparams.get("out_features", self.out_features)
        self.bias = hyperparams.get("bias", self.bias)
        self.activation = hyperparams.get("activation", self.activation)
        self.init_magnitude = hyperparams.get(
            "init_magnitude", self.init_magnitude
        )
        self.real = hyperparams.get("real", self.real)
        super().set_hyperparameters(hyperparams)
