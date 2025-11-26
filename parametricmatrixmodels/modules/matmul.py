from __future__ import annotations

import jax

from ..tree_util import is_shape_leaf
from ..typing import (
    Any,
    DataShape,
    HyperParams,
    Params,
    PyTree,
    Tuple,
)
from .einsum import Einsum


class MatMul(Einsum):
    r"""
    Module for trainable matrix multiplication. Just a special case of Einsum
    where the trainable parameters are all matrices with dimensions matching
    the data dimensions. The einsum string is constructed automatically based
    on the data shape. Matrix multiplication is performed over the last
    dimension of the data.
    """

    def __init__(
        self,
        output_dims: PyTree[int, "Params"] | int | None = None,
        init_magnitude: float = 1e-2,
        real: bool = True,
    ) -> None:
        r"""
        Initialize the MatMul module.

        Parameters
        ----------
        output_dims
            The output dimensions for each data dimension. Can be a PyTree
            matching the data PyTree, or a single integer to be used for all
            arrays in the data PyTree.
        init_magnitude
            The magnitude of the random initialization for the trainable
            parameters.
        real
            Whether to use real or complex parameters.
        """
        self.params = None
        self.output_dims = output_dims
        self.init_magnitude = init_magnitude
        self.real = real

    @property
    def name(self) -> str:
        return f"MatMul(output_dims={self.output_dims})"

    def compile(self, rng: Any, input_shape: DataShape) -> None:
        if self.output_dims is None:
            raise ValueError(
                "output_dims must be specified for MatMul module."
            )

        if isinstance(self.output_dims, int):
            output_dims = jax.tree.map(
                lambda _: self.output_dims,
                input_shape,
                is_leaf=is_shape_leaf,
            )
        else:
            output_dims = self.output_dims

        # can't check for shapes as tuples of ints with bare output_dims since
        # the entire thing would be a leaf
        # so a bit of a hack, but we turn each output dim into a tuple of a
        # single int
        output_dims = jax.tree.map(
            lambda d: (d,) if isinstance(d, int) else d,
            output_dims,
        )

        # assert that the structure of output_dims matches input_shape
        output_dims_struct = jax.tree.structure(
            output_dims, is_leaf=is_shape_leaf
        )
        input_shape_struct = jax.tree.structure(
            input_shape,
            is_leaf=is_shape_leaf,
        )
        if output_dims_struct != input_shape_struct:
            raise ValueError(
                "The structure of input_shape and output_dims must match. "
                f"Got input_shape structure {input_shape_struct} and "
                f"shapes structure {output_dims_struct}."
            )

        # get the shapes of the parameter matrices from the final dimensions of
        # the input shapes and the desired output dimensions
        self.shapes = jax.tree.map(
            lambda in_shape, out_dim: (in_shape[~0], out_dim[0]),
            input_shape,
            output_dims,
            is_leaf=is_shape_leaf,
        )

        # Construct the einsum string based on the input shape and output dims
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

        def make_str(
            shape: Tuple[int, ...],
            out_dim: Tuple[int],
        ) -> str:
            out_dim = out_dim[0]
            unchanged_dims = chars[: len(shape) - 1]
            mult_dim = chars[len(shape) - 1]
            out_dim = chars[len(shape)]
            return (
                f"{unchanged_dims}{mult_dim},{mult_dim}{out_dim}"
                f"->{unchanged_dims}{out_dim}"
            )

        self.einsum_str = jax.tree.map(
            make_str,
            input_shape,
            output_dims,
            is_leaf=is_shape_leaf,
        )

        super().compile(rng, input_shape)

    def get_hyperparameters(self) -> HyperParams:
        return {
            "output_dims": self.output_dims,
            "init_magnitude": self.init_magnitude,
            "real": self.real,
        }
