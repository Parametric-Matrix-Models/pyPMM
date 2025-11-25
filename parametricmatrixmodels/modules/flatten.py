import jax

from ..tree_util import is_shape_leaf
from ..typing import (
    Any,
    DataShape,
)
from .reshape import Reshape


class Flatten(Reshape):
    """
    Module that flattens the input to 1D. Ignores the batch dimension. Operates
    on all leafs of a PyTree input.
    """

    def __init__(self) -> None:
        # initialize with shape=None, shape will be determined at compile time
        # this accounts for both array and PyTree inputs
        super().__init__(shape=None)

    def name(self) -> str:
        return "Flatten"

    def compile(self, key: Any, input_shape: DataShape) -> None:
        try:
            len(input_shape)
        except TypeError:
            raise TypeError(
                "Input shape must be a tuple, list, or PyTree of shapes."
            )

        # if input_shape is an iterable of ints, then the input is a single
        # array
        if all(isinstance(dim, int) for dim in input_shape):
            self.shape = (-1,)
            return

        # construct the tree of output shapes (all (-1,))
        input_shapes, input_struct = jax.tree.flatten(
            input_shape, is_leaf=is_shape_leaf
        )
        self.shape = jax.tree.unflatten(
            input_struct,
            [(-1,) for _ in input_shapes],
        )
