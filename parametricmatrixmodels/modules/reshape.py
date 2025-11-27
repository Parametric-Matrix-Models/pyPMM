import jax
import jax.numpy as np
from beartype import beartype
from jaxtyping import jaxtyped

from ..tree_util import is_shape_leaf
from ..typing import (
    Any,
    ArrayData,
    ArrayDataShape,
    Data,
    DataShape,
    HyperParams,
    ModuleCallable,
    Params,
    State,
    Tuple,
)
from .basemodule import BaseModule


class Reshape(BaseModule):
    """
    Module that reshapes the input array to a specified shape. Ignores the
    batch dimension.
    """

    def __init__(self, shape: DataShape = None) -> None:
        """
        Initialize a ``Reshape`` module.

        Parameters
        ----------
        shape
            The target shape to reshape the input to, by default None.
            If None, the input shape will remain unchanged.
            Does not include the batch dimension. If the input to the module is
            a PyTree, then ``shape`` should be a PyTree of matching structure.
            Any ``None`` values in the PyTree will leave the corresponding leaf
            arrays unchanged.

        Examples
        --------

        .. code-block:: python

            # Prepare to accept only bare array data (no PyTrees) and reshape
            # to (2, 3)
            reshape_module = Reshape(shape=(2, 3))

            # Prepare to accept a PyTree of arrays with structure [*, (*, *)]
            # and reshape the first leaf to (2, 3), leave the second leaf
            # unchanged, and flatten the final leaf
            reshape_module = Reshape(shape=[(2, 3), (None, (-1,))])
        """

        # validate shape
        # unless it is entirely an iterable of ints, none of the elements can
        # be ints
        if shape is None:
            self.shape = shape
            return
        try:
            len(shape)
        except TypeError:
            raise AssertionError(
                "Shape must be a tuple, list, or PyTree of shapes."
            )
        if all(isinstance(dim, int) for dim in shape):
            # shape is just an iterable of ints
            pass
        elif any(isinstance(dim, int) for dim in shape):
            # shape is a PyTree, but not all the of the leaves are shapes
            # (iterables themselves)
            # e.g. shape = [(2, 3), 2, (4, 5)], the second element (2) is
            # invalid and should be (2,) instead
            raise TypeError(
                "If shape is a PyTree, all leaves must be shapes "
                "(iterables of ints)."
            )

        # at this point shape is either an iterable of ints, or a PyTree of
        # shapes or Nones

        self.shape = shape

    @property
    def name(self) -> str:
        return f"Reshape(shape={self.shape})"

    def is_ready(self) -> bool:
        return True

    def get_num_trainable_floats(self) -> int | None:
        return 0

    def _get_callable(
        self,
    ) -> ModuleCallable:

        @jaxtyped(typechecker=beartype)
        def reshape_array(
            arr: ArrayData,
            shape: ArrayDataShape | None,
        ) -> ArrayData:
            batch_dim = arr.shape[0]
            if shape is None:
                return arr
            else:
                return np.reshape(arr, (batch_dim, *shape))

        @jaxtyped(typechecker=beartype)
        def reshape_callable(
            params: Params,
            data: Data,
            training: bool,
            state: State,
            rng: Any,
        ) -> Tuple[Data, State]:
            if self.shape is None:
                return data, state
            else:
                reshaped_data = jax.tree.map(
                    reshape_array,
                    data,
                    self.shape,
                )
                return reshaped_data, state

        return reshape_callable

    def validate_and_concretify_shape(
        self, input_shape: DataShape
    ) -> DataShape:
        # check that input_shape and self.shape are compatible
        if self.shape is None:
            return input_shape

        assert input_shape is not None, "Input shape must not be None."
        assert not (None in input_shape), "Input shape must not contain None."
        try:
            len(input_shape)
        except TypeError:
            raise TypeError(
                "Input shape must be a tuple, list, or PyTree of shapes."
            )

        promoted = False

        # if input_shape is an iterable of ints, convert to a single-element
        # PyTree for consistency
        if all(isinstance(dim, int) for dim in input_shape):
            input_shape = (input_shape,)
            promoted = True

        # same for self.shape
        if all(isinstance(dim, int) for dim in self.shape):
            selfshape = (self.shape,)
        else:
            selfshape = self.shape

        input_struct = jax.tree.structure(input_shape, is_leaf=is_shape_leaf)
        shape_struct = jax.tree.structure(selfshape, is_leaf=is_shape_leaf)

        assert input_struct == shape_struct, (
            f"Input shape structure {input_struct} does not match target shape"
            f" structure {shape_struct}"
        )

        def check_compatibility_and_concretify(
            in_shape: ArrayDataShape,
            target_shape: ArrayDataShape | None,
        ) -> ArrayDataShape:
            if target_shape is None:
                return

            in_size = np.prod(np.array(in_shape)).item()
            if -1 in target_shape:
                # make sure there is only one -1
                assert (
                    target_shape.count(-1) == 1
                ), "Target shape can only contain one -1 dimension"
                # infer the size of the -1 dimension
                known_size = 1
                for dim in target_shape:
                    if dim != -1:
                        known_size *= dim
                inferred_dim = in_size // known_size
                target_size = known_size * inferred_dim
            else:
                target_size = np.prod(np.array(target_shape)).item()

            assert in_size == target_size, (
                f"Input shape {in_shape} is not compatible with target shape"
                f" {target_shape}"
            )

            if -1 in target_shape:
                # replace -1 with inferred dimension
                return tuple(
                    inferred_dim if dim == -1 else dim for dim in target_shape
                )
            else:
                return target_shape

        concrete_shape = jax.tree.map(
            check_compatibility_and_concretify,
            input_shape,
            selfshape,
            is_leaf=is_shape_leaf,
        )
        if promoted:
            return concrete_shape[0]
        else:
            return concrete_shape

    def compile(self, rng: Any, input_shape: DataShape) -> None:
        self.validate_and_concretify_shape(input_shape)

    def get_output_shape(self, input_shape: DataShape) -> DataShape:
        return self.validate_and_concretify_shape(input_shape)

    def get_hyperparameters(self) -> HyperParams:
        return {
            "shape": self.shape,
        }

    def get_params(self) -> Params:
        return ()

    def set_params(self, params: Params) -> None:
        pass
