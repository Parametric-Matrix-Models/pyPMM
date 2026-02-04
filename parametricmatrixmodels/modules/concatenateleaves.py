import jax
import jax.numpy as np
from beartype import beartype
from jaxtyping import jaxtyped

from ..tree_util import get_shapes, getitem_by_strpath, is_shape_leaf
from ..typing import (
    Any,
    Data,
    DataShape,
    HyperParams,
    List,
    ModuleCallable,
    Params,
    State,
    Tuple,
)
from .basemodule import BaseModule


class ConcatenateLeaves(BaseModule):
    """
    Module that concatenates all the array leaves of a PyTree of arrays into a
    single array.
    """

    __version__: str = "0.0.0"

    def __init__(
        self,
        axis: int | None = None,
        path_order: Tuple[str, ...] | List[str] | None = None,
        separator: str = ".",
    ) -> None:
        r"""
        Initialize the ConcatenateLeaves module.

        Parameters
        ----------
        axis : int | None, optional
            The axis along which to concatenate the leaves. If None, the leaves
            are flattened before concatenation. Default is None.

        path_order : Tuple[str, ...] | List[str] | None, optional
            Ordered list or tuple of string paths specifying the order in which
            to traverse the leaves of the input PyTree for concatenation. If
            None, the default traversal order is used. Default is None.

        separator : str, optional
            The separator used in the string paths for path_order. Default is
            ".".

        See Also
        --------
        jax.tree_util.keystr
            Function to generate string paths for PyTree leaves.
        """
        self.axis = axis
        self.path_order = path_order
        self.separator = separator

    @property
    def name(self) -> str:
        axis_str = f"axis={self.axis}" if self.axis is not None else ""
        path_order_str = (
            f"path_order={self.path_order}"
            if self.path_order is not None
            else ""
        )
        info_str = ", ".join(s for s in [axis_str, path_order_str] if s)
        return f"ConcatenateLeaves({info_str})"

    def compile(self, rng: Any, input_shape: DataShape) -> None:
        # make sure input_shape is valid
        self.get_output_shape(input_shape)

    def is_ready(self) -> bool:
        return True

    def _get_callable(self) -> ModuleCallable:

        @jaxtyped(typechecker=beartype)
        def concatenate_leaves_callable(
            params: Params, data: Data, training: bool, state: State, rng: Any
        ) -> Tuple[Data, State]:
            if self.path_order is not None:
                leaves = jax.tree.map(
                    lambda p: getitem_by_strpath(
                        data,
                        p,
                        separator=self.separator,
                        allow_early_return=False,
                        return_remainder=False,
                    ),
                    self.path_order,
                )
            else:
                leaves = jax.tree.leaves(data)

            # need to modify axis to preserve batch dimension
            if self.axis is None:
                # flatten all trailing dimensions except batch, then
                # concatenate along last axis
                return (
                    np.concatenate(
                        [
                            np.reshape(leaf, (leaf.shape[0], -1))
                            for leaf in leaves
                        ],
                        axis=-1,
                    ),
                    state,
                )
            else:
                return np.concatenate(leaves, axis=self.axis + 1), state

        return concatenate_leaves_callable

    def get_output_shape(self, input_shape: DataShape) -> DataShape:
        # easiest way is to create dummy data with the input shape (with added
        # batch dimension) and run concatenate on it
        dummy_data = jax.tree.map(
            lambda shape: np.zeros((1,) + shape, dtype=np.float32),
            input_shape,
            is_leaf=is_shape_leaf,
        )
        if self.path_order is not None:
            try:
                leaves = jax.tree.map(
                    lambda p: getitem_by_strpath(
                        dummy_data,
                        p,
                        separator=self.separator,
                        allow_early_return=False,
                        return_remainder=False,
                    ),
                    self.path_order,
                )
            except Exception as e:
                raise ValueError(
                    "Could not retrieve leaves using the specified "
                    f"path_order: {self.path_order}"
                ) from e
        else:
            leaves = jax.tree.leaves(dummy_data)

        if not all(isinstance(leaf, np.ndarray) for leaf in leaves):
            if self.path_order is not None:
                raise ValueError(
                    "All leaves must be arrays to concatenate, but found "
                    f"leaves with types: {[type(leaf) for leaf in leaves]}. "
                    "Make sure all paths in path_order are valid and lead to "
                    "array leaves."
                )
            else:
                raise ValueError(
                    "All leaves must be arrays to concatenate, but found "
                    f"leaves with types: {[type(leaf) for leaf in leaves]}"
                )

        try:
            if self.axis is None:
                concatenated = np.concatenate(
                    [np.reshape(leaf, (leaf.shape[0], -1)) for leaf in leaves],
                    axis=-1,
                )
            else:
                concatenated = np.concatenate(leaves, axis=self.axis + 1)
        except Exception as e:
            raise ValueError(
                "Could not concatenate leaves with shapes "
                f"{[leaf.shape for leaf in leaves]} along axis {self.axis}"
            ) from e
        return get_shapes(concatenated, slice(1, None))

    def get_hyperparameters(self) -> HyperParams:
        return {
            "axis": self.axis,
        }

    def get_params(self) -> Params:
        return ()

    def set_params(self, params: Params) -> None:
        pass
