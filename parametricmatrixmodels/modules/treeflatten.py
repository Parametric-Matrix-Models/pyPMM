import jax
from beartype import beartype
from jaxtyping import jaxtyped

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


class TreeFlatten(BaseModule):
    """
    Module that flattens an input tree of arrays into a list of arrays
    """

    __version__: str = "0.0.0"

    def __init__(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "TreeFlatten"

    def compile(self, rng: Any, input_shape: DataShape) -> None:
        pass

    def is_ready(self) -> bool:
        return True

    def _get_callable(self) -> ModuleCallable:

        @jaxtyped(typechecker=beartype)
        def flatten_tree_callable(
            params: Params, data: Data, training: bool, state: State, rng: Any
        ) -> Tuple[Data, State]:
            return jax.tree.leaves(data), state

        return flatten_tree_callable

    def get_output_shape(self, input_shape: DataShape) -> DataShape:
        return jax.tree.leaves(input_shape, is_leaf=is_shape_leaf)

    def get_hyperparameters(self) -> HyperParams:
        return {}

    def get_params(self) -> Params:
        return ()

    def set_params(self, params: Params) -> None:
        pass
