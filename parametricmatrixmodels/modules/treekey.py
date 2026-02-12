from __future__ import annotations

import jax
from beartype import beartype
from jaxtyping import PyTree, jaxtyped

from ..tree_util import getitem_by_strpath, is_shape_leaf
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


class TreeKey(BaseModule):
    """
    Module that takes an input tree and takes subtrees or leaves based on
    specified keypaths.
    """

    __version__: str = "0.0.0"

    def __init__(
        self, keypaths: PyTree[str] | None = None, separator: str = "."
    ) -> None:
        r"""
        Initializes the TreeKey module.

        Parameters
        ----------
        keypaths
            A PyTree of strings representing the keypaths to extract from the
            input tree. The structure of `keypaths` determines the structure of
            the output tree. If `None`, the entire input tree is returned
            unchanged.
        separator
            A string used to separate keys in the keypaths. Default is ".".

        Examples
        --------

        For an input PyTree like ``[{"x": ..., "y": ...}, ...]``, the
        ``TreeKey`` module that extracts the subtree or leaf at keypath
        ``"0.x"`` as well as the second element (keypath ``"1"``) and places
        these into a new PyTree with keys ``"a"`` and ``"b"`` respectively can
        be created as follows:

        >>> TreeKey(keypaths={"a": "0.x", "b": "1"})

        The same, but instead the output structure is a Tuple:

        >>> TreeKey(keypaths=("0.x", "1"))

        """
        self.keypaths = keypaths
        self.separator = separator

    @property
    def name(self) -> str:
        if self.keypaths is None:
            return "TreeKey"
        return f"TreeKey({self.keypaths})"

    def is_ready(self) -> bool:
        return True

    def compile(self, rng: Any, input_shape: DataShape) -> None:
        # just validate that the keypaths are valid by attempting to get them
        self.get_output_shape(input_shape)

    def get_output_shape(self, input_shape: DataShape) -> DataShape:
        if self.keypaths is None:
            return input_shape
        else:
            try:
                return jax.tree.map(
                    lambda kp: getitem_by_strpath(
                        input_shape,
                        kp,
                        separator=self.separator,
                        allow_early_return=True,
                        return_remainder=False,
                        is_leaf=is_shape_leaf,
                    ),
                    self.keypaths,
                )
            except (KeyError, IndexError, ValueError) as e:
                raise ValueError(
                    f"Invalid keypaths {self.keypaths} for input shape "
                    f"{input_shape}"
                ) from e

    def _get_callable(self) -> ModuleCallable:

        if self.keypaths is None:

            @jaxtyped(typechecker=beartype)
            def treekey_callable(
                params: Params,
                data: Data,
                training: bool,
                state: State,
                rng: Any,
            ) -> Tuple[Data, State]:
                return data, state

        else:

            @jaxtyped(typechecker=beartype)
            def treekey_callable(
                params: Params,
                data: Data,
                training: bool,
                state: State,
                rng: Any,
            ) -> Tuple[Data, State]:
                out = jax.tree.map(
                    lambda kp: getitem_by_strpath(
                        data,
                        kp,
                        separator=self.separator,
                        allow_early_return=True,
                        return_remainder=False,
                    ),
                    self.keypaths,
                )
                return out, state

        return treekey_callable

    def get_hyperparameters(self) -> HyperParams:
        return {
            "keypaths": self.keypaths,
            "separator": self.separator,
        }

    def get_params(self) -> Params:
        return ()

    def set_params(self, params: Params) -> None:
        pass
