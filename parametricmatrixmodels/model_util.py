from __future__ import annotations

import warnings
from functools import wraps
from typing import TypeAlias

import jax
import jax.numpy as np
from beartype import beartype
from jaxtyping import Array, Num, PyTree, PyTreeDef, jaxtyped

from .modules import BaseModule
from .typing import (
    Any,
    ArrayData,
    Callable,
    Data,
    List,
    ModuleCallable,
    Params,
    State,
    Tuple,
)

# type aliases for Models

#: Type alias for a PyTree of modules in a model.
ModelModules: TypeAlias = PyTree[BaseModule, " moduletree"]

#: Type alias for a PyTree of parameters in a model.
ModelParams: TypeAlias = PyTree[Num[Array, "..."]]

#: Type alias for a PyTree of states in a model.
ModelState: TypeAlias = PyTree[Num[Array, "..."]]

#: Type alias for the callable signature of a model's forward method. Similar
#: (actually identical) to ModuleCallable, but suggestively uses ModelParams
#: and ModelState, which are nested PyTrees.
ModelCallable: TypeAlias = Callable[
    [
        ModelParams,
        Data,
        bool,
        ModelState,
        Any,
    ],
    Tuple[Data, ModelState],
]


@jaxtyped(typechecker=beartype)
def safecast(X: Data, dtype: Any) -> Data:
    r"""
    Safely cast input data to a specified dtype, ensuring that complex types
    are not inadvertently cast to float types. And issues a warning if the
    requested dtype was not successfully applied, usually due to JAX settings.

    Parameters
    ----------
    X
        Input data to be cast.
    dtype
        Desired data type for the output.
    """

    # make sure that we don't cast complex to float
    def cast_with_complex_check(x: np.ndarray, dtype: Any) -> np.ndarray:
        if np.issubdtype(x.dtype, np.complexfloating) and not np.issubdtype(
            dtype, np.complexfloating
        ):
            raise ValueError(
                f"Cannot cast complex input dtype {x.dtype} to "
                f"float output dtype {dtype}."
            )
        return x.astype(dtype)

    X_ = jax.tree.map(lambda x: cast_with_complex_check(x, dtype), X)

    # make sure the dtype was converted, issue a warning if not
    def check_cast(x: np.ndarray, dtype: Any) -> None:
        if x.dtype != dtype:
            warnings.warn(
                "While performing inference with model: "
                f"Requested dtype ({dtype}) was not successfully applied. "
                "This is most likely due to JAX_ENABLE_X64 not being set. "
                "See accompanying JAX warning for more details.",
                UserWarning,
            )

    jax.tree.map(lambda x: check_cast(x, dtype), X_)

    return X_


@jaxtyped(typechecker=beartype)
def autobatch(
    fn: ModelCallable | ModuleCallable,
    max_batch_size: int | None,
) -> ModelCallable | ModuleCallable:
    r"""
    Decorator to automatically limit the batch size of a ``ModelCallable`` or
    ``ModuleCallable`` function. This is not the same as taking a function and
    vmap'ing it. The original function must already be able to handle batches
    of data. This decorator simply breaks up large batches into smaller batches
    of size ``max_batch_size``, calls the original function on each smaller
    batch, and then concatenates the results.

    This would usually be used on a function that has already been
    jit-compiled.

    The returned state is the state returned from the last batch processed.

    The rng parameter is passed through to each call of the original function
    unchanged.

    Parameters
    ----------
    fn
        The function to be decorated. Must be a ``ModelCallable`` or
        ``ModuleCallable``.
    max_batch_size
        The maximum batch size to use when calling the function. If ``None``,
        then no batching is performed and the original function is returned.
    """

    @wraps(fn)
    @jaxtyped(typechecker=beartype)
    def batched_fn(
        params: Params | ModelParams,
        X: Data,
        training: bool,
        state: State | ModelState,
        rng: Any,
    ) -> Tuple[Data, State] | Tuple[Data, ModelState]:

        orig_batch_size = jax.tree.leaves(X)[0].shape[0]

        if max_batch_size is None or orig_batch_size <= max_batch_size:
            # nothing to do
            return fn(params, X, training, state, rng)
        else:
            flattened_outputs: List[List[ArrayData]] = []
            output_structs: List[PyTreeDef] = []
            new_state = state

            def get_batch(
                arr: ArrayData,
                batch_idx: int,
                batch_size: int,
            ) -> ArrayData:
                return arr[
                    batch_idx * batch_size : (batch_idx + 1) * batch_size, ...
                ]

            num_batches = int(np.ceil(orig_batch_size / max_batch_size))

            for batch_idx in range(num_batches):
                batch_X = jax.tree.map(
                    lambda x: get_batch(x, batch_idx, max_batch_size), X
                )

                batch_output, new_state = fn(
                    params,
                    batch_X,
                    training,
                    new_state,
                    rng,
                )

                flat_output, output_struct = jax.tree.flatten(batch_output)
                flattened_outputs.append(flat_output)
                output_structs.append(output_struct)

            # check that the structures are all the same
            first_struct = output_structs[0]
            if not all(struct == first_struct for struct in output_structs):
                raise ValueError(
                    "Inconsistent output structures from autobatched function."
                )

            # concatenate the outputs along the batch dimension
            concatenated_outputs = [
                np.concatenate(
                    [
                        flattened_outputs[batch_idx][i]
                        for batch_idx in range(num_batches)
                    ],
                    axis=0,
                )
                for i in range(len(flattened_outputs[0]))
            ]

            final_output = jax.tree.unflatten(
                first_struct, concatenated_outputs
            )

            return final_output, new_state

    return batched_fn


def strfmt_pytree(
    tree: PyTree,
    indent: int = 0,
    indentation: int = 1,
    max_leaf_chars: int | None = None,
    base_indent_str: str = "",
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> str:
    """
    Format a JAX PyTree into a nicely indented string representation.

    Parameters
    ----------
        tree
            An arbitrary JAX PyTree (dict, list, tuple, or leaf value)
        indent
            Current indentation level (used for recursion)
        indentation
            Number of spaces to indent for each level
        max_leaf_chars
            Maximum characters for leaf value representation before truncation
        base_indent_str
            Base indentation string to prepend to each line
        is_leaf
            Optional function to determine if a node is a leaf

    Returns:
        A formatted string representation of the PyTree
    """
    indent_str = " " * indent * indentation
    next_indent_str = " " * (indent + 1) * indentation

    def truncate_leaf(s: str) -> str:
        """Truncate leaf representation if it exceeds max_leaf_chars."""
        if max_leaf_chars is None:
            return s
        if len(s) > max_leaf_chars:
            return s[: max_leaf_chars - 3] + "..."
        return s

    # handle custom leaf detection
    if is_leaf is not None and is_leaf(tree):
        ret_str = truncate_leaf(repr(tree))

    # handle dictionaries
    elif isinstance(tree, dict):
        if not tree:
            if indent == 0:
                return base_indent_str + "{}"
            else:
                return "{}"

        items = []
        for key, value in tree.items():
            formatted_value = strfmt_pytree(
                value,
                indent + 1,
                indentation,
                max_leaf_chars,
                base_indent_str,
                is_leaf,
            )
            items.append(
                f"{base_indent_str}{next_indent_str}{key}: {formatted_value}"
            )

        ret_str = (
            "{{\n" + ",\n".join(items) + f",\n{base_indent_str}{indent_str}}}"
        )

    # handle lists
    elif isinstance(tree, list):
        if not tree:
            if indent == 0:
                return base_indent_str + "[]"
            else:
                return "[]"

        items = []
        for item in tree:
            formatted_item = strfmt_pytree(
                item,
                indent + 1,
                indentation,
                max_leaf_chars,
                base_indent_str,
                is_leaf,
            )
            items.append(f"{base_indent_str}{next_indent_str}{formatted_item}")

        ret_str = (
            "[\n" + ",\n".join(items) + f",\n{base_indent_str}{indent_str}]"
        )

    # handle tuples
    elif isinstance(tree, tuple):
        if not tree:
            if indent == 0:
                return base_indent_str + "()"
            else:
                return "()"

        items = []
        for item in tree:
            formatted_item = strfmt_pytree(
                item,
                indent + 1,
                indentation,
                max_leaf_chars,
                base_indent_str,
                is_leaf,
            )
            items.append(f"{base_indent_str}{next_indent_str}{formatted_item}")

        ret_str = (
            "(\n" + ",\n".join(items) + f",\n{base_indent_str}{indent_str})"
        )

    # handle leaves
    else:
        ret_str = truncate_leaf(repr(tree))

    if indent == 0:
        return base_indent_str + ret_str
    else:
        return ret_str
