from __future__ import annotations

from functools import wraps
from typing import TypeAlias

import jax
import jax.numpy as np
from beartype import beartype
from jaxtyping import Array, Inexact, Num, PyTree, PyTreeDef, jaxtyped

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

ModelStruct: str = " modelstruct"

#: ModelParamsStruct is a type tag for PyTrees of model parameters. It really
#: should be the composition of ModelStruct and "params" or " ..." but this
#: doesn't work currently with jaxtyping due to unbound types in returns.
#: See: https://github.com/patrick-kidger/jaxtyping/issues/357
ModelParamsStruct: str = " modelparamsstruct"
#: ModelStateStruct is a type tag for PyTrees of model states. It really
#: should be the composition of ModelStruct and "state" or " ..." but this
#: doesn't work currently with jaxtyping due to unbound types in returns.
#: See: https://github.com/patrick-kidger/jaxtyping/issues/357
ModelStateStruct: str = " modelstatestruct"

#: Type alias for a PyTree of modules in a model.
ModelModules: TypeAlias = PyTree[BaseModule, ModelStruct]

#: Type alias for a PyTree of parameters in a model.
ModelParams: TypeAlias = PyTree[Inexact[Array, "..."], ModelParamsStruct]

#: Type alias for a PyTree of states in a model.
ModelState: TypeAlias = PyTree[
    Num[Array, "..."] | Tuple | List, ModelStateStruct
]

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
    ) -> Tuple[Data, ModelState]:

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
