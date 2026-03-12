from __future__ import annotations

from functools import wraps
from typing import TypeAlias

import jax
import jax.numpy as np
from beartype import beartype
from jaxtyping import (
    Array,
    Inexact,
    Num,
    PyTree,
    jaxtyped,
)

from . import tree_util
from .modules import BaseModule
from .progressbar import ProgressBar
from .typing import (
    Any,
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
    avoid_recompilation: bool = False,
    verbose: bool = False,
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

    if avoid_recompilation:
        jfn = jax.jit(fn, static_argnames=["training"])
    else:
        jfn = fn

    # rewrite to be jittable
    @wraps(jfn)
    @jaxtyped(typechecker=beartype)
    def batched_fn(
        params: Params | ModelParams,
        X: Data,
        training: bool,
        state: State | ModelState,
        rng: Any,
    ) -> Tuple[Data, ModelState]:

        orig_batch_size = jax.tree.leaves(X)[0].shape[0]

        # max_batch_size is a static argument and so is the size of the batch,
        # so this condition will be traced out
        if max_batch_size is None or orig_batch_size <= max_batch_size:
            # nothing to do
            return jfn(params, X, training, state, rng)
        else:
            num_batches = orig_batch_size // max_batch_size
            remainder = orig_batch_size % max_batch_size

            if verbose:
                pb = ProgressBar(
                    num_batches + 1 + (1 if remainder > 0 else 0), length=20
                )

            def update_pb(out: Data, i: int) -> Data:
                pb.update(i)
                return out

            def end_pb(out: Data) -> Data:
                pb.end()
                return out

            def body_fn(i_new_state, X_batch):
                i, new_state = i_new_state
                out, new_state = jfn(params, X_batch, training, new_state, rng)
                if verbose:
                    if avoid_recompilation:
                        pb.update(i + 1)
                    else:
                        out = jax.pure_callback(update_pb, out, out, i + 1)
                return (i + 1, new_state), out

            X_batches, X_remainder = tree_util.pytree_split(
                X, max_batch_size, axis=0
            )

            if avoid_recompilation:
                i_new_state = (0, state)
                out = []
                for j in range(num_batches):
                    X_batch = jax.tree.map(lambda x: x[j], X_batches)
                    i_new_state, out_ = body_fn(i_new_state, X_batch)
                    out.append(out_)

                out = (
                    tree_util.concatenate(out, axis=0)
                    if len(out) > 0
                    else None
                )

            else:
                i_new_state, out = jax.lax.scan(
                    body_fn,
                    (0, state),
                    X_batches,
                )

                # each leaf of out is of shape [num_batches, batch_size, ...],
                # so we need to reshape to [num_batches * batch_size, ...]
                out = jax.tree.map(
                    lambda o_leaf: np.reshape(
                        o_leaf, (-1,) + o_leaf.shape[2:]
                    ),
                    out,
                )

            _, new_state = i_new_state

            if remainder > 0:
                # process the remainder batch first, so that the final state is
                # from the last batch processed
                if verbose:
                    if avoid_recompilation:
                        pb.update(num_batches + 2)
                    else:
                        out = jax.pure_callback(
                            update_pb, out, out, num_batches + 2
                        )

                if avoid_recompilation:
                    out_remainders = []
                    for j in range(remainder):
                        X_rem = jax.tree.map(
                            lambda x: x[j][None, :], X_remainder
                        )
                        out_remainder, new_state = jfn(
                            params,
                            X_rem,
                            training,
                            new_state,
                            rng,
                        )
                        out_remainders.append(out_remainder)
                    out_remainder = (
                        tree_util.concatenate(out_remainders, axis=0)
                        if len(out_remainders) > 0
                        else None
                    )

                    out = (
                        tree_util.concatenate([out, out_remainder], axis=0)
                        if out is not None
                        else out_remainder
                    )

                else:
                    out_remainder, new_state = jfn(
                        params,
                        X_remainder,
                        training,
                        new_state,
                        rng,
                    )
                    out = tree_util.concatenate([out, out_remainder], axis=0)

            if verbose:
                if avoid_recompilation:
                    pb.end()

                else:
                    out = jax.pure_callback(end_pb, out, out)

            return out, new_state

    if not avoid_recompilation:
        batched_fn = jax.jit(batched_fn, static_argnames=["training"])

    return batched_fn
