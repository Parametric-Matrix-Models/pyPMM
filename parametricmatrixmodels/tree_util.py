from __future__ import annotations

import jax
from beartype import beartype
from jaxtyping import Any, Array, PyTree, Shaped, jaxtyped


@jaxtyped(typechecker=beartype)
def batch_leaves(
    pytree: PyTree[Shaped[Array, "..."], " T"],
    batch_size: int,
    batch_idx: int,
    length: int | None = None,
    axis: int = 0,
) -> PyTree[Shaped[Array, "..."], " T"]:
    r"""
    Extracts a batch of values from all leaves of a PyTree of arrays.

    Each leaf is sliced along the specified axis by
    ``[batch_idx * batch_size : batch_idx * batch_size + length]`` if
    ``length`` is provided, otherwise by
    ``[batch_idx * batch_size : (batch_idx + 1) * batch_size]``.

    Parameters
    ----------
    pytree
        A PyTree where each leaf is an array with a matching size along the
        specified axis.
    batch_size
        The size of the batches that the leaves are divided into.
    batch_idx
        The index of the batch to extract.
    length
        Optional length of the slice to extract. If not provided,
        defaults to `batch_size`. Useful for getting the last batch which
        may be smaller than `batch_size`.
    axis
        The axis along which to slice the leaves. Default is 0.

    Returns
    -------
    A PyTree with the same structure as the input, but with each leaf
    sliced to contain only the specified batch.
    """

    start = batch_idx * batch_size
    length = length if length is not None else batch_size
    return jax.tree_map(
        lambda x: jax.lax.dynamic_slice_in_dim(x, start, length, axis=axis),
        pytree,
    )


@jaxtyped(typechecker=beartype)
def random_permute_leaves(
    pytree: PyTree[Shaped[Array, "..."], " T"],
    key: Any,
    axis: int = 0,
    independent_arrays: bool = False,
    independent_leaves: bool = False,
) -> PyTree[Shaped[Array, "..."], " T"]:
    r"""
    Randomly permutes the arrays in the leaves of a PyTree of arrays along a
    specified axis.

    Parameters
    ----------
    pytree
        A PyTree where each leaf is an array with a matching size along the
        specified axis.
    key
        A JAX PRNG key used for generating random permutations.
    axis
        The axis along which to permute the leaves. Default is 0.
    independent_arrays
        If True, each individual vector along the given axis is shuffled
        independently. Default is False. See the ``independent`` argument of
        ``jax.random.permutation`` for more details.
    independent_leaves
        If True, each leaf in the PyTree is permuted independently using
        different random keys. Default is False.

    Returns
    -------
    A PyTree with the same structure as the input, but with each leaf
    randomly permuted along the specified axis.
    """

    if independent_leaves:
        keys = jax.random.split(key, len(jax.tree.leaves(pytree)))
        keys = jax.tree.unflatten(
            jax.tree.structure(pytree),
            keys,
        )
        return jax.tree.map(
            lambda x, k: jax.random.permutation(
                k,
                x,
                axis=axis,
                independent=independent_arrays,
            ),
            pytree,
            keys,
        )
    else:
        return jax.tree.map(
            lambda x: jax.random.permutation(
                key,
                x,
                axis=axis,
                independent=independent_arrays,
            ),
            pytree,
        )
