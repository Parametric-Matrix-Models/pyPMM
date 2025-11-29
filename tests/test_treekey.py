import jax
import jax.numpy as np

import parametricmatrixmodels as pmm


def test_treekey():
    r"""
    Test the TreeKey module
    """

    batch_dim = 10

    key = jax.random.key(0)

    arr = jax.random.normal(key, (batch_dim, 3, 4))

    data = ({"a": arr, "b": {"c": arr, "d": [arr, arr]}},)
    tk = pmm.modules.TreeKey({"x": "0.a", "y": "0.b.d.1"})
    tk.compile(key, jax.tree.map(lambda x: x.shape[1:], data))
    out, _ = tk(data)

    expected_out = {"x": arr, "y": arr}

    expected_struct = jax.tree.structure(expected_out)
    out_struct = jax.tree.structure(out)

    assert expected_struct == out_struct
    expected_leaves = jax.tree.leaves(expected_out)
    out_leaves = jax.tree.leaves(out)
    for expected_leaf, out_leaf in zip(expected_leaves, out_leaves):
        assert np.allclose(expected_leaf, out_leaf)

    # extract a single key into a bare array
    tk = pmm.modules.TreeKey("0.b.c")
    tk.compile(key, jax.tree.map(lambda x: x.shape[1:], data))
    out, _ = tk(data)
    expected_out = arr
    assert np.allclose(expected_out, out)
