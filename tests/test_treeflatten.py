import jax
import jax.numpy as np

import parametricmatrixmodels as pmm


def test_treeflatten():
    r"""
    Test the TreeFlatten module
    """

    batch_dim = 10

    key = jax.random.key(0)

    arr = jax.random.normal(key, (batch_dim, 3, 4))

    data = ({"a": arr, "b": {"c": arr, "d": [arr, arr]}},)
    tree_flatten = pmm.modules.TreeFlatten()
    tree_flatten.compile(key, jax.tree.map(lambda x: x.shape[1:], data))
    out, _ = tree_flatten(data)

    expected_out = jax.tree.leaves(data)

    expected_struct = jax.tree.structure(expected_out)
    out_struct = jax.tree.structure(out)

    assert expected_struct == out_struct
    for expected, actual in zip(expected_out, out):
        assert np.allclose(expected, actual)
