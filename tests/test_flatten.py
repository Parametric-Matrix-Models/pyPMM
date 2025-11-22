import jax
import jax.numpy as np

import parametricmatrixmodels as pmm


def test_flatten_array():
    r"""
    Test Flatten module for ArrayData
    """

    batch_size = 2
    in_arr = np.arange(batch_size * 12).reshape((batch_size, 3, 4))

    flatten = pmm.modules.Flatten()
    flatten.compile(None, in_arr.shape[1:])  # Exclude batch dimension
    out_arr, _ = flatten(in_arr)
    assert out_arr.shape == (batch_size, 12)
    assert np.all(out_arr == in_arr.reshape((batch_size, 12)))


def test_flatten_pytree():
    r"""
    Test Flatten module for Data
    """

    batch_size = 2
    in_arr_1 = np.arange(batch_size * 12).reshape((batch_size, 3, 4))
    in_arr_2 = np.arange(batch_size * 8).reshape((batch_size, 2, 2, 2))
    in_data = [in_arr_1, in_arr_2]

    flatten = pmm.modules.Flatten()
    flatten.compile(
        None, [arr.shape[1:] for arr in in_data]
    )  # Exclude batch dimension
    out_data, _ = flatten(in_data)

    in_struct = jax.tree.structure(in_data)
    out_struct = jax.tree.structure(out_data)

    assert in_struct == out_struct
    assert out_data[0].shape == (batch_size, 12)
    assert out_data[1].shape == (batch_size, 8)
    assert np.all(out_data[0] == in_arr_1.reshape((batch_size, 12)))
    assert np.all(out_data[1] == in_arr_2.reshape((batch_size, 8)))
