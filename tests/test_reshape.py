import jax.numpy as np
import pytest

import parametricmatrixmodels as pmm


def test_reshape_init():
    r"""
    Test runtime type checking of Reshape module's __init__
    """
    # valid shape: tuple of ints
    reshape_module = pmm.modules.Reshape(shape=(4, 3))  # noqa: F841

    # valid shape: PyTree with leaves as tuple of ints
    reshape_module = pmm.modules.Reshape(
        shape=[(4, 3), [(2, 2), (2,)]]
    )  # noqa: F841

    # invalid shape: list instead of tuple
    with pytest.raises(TypeError):
        reshape_module = pmm.modules.Reshape(shape=[4, 3])  # noqa: F841

    # invalid shape: tuple with non-int element
    with pytest.raises(TypeError):
        reshape_module = pmm.modules.Reshape(shape=(4, 3.0))  # noqa: F841

    # invalid shape: raw int
    with pytest.raises(TypeError):
        reshape_module = pmm.modules.Reshape(shape=12)  # noqa: F841


def test_reshape_array():
    r"""
    Test Reshape module for ArrayData
    """

    batch_size = 2
    in_arr = np.arange(batch_size * 12).reshape((batch_size, 3, 4))

    new_shape = (4, 3)
    reshape_module = pmm.modules.Reshape(shape=new_shape)
    reshape_module.validate_shape(in_arr.shape[1:])  # Validate shape
    out_arr, _ = reshape_module(in_arr)
    assert out_arr.shape == (batch_size, *new_shape)

    new_shape = (-1, 2)
    reshape_module = pmm.modules.Reshape(shape=new_shape)
    reshape_module.validate_shape(in_arr.shape[1:])  # Validate shape
    out_arr, _ = reshape_module(in_arr)
    assert out_arr.shape == (batch_size, 6, 2)

    # test reshape with different number of dimensions
    new_shape = (2, 2, 3)
    reshape_module = pmm.modules.Reshape(shape=new_shape)
    reshape_module.validate_shape(in_arr.shape[1:])  # Validate shape
    out_arr, _ = reshape_module(in_arr)
    assert out_arr.shape == (batch_size, *new_shape)

    # Test invalid reshape
    new_shape = (5, 3)
    reshape_module = pmm.modules.Reshape(shape=new_shape)
    with pytest.raises(AssertionError):
        reshape_module.validate_shape(in_arr.shape[1:])
    with pytest.raises(TypeError):
        out_arr, _ = reshape_module(in_arr)

    # test identity reshape
    new_shape = None
    reshape_module = pmm.modules.Reshape(shape=new_shape)
    reshape_module.validate_shape(in_arr.shape[1:])  # Validate shape
    out_arr, _ = reshape_module(in_arr)
    assert out_arr.shape == in_arr.shape


def test_reshape_pytree():
    r"""
    Test Reshape module for Data
    """

    batch_size = 2
    in_arr_1 = np.arange(batch_size * 12).reshape((batch_size, 3, 4))
    in_arr_2 = np.arange(batch_size * 8).reshape((batch_size, 2, 2, 2))
    in_data = [in_arr_1, in_arr_2]

    new_shape = [(4, 3), (4, 2)]
    reshape_module = pmm.modules.Reshape(shape=new_shape)
    reshape_module.validate_shape(
        [arr.shape[1:] for arr in in_data]
    )  # Validate shape
    out_data, _ = reshape_module(in_data)
    assert out_data[0].shape == (batch_size, *new_shape[0])
    assert out_data[1].shape == (batch_size, *new_shape[1])

    # test only one reshape
    new_shape = [(4, 3), None]
    reshape_module = pmm.modules.Reshape(shape=new_shape)
    reshape_module.validate_shape(
        [arr.shape[1:] for arr in in_data]
    )  # Validate shape
    out_data, _ = reshape_module(in_data)
    assert out_data[0].shape == (batch_size, *new_shape[0])
    assert out_data[1].shape == in_data[1].shape

    # test reshape with -1 and different number of dimensions
    new_shape = [(-1, 2), (8,)]
    reshape_module = pmm.modules.Reshape(shape=new_shape)
    reshape_module.validate_shape(
        [arr.shape[1:] for arr in in_data]
    )  # Validate shape
    out_data, _ = reshape_module(in_data)
    assert out_data[0].shape == (batch_size, 6, 2)
    assert out_data[1].shape == (batch_size, 8)

    # Test invalid reshape
    new_shape = [(5, 3), (4, 2)]
    reshape_module = pmm.modules.Reshape(shape=new_shape)
    with pytest.raises(AssertionError):
        reshape_module.validate_shape(
            [arr.shape[1:] for arr in in_data]
        )  # Validate shape
    with pytest.raises(TypeError):
        out_data, _ = reshape_module(in_data)

    # test identity reshape
    new_shape = None
    reshape_module = pmm.modules.Reshape(shape=new_shape)
    reshape_module.validate_shape(
        [arr.shape[1:] for arr in in_data]
    )  # Validate shape
    out_data, _ = reshape_module(in_data)
    assert out_data[0].shape == in_data[0].shape
    assert out_data[1].shape == in_data[1].shape

    # test invalid reshape for mismatched PyTree structure
    new_shape = [((4, 3),), (4, 2)]
    reshape_module = pmm.modules.Reshape(shape=new_shape)
    with pytest.raises(AssertionError):
        reshape_module.validate_shape(
            [arr.shape[1:] for arr in in_data]
        )  # Validate shape
    with pytest.raises(TypeError):
        out_data, _ = reshape_module(in_data)
