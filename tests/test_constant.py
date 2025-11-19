import jax.numpy as np

import parametricmatrixmodels as pmm


def test_constant_scalar():
    r"""
    Test Constant module with scalar value.
    """
    const = pmm.modules.Constant(1.0)

    # check with array and pytree inputs
    input_array = np.array([[1.0, 2.0], [3.0, 4.0]])
    input_pytree = (
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([[5.0, 6.0], [7.0, 8.0]]),
    )

    out_array, _ = const(input_array)
    out_pytree, _ = const(input_pytree)

    # check array output
    assert np.allclose(out_array, np.array(1.0))
    assert np.allclose(out_pytree, np.array(1.0))


def test_constant_array():
    r"""
    Test Constant module with array value.
    """
    const_array = np.array([1.0, 2.0])

    const = pmm.modules.Constant(const_array)

    # check with array and pytree inputs
    input_array = np.array([[1.0, 2.0], [3.0, 4.0]])
    input_pytree = (
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([[5.0, 6.0], [7.0, 8.0]]),
    )

    out_array, _ = const(input_array)
    out_pytree, _ = const(input_pytree)

    # check array output
    assert np.allclose(out_array, const_array)
    assert np.allclose(out_pytree, const_array)
