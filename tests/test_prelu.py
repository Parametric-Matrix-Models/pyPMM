import jax
import jax.numpy as np

import parametricmatrixmodels as pmm


def test_prelu_single():
    r"""
    Test PReLU module with scalar value.
    """
    # check with array and pytree inputs
    input_array = np.array([[1.0, 2.0], [3.0, 4.0]])
    input_pytree = (
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([[5.0, 6.0], [7.0, 8.0]]),
    )

    key = jax.random.key(0)

    prelu = pmm.modules.PReLU(True)

    prelu.compile(key, input_array.shape[1:])  # exclude batch dimension

    a = prelu.get_params()[0]

    out_array, _ = prelu(input_array)
    expected_array = jax.nn.leaky_relu(input_array, negative_slope=a)
    assert np.allclose(out_array, expected_array)

    # Test with pytree input
    prelu = pmm.modules.PReLU(True)
    prelu.compile(key, (input_pytree[0].shape[1:], input_pytree[1].shape[1:]))
    a = prelu.get_params()[0]
    out_pytree, _ = prelu(input_pytree)
    expected_pytree = (
        jax.nn.leaky_relu(input_pytree[0], negative_slope=a),
        jax.nn.leaky_relu(input_pytree[1], negative_slope=a),
    )
    assert np.allclose(out_pytree[0], expected_pytree[0])
    assert np.allclose(out_pytree[1], expected_pytree[1])
    assert jax.tree.structure(out_pytree) == jax.tree.structure(input_pytree)


def test_prelu_nonsingle():
    r"""
    Test PReLU module with array/pytree values.
    """

    # check with array and pytree inputs
    input_array = np.array([[1.0, -2.0], [-3.0, 4.0]])
    input_pytree = (
        np.array([[1.0, -2.0], [-3.0, 4.0]]),
        np.array([[-5.0, 6.0], [7.0, -8.0]]),
    )

    key = jax.random.key(0)
    prelu = pmm.modules.PReLU(False)

    prelu.compile(key, input_array.shape[1:])  # exclude batch dimension

    a = prelu.get_params()

    out_array, _ = prelu(input_array)
    expected_array = jax.nn.leaky_relu(input_array, negative_slope=a)

    assert np.allclose(out_array, expected_array)
    assert out_array.shape == input_array.shape

    # Test with pytree input
    prelu = pmm.modules.PReLU(False)
    prelu.compile(key, (input_pytree[0].shape[1:], input_pytree[1].shape[1:]))
    a = prelu.get_params()
    out_pytree, _ = prelu(input_pytree)
    expected_pytree = (
        jax.nn.leaky_relu(input_pytree[0], negative_slope=a[0]),
        jax.nn.leaky_relu(input_pytree[1], negative_slope=a[1]),
    )
    assert np.allclose(out_pytree[0], expected_pytree[0])
    assert np.allclose(out_pytree[1], expected_pytree[1])
    assert out_pytree[0].shape == input_pytree[0].shape
    assert out_pytree[1].shape == input_pytree[1].shape
    assert jax.tree.structure(out_pytree) == jax.tree.structure(input_pytree)
