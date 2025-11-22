import jax
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
    assert np.all(out_array == 1.0)
    assert np.all(out_pytree == 1.0)
    assert out_array.shape == (2,)  # batch dimension
    assert out_pytree.shape == (2,)  # batch dimension

    # check params
    const.get_params()
    const.get_hyperparameters()


def test_constant_array():
    r"""
    Test Constant module with array value.
    """
    const_array = np.array([1.0, 2.0, 3.0])

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
    assert np.all(const_array == out_array)
    assert np.all(const_array == out_pytree)
    assert out_array.shape == (2, 3)  # batch dimension + array shape
    assert out_pytree.shape == (2, 3)  # batch dimension + array shape


def test_constant_pytree():
    r"""
    Test Constant module with pytree value.
    """

    const_pytree = (
        np.array(0.0),
        np.array([1.0, 2.0]),
        np.array([[[3.0]], [[4.0]]]),
    )

    const = pmm.modules.Constant(const_pytree)

    # check with array and pytree inputs
    input_array = np.array([[1.0, 2.0], [3.0, 4.0]])
    input_pytree = (
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([[5.0, 6.0], [7.0, 8.0]]),
    )

    out_array, _ = const(input_array)
    out_pytree, _ = const(input_pytree)

    def check_equal(leaf1, leaf2):
        assert np.all(leaf1 == leaf2)

    jax.tree.map(check_equal, const_pytree, out_array)
    jax.tree.map(check_equal, const_pytree, out_pytree)


def test_trainable_constant():
    r"""
    Test Constant module with trainable parameter.
    """

    # trainable array constant
    shape = (2, 3)
    const = pmm.modules.Constant(
        trainable=True,
        shape=shape,
        real=True,
    )

    k = jax.random.key(0)

    print(const.real)
    print(const.shape)

    const.compile(k, (2,))

    params = const.get_params()
    out, _ = const(np.array([[1.0, 2.0], [3.0, 4.0]]))

    assert np.all(params == out)
    assert out.shape == (2,) + params.shape  # batch dimension + param shape

    # trainable pytree constant
    shape = (
        (1,),
        (
            (
                2,
                2,
            ),
        ),
    )
    const = pmm.modules.Constant(
        trainable=True, shape=shape, real=(True, (False,))
    )

    k = jax.random.key(0)
    const.compile(k, (2,))

    params = const.get_params()
    out, _ = const(np.array([[1.0, 2.0], [3.0, 4.0]]))

    def check_equal(leaf1, leaf2):
        assert np.all(leaf1 == leaf2)

    jax.tree.map(check_equal, params, out)

    def check_shape(leaf, shape):
        assert leaf.shape == (2,) + shape  # batch dimension + param shape

    jax.tree.map(check_shape, out, shape)
