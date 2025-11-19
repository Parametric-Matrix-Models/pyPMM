import jax
import jax.numpy as np
import jaxtyping
import pytest

import parametricmatrixmodels as pmm


def test_bias_scalar():
    r"""
    Test Bias module with scalar bias.
    """
    bias = pmm.modules.Bias(1.0, scalar=True)

    # check with array and pytree inputs
    input_array = np.array([[1.0, 2.0], [3.0, 4.0]])
    input_pytree = (
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([[5.0, 6.0], [7.0, 8.0]]),
    )

    out_array, _ = bias(input_array)
    out_pytree, _ = bias(input_pytree)

    # check array output
    assert np.allclose(out_array, input_array + 1.0)

    # check that the pytree structure is preserved
    assert jax.tree.structure(out_pytree) == jax.tree.structure(input_pytree)
    # check the output values
    assert all(
        [
            np.allclose(o, i + 1.0)
            for o, i in zip(
                jax.tree.leaves(out_pytree), jax.tree.leaves(input_pytree)
            )
        ]
    )


def test_bias_array():
    r"""
    Test Bias module with array bias.
    """
    bias_array = np.array([1.0, 2.0])

    bias = pmm.modules.Bias(bias_array, scalar=False)

    # check with array and pytree inputs
    input_array = np.array([[1.0, 2.0], [3.0, 4.0]])
    input_pytree = (
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([[5.0, 6.0], [7.0, 8.0]]),
    )

    out_array, _ = bias(input_array)
    out_pytree, _ = bias(input_pytree)

    # check array output
    expected_array = input_array + bias_array
    assert np.allclose(out_array, expected_array)

    # check that the pytree structure is preserved
    assert jax.tree.structure(out_pytree) == jax.tree.structure(input_pytree)

    # check the output values
    expected_pytree = jax.tree.map(lambda x: x + bias_array, input_pytree)

    assert all(
        [
            np.allclose(o, e)
            for o, e in zip(
                jax.tree.leaves(out_pytree), jax.tree.leaves(expected_pytree)
            )
        ]
    )


def test_type_checking():
    r"""
    Test type checking of Bias module.
    """

    bias = pmm.modules.Bias(1.0, scalar=True)

    # check that valid types pass
    in_array = np.array([[1.0, 2.0], [3.0, 4.0]])
    in_tuple = (in_array, in_array)
    in_list = [in_array, in_array]
    in_dict = {"a": in_array, "b": in_array}
    in_mixed = (in_array, [in_array, {"a": in_array}])

    for valid_input in [in_array, in_tuple, in_list, in_dict, in_mixed]:
        try:
            bias(valid_input)
        except Exception as e:
            pytest.fail(f"Valid input type raised an exception: {e}")

    # check that invalid types raise TypeError
    invalid_inputs = [
        42,
        "invalid",
        3.14,
        (in_array, 42),
        [in_array, "invalid"],
        {"a": in_array, "b": 3.14},
    ]
    for invalid_input in invalid_inputs:
        with pytest.raises(jaxtyping.TypeCheckError):
            bias(invalid_input)
