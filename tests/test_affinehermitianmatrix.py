import jax
import jax.numpy as np
import pytest

import parametricmatrixmodels as pmm

jax.config.update("jax_enable_x64", True)


def test_affine_hermitian_matrix():
    r"""
    Test the AffineHermitianMatrix module
    """
    ahm = pmm.modules.AffineHermitianMatrix

    key = jax.random.key(0)

    batch_size = 5
    num_features = 3
    matrix_size = 4
    Ms = jax.random.normal(
        key, (num_features + 1, matrix_size, matrix_size), dtype=np.complex64
    ) + 1j * jax.random.normal(
        key, (num_features + 1, matrix_size, matrix_size), dtype=np.complex64
    )
    Ms = (Ms + np.conj(np.swapaxes(Ms, 1, 2))) / 2  # Make Hermitian

    arr = jax.random.normal(key, (batch_size, num_features), dtype=np.float32)

    m = ahm(matrix_size=matrix_size, Ms=Ms, bias_term=True)
    m.compile(None, arr.shape[1:])  # remove batch dimension

    out, _ = m(arr)

    expected = Ms[0][None, :, :]  # bias term
    expected += np.einsum("bf,fij->bij", arr, Ms[1:])
    assert np.allclose(out, expected)

    # no bias (different callable and signature)
    m = ahm(matrix_size=matrix_size, Ms=Ms[1:], bias_term=False)
    m.compile(None, arr.shape[1:])  # remove batch dimension
    out, _ = m(arr)
    expected = np.einsum("bf,fij->bij", arr, Ms[1:])
    assert np.allclose(out, expected)

    # inferred feature count (Ms not given)
    m = ahm(matrix_size=matrix_size, bias_term=True)
    m.compile(key, arr.shape[1:])  # remove batch dimension
    out, _ = m(arr)

    Ms = m.get_params()

    expected = Ms[0][None, :, :]  # bias term
    expected += np.einsum("bf,fij->bij", arr, Ms[1:])
    assert np.allclose(out, expected)

    # pytree preservation
    data = {"input_data": [(arr,)]}
    m = ahm(matrix_size=matrix_size, bias_term=True)
    m.compile(key, jax.tree.map(lambda x: x.shape[1:], data))
    out, _ = m(data)
    Ms = m.get_params()
    expected = Ms[0][None, :, :]  # bias term
    expected += np.einsum("bf,fij->bij", arr, Ms[1:])

    orig_struct = jax.tree.structure(data)
    out_struct = jax.tree.structure(out)
    assert orig_struct == out_struct
    assert np.allclose(out["input_data"][0][0], expected)

    # purposeful errors
    with pytest.raises(ValueError):
        # only trees with a single array leaf are allowed
        data = {"input_data": [(arr, arr)]}
        m = ahm(matrix_size=matrix_size, bias_term=True)
        m.compile(key, jax.tree.map(lambda x: x.shape[1:], data))
    with pytest.raises(ValueError):
        # the leaf in the pytree must be 1D (excluding batch dim)
        arr_2d = jax.random.normal(key, (batch_size, num_features, 2))
        data = {"input_data": [(arr_2d,)]}
        m = ahm(matrix_size=matrix_size, bias_term=True)
        m.compile(key, jax.tree.map(lambda x: x.shape[1:], data))
