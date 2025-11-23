import jax
import jax.numpy as np

import parametricmatrixmodels as pmm


def test_matmul_array():
    r"""
    Test MatMul module with array inputs.
    """

    # testing with bare arrays
    key = jax.random.key(0)
    dkey, mkey = jax.random.split(key)
    d = jax.random.normal(dkey, (10, 4, 5))

    matmul = pmm.modules.MatMul(3)

    matmul.compile(mkey, d.shape[1:])  # remove batch dimension
    out, _ = matmul(d)

    matrix = matmul.get_params()
    expected_out = np.einsum("bij,jk->bik", d, matrix)

    assert out.shape == (10, 4, 3)
    assert np.allclose(out, expected_out)


def test_matmul_pytree():
    r"""
    Test MatMul module with pytree inputs.
    """

    # testing with pytrees
    key = jax.random.key(0)
    dkey0, dkey1, dkey2, mkey = jax.random.split(key, 4)
    d = (
        jax.random.normal(dkey0, (10, 4, 5)),
        (
            jax.random.normal(dkey1, (10, 4, 6)),
            jax.random.normal(dkey2, (10, 6)),
        ),
    )

    matmul = pmm.modules.MatMul((3, (1, 4)))
    matmul.compile(
        mkey, jax.tree.map(lambda x: x.shape[1:], d)
    )  # remove batch dimension

    out, _ = matmul(d)

    matrices = matmul.get_params()

    expected_out = (
        np.einsum("bij,jk->bik", d[0], matrices[0]),
        (
            np.einsum("bij,jk->bik", d[1][0], matrices[1][0]),
            np.einsum("bi,ik->bk", d[1][1], matrices[1][1]),
        ),
    )

    assert jax.tree.structure(out) == jax.tree.structure(expected_out)

    def check_shape_and_close(o, e):
        assert o.shape == e.shape
        assert np.allclose(o, e)

    jax.tree.map(check_shape_and_close, out, expected_out)
