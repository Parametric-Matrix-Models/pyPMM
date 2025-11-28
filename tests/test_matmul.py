import jax
import jax.numpy as np

import parametricmatrixmodels as pmm


def test_matmul():
    r"""
    Test MatMul module
    """

    # testing with bare arrays
    key = jax.random.key(0)
    dkey, mkey = jax.random.split(key)
    d = jax.random.normal(dkey, (10, 4, 5))

    matmul = pmm.modules.MatMul(output_shape=3, trainable=True)

    matmul.compile(mkey, d.shape[1:])  # remove batch dimension
    out, _ = matmul(d)

    matrix = matmul.get_params()
    expected_out = np.einsum("ij,bjk->bik", matrix, d)

    assert out.shape == (10, 3, 5)
    assert np.allclose(out, expected_out)

    # pytree of arrays, multiplied in flattened order with a fixed matrix
    batch_dim = 10
    d = (
        [
            jax.random.normal(dkey, (10, 4, 5)),
        ],
        jax.random.normal(dkey, (10, 5, 3)),
        {
            "a": jax.random.normal(dkey, (10, 3, 2)),
        },
    )
    M = jax.random.normal(mkey, (4 * 5 * 3, 4))

    matmul = pmm.modules.MatMul(params=M, trainable=False)
    matmul.compile(
        None, jax.tree.map(lambda x: x.shape[1:], d)
    )  # remove batch dimension

    out, _ = matmul(d)
    expected_out = np.einsum(
        "ij,bjk,bkl,blm->bim", M, d[0][0], d[1], d[2]["a"]
    )

    assert out.shape == (10, 4 * 5 * 3, 2)
    assert np.allclose(out, expected_out)

    # pytree of arrays, multiplied in an order different from the flattened
    # one, with a fixed vector at the end
    d = (
        [
            jax.random.normal(dkey, (10, 5, 3)),
        ],
        jax.random.normal(dkey, (10, 3, 2)),
        {
            "a": jax.random.normal(dkey, (10, 4, 5)),
        },
    )

    path_order = ["2.a", "0.0", "1", ".."]

    M = jax.random.normal(mkey, (2,))

    matmul = pmm.modules.MatMul(
        params=M, path_order=path_order, trainable=False
    )
    matmul.compile(
        None, jax.tree.map(lambda x: x.shape[1:], d)
    )  # remove batch dimension

    out, _ = matmul(d)
    expected_out = np.einsum(
        "bij,bjk,bkl,l->bi",
        d[2]["a"],
        d[0][0],
        d[1],
        M,
    )

    # trainable matrix with input bare vector
    d = jax.random.normal(dkey, (batch_dim, 6))
    matmul = pmm.modules.MatMul(output_shape=4, trainable=True)
    matmul.compile(mkey, d.shape[1:])  # remove batch dimension
    out, _ = matmul(d)
    matrix = matmul.get_params()
    expected_out = np.einsum("ij,bj->bi", matrix, d)
    assert out.shape == (batch_dim, 4)
    assert np.allclose(out, expected_out)
