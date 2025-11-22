import jax
import jax.numpy as np

import parametricmatrixmodels as pmm


def test_einsum_array():
    r"""
    Test Einsum module with array inputs.
    """

    # testing with bare arrays
    key = jax.random.key(0)
    A = jax.random.normal(key, (3, 4))
    d = jax.random.normal(key, (10, 4, 5))

    # testing with explicit indexing
    einsum_ex = pmm.modules.Einsum("ij,ki->jk", A)
    # testing with implicit indexing
    einsum_im = pmm.modules.Einsum("ij,ki", A)

    einsum_ex.compile(key, d.shape)
    einsum_im.compile(key, d.shape)

    out_ex, _ = einsum_ex(d)
    out_im, _ = einsum_im(d)

    expected_out = np.einsum("aij,ki->ajk", d, A)

    assert np.allclose(out_ex, expected_out)
    assert np.allclose(out_im, expected_out)


def test_einsum_pytree():
    r"""
    Test Einsum module with pytree inputs.
    """

    # testing with pytrees
    key = jax.random.key(0)
    A = (
        jax.random.normal(key, (3, 4)),
        (jax.random.normal(key, (5, 4)), jax.random.normal(key, (6,))),
    )
    d = (
        jax.random.normal(key, (10, 4, 5)),
        (
            jax.random.normal(key, (10, 4, 6)),
            jax.random.normal(key, (10, 6, 6)),
        ),
    )

    # testing with explicit indexing
    einsum_ex = pmm.modules.Einsum(("ij,ki->jk", ("ia,ki->ak", "aa,a->")), A)
    # testing with implicit indexing
    einsum_im = pmm.modules.Einsum(("ij,ki", ("aj,ka", "aa,a")), A)

    einsum_ex.compile(key, (d[0].shape, (d[1][0].shape, d[1][1].shape)))
    einsum_im.compile(key, (d[0].shape, (d[1][0].shape, d[1][1].shape)))

    print(einsum_ex._batch_einsum_str)

    out_ex, _ = einsum_ex(d)
    out_im, _ = einsum_im(d)

    expected_out = (
        np.einsum("aij,ki->ajk", d[0], A[0]),
        (
            np.einsum("aib,ki->abk", d[1][0], A[1][0]),
            np.einsum("abb,b->a", d[1][1], A[1][1]),
        ),
    )
    assert np.allclose(out_ex[0], expected_out[0])
    assert np.allclose(out_ex[1][0], expected_out[1][0])
    assert np.allclose(out_ex[1][1], expected_out[1][1])
    assert np.allclose(out_im[0], expected_out[0])
    assert np.allclose(out_im[1][0], expected_out[1][0])
    assert np.allclose(out_im[1][1], expected_out[1][1])
