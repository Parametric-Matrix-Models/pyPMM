import jax
import jax.numpy as np
import pytest

import parametricmatrixmodels as pmm

jax.config.update("jax_enable_x64", True)


def _string_check(s: str, expected: str) -> bool:
    assert s == expected, f"Expected {expected}, but got {s}"


def test_einsum_string():
    r"""
    Test Einsum module _get_concrete_einsum_str method.
    """
    E = pmm.modules.Einsum

    # explicit string
    e = E("ij,jk->ik")

    # no leading arrays
    s = e._get_concrete_einsum_str(((3, 4), (4, 5)))
    _string_check(s, "aij,ajk->aik")
    # leading arrays
    s = e._get_concrete_einsum_str((4, 5))
    _string_check(s, "ij,ajk->aik")

    # implicit string
    e = E("ij,jk")

    # no leading arrays
    s = e._get_concrete_einsum_str(((3, 4), (4, 5)))
    _string_check(s, "aij,ajk->aik")
    # leading arrays
    s = e._get_concrete_einsum_str((4, 5))
    _string_check(s, "ij,ajk->aik")

    # PyTrees

    # no leading arrays, explicit output
    e = E(({"a": "ab", "b": ("bc", "c")}, "ac"))
    s = e._get_concrete_einsum_str(({"a": (3, 4), "b": ((4, 5), (5,))}))
    _string_check(s, "dab,dbc,dc->dac")

    # no leading arrays, explicit output, different order than flattened order
    e = E(({"b": "ab", "a": ("bc", "c")}, "ac"))
    s = e._get_concrete_einsum_str(({"a": ((4, 5), (5,)), "b": (3, 4)}))
    _string_check(s, "dbc,dc,dab->dac")

    # no leading arrays, explicit output, different order than flattened order
    e = E(({"a": "ab", "b": ("bc", "c")}, "ac"))
    s = e._get_concrete_einsum_str(({"b": ((4, 5), (5,)), "a": (3, 4)}))
    _string_check(s, "dab,dbc,dc->dac")

    # no leading arrays, implicit output
    e = E({"a": "ab", "b": ("bc", "c")})
    s = e._get_concrete_einsum_str(({"a": (3, 4), "b": ((4, 5), (5,))}))
    _string_check(s, "dab,dbc,dc->da")

    # leading arrays, explicit output
    e = E(("ij,jk", {"a": "ab", "b": ("bc", "c")}, "acik"))
    s = e._get_concrete_einsum_str({"a": (5, 6), "b": ((6, 7), (7,))})
    _string_check(s, "ij,jk,dab,dbc,dc->dacik")

    # leading arrays, implicit output
    e = E(("ij,jk", {"a": "ab", "b": ("bc", "c")}))
    s = e._get_concrete_einsum_str({"a": (5, 6), "b": ((6, 7), (7,))})
    _string_check(s, "ij,jk,dab,dbc,dc->daik")

    # deliberately confusing case
    # it looks like it should be a leading array, PyTree (dict) input, and
    # explicit out, but its actually just a weird PyTree (tuple of arr, dict,
    # arr) input with implicit out
    e = E(("ij", {"a": "ab", "b": ("bc", "c")}, "acij"))
    s = e._get_concrete_einsum_str(
        ((3, 4), {"a": (5, 6), "b": ((6, 7), (7,))}, (5, 7, 3, 4))
    )
    _string_check(s, "dij,dab,dbc,dc,dacij->d")

    # deliberate errors
    with pytest.raises(ValueError):
        E(("ij", "jk,kl,l"))
        e._get_concrete_einsum_str(((3, 4), (5, 6)))
    with pytest.raises(ValueError):
        E(("ij", "jk"))
        # wrong structure
        e._get_concrete_einsum_str([(3, 4), (5, 6)])
    with pytest.raises(ValueError):
        # num dimensions mismatch
        E("ij,jk->ik")
        e._get_concrete_einsum_str(((3, 4), (5, 6, 7)))
    with pytest.raises(ValueError):
        # shape mismatch
        E("ij,jk->ik")
        e._get_concrete_einsum_str(((3, 4), (6, 7)))
        e._get_dimension_map(s, ((3, 4), (6, 7)))


def test_einsum_output_shape():
    r"""
    Test Einsum module get_output_shape method.
    """

    E = pmm.modules.Einsum

    # explicit string
    e = E("ij,jk->ik")
    out_shape = e.get_output_shape(((3, 4), (4, 5)))
    assert out_shape == (3, 5)

    # implicit string
    e = E("ij,jk")
    out_shape = e.get_output_shape(((3, 4), (4, 5)))
    assert out_shape == (3, 5)

    # PyTrees

    # no leading arrays, explicit output
    e = E(({"a": "ab", "b": ("bc", "c")}, "ac"))
    out_shape = e.get_output_shape(({"a": (3, 4), "b": ((4, 5), (5,))}))
    assert out_shape == (3, 5)

    # no leading arrays, implicit output
    e = E({"a": "ab", "b": ("bc", "c")})
    out_shape = e.get_output_shape(({"a": (3, 4), "b": ((4, 5), (5,))}))
    assert out_shape == (3,)


def test_einsum():
    r"""
    Test Einsum module.
    """

    # testing with bare arrays
    key = jax.random.key(0)
    A = jax.random.normal(key, (3, 4))
    d = jax.random.normal(key, (10, 4, 5))

    # testing with explicit indexing
    einsum_ex = pmm.modules.Einsum("ij,jk->ik", A, trainable=False)
    # testing with implicit indexing
    einsum_im = pmm.modules.Einsum("ij,jk", A, trainable=False)

    einsum_ex.compile(key, d.shape[1:])  # remove batch dim
    einsum_im.compile(key, d.shape[1:])

    out_ex, _ = einsum_ex(d)
    out_im, _ = einsum_im(d)

    expected_out = np.einsum("ij,ajk->aik", A, d)

    assert np.allclose(out_ex, expected_out)
    assert np.allclose(out_im, expected_out)

    # repeat without specifying A during initialization
    einsum_ex = pmm.modules.Einsum(
        "ij,jk->ik", dim_map={"i": 3, "j": 4, "k": 5}, trainable=True
    )
    einsum_im = pmm.modules.Einsum(
        "ij,jk", dim_map={"i": 3, "j": 4, "k": 5}, trainable=True
    )

    einsum_ex.compile(key, d.shape[1:])  # remove batch dim
    einsum_im.compile(key, d.shape[1:])
    out_ex, _ = einsum_ex(d)
    out_im, _ = einsum_im(d)

    Aex = einsum_ex.get_params()
    Aim = einsum_im.get_params()

    expected_out = np.einsum("ij,ajk->aik", Aex, d)
    assert np.allclose(out_ex, expected_out)
    expected_out = np.einsum("ij,ajk->aik", Aim, d)
    assert np.allclose(out_im, expected_out)

    # no parameters
    einsum_ex = pmm.modules.Einsum("ij,jk->ik")
    einsum_im = pmm.modules.Einsum("ij,jk")

    einsum_ex.compile(key, ((3, 4), (4, 5)))
    einsum_im.compile(key, ((3, 4), (4, 5)))

    d1 = jax.random.normal(key, (10, 3, 4))
    d2 = jax.random.normal(key, (10, 4, 5))
    ds = (d1, d2)
    out_ex, _ = einsum_ex(ds)
    out_im, _ = einsum_im(ds)
    expected_out = np.einsum("aij,ajk->aik", d1, d2)
    assert np.allclose(out_ex, expected_out)
    assert np.allclose(out_im, expected_out)

    # PyTrees and parameter arrays
    batch_dim = 10
    key = jax.random.key(0)
    d = {
        "y": (
            jax.random.normal(key, (batch_dim, 4, 5)),
            jax.random.normal(key, (batch_dim, 5)),
        ),
        "x": jax.random.normal(key, (batch_dim, 3, 4)),
    }

    einsum_ex = pmm.modules.Einsum(
        ("ij,jk", {"x": "ab", "y": ("bc", "c")}, "aik"),
        dim_map={"i": 3, "j": 4, "k": 3},  # only specify uninferrable dims
        trainable=True,
    )
    einsum_im = pmm.modules.Einsum(
        ("ij,jk", {"x": "ab", "y": ("bc", "c")}),
        dim_map={"i": 3, "j": 4, "k": 3},  # only specify uninferrable dims
        trainable=True,
    )

    einsum_ex.compile(key, jax.tree.map(lambda x: x.shape[1:], d))
    einsum_im.compile(key, jax.tree.map(lambda x: x.shape[1:], d))
    out_ex, _ = einsum_ex(d)
    out_im, _ = einsum_im(d)

    params = einsum_ex.get_params()

    # get common dtype
    dtypes = [p.dtype for p in params]
    dtypes += [v.dtype for v in jax.tree.leaves(d)]
    dtype = np.result_type(*dtypes)
    params = [p.astype(dtype) for p in params]
    d = jax.tree.map(lambda x: x.astype(dtype), d)

    expected_out = np.einsum(
        "ij,jk,dab,dbc,dc->daik", *params, d["x"], d["y"][0], d["y"][1]
    )
    assert np.allclose(out_ex, expected_out)
    assert np.allclose(out_im, expected_out)
