import jax
import jax.numpy as np

import parametricmatrixmodels as pmm

jax.config.update("jax_enable_x64", True)


def test_transitionamplitudesum():
    r"""
    Test the TransitionAmplitudeSum module.
    """
    tas = pmm.modules.TransitionAmplitudeSum

    key = jax.random.key(0)

    rkey, ikey = jax.random.split(key)

    batch_size = 15
    matrix_size = 5
    num_eig = 3
    num_observables = 3
    output_size = 2
    init_magnitude = 1.0
    Vs = jax.random.normal(
        rkey, (batch_size, matrix_size, num_eig), dtype=np.complex64
    ) + 1j * jax.random.normal(
        rkey,
        (batch_size, matrix_size, num_eig),
    )
    # orthonormalize Vs
    Vs = jax.vmap(lambda x: jax.numpy.linalg.qr(x)[0], in_axes=0, out_axes=0)(
        Vs
    )

    m = tas(
        num_observables=num_observables,
        output_size=output_size,
        init_magnitude=init_magnitude,
    )
    m.compile(key, Vs.shape[1:])  # remove batch dimension
    Ds = m.get_params()

    out, _ = m(Vs)

    expected = np.einsum("dai,klab,dbj->dklij", Vs.conj(), Ds, Vs)
    expected = np.sum(np.abs(expected) ** 2, axis=(2, 3, 4))
    norm_term = 0.5 * np.sum(
        np.linalg.norm(Ds, axis=(2, 3), ord=2) ** 2, axis=1
    )
    expected -= norm_term

    assert np.allclose(out, expected)

    # test pytree preservation

    Vs = {"a": ([Vs],)}
    m = tas(
        num_observables=num_observables,
        output_size=output_size,
        init_magnitude=init_magnitude,
    )
    m.compile(
        key, pmm.tree_util.get_shapes(Vs, slice(1, None))
    )  # remove batch dimension
    out, _ = m(Vs)

    out_struct = jax.tree.structure(out)
    expected_struct = jax.tree.structure(Vs)
    assert out_struct == expected_struct
    _, leaf = pmm.tree_util.is_single_leaf(out)
    assert np.allclose(leaf, expected)
