import jax
import jax.numpy as np

import parametricmatrixmodels as pmm

jax.config.update("jax_enable_x64", True)


def test_aepmm():
    r"""
    Test the AffineEigenvaluePMM Model/Module
    """

    key = jax.random.key(0)

    matrix_size = 5
    features = 3
    batch_size = 4

    arr = jax.random.normal(key, (batch_size, features), dtype=np.float32)

    aepmm = pmm.modules.AffineEigenvaluePMM(
        matrix_size=matrix_size,
        num_eig=1,
        which="SA",
        init_magnitude=1.0,
    )
    aepmm.compile(key, jax.tree.map(lambda x: x.shape[1:], arr))
    eigs = aepmm(arr, dtype=np.float32)

    Ms, _ = aepmm.get_params()

    M = Ms[0][None, :, :] + np.einsum("bp,pij->bij", arr, Ms[1:])
    true_eigs = np.linalg.eigvalsh(M)
    true_eigs = np.sort(true_eigs, axis=-1)[..., :1]
    assert np.allclose(eigs, true_eigs)

    # test preservation of PyTree structure

    data = {"data": [(arr,)]}
    aepmm = pmm.modules.AffineEigenvaluePMM(
        matrix_size=matrix_size,
        num_eig=2,
        which="LM",  # largest magnitude
        init_magnitude=1.0,
    )
    aepmm.compile(key, jax.tree.map(lambda x: x.shape[1:], data))
    eigs = aepmm(data, dtype=np.float32)
    Ms, _ = aepmm.get_params()
    M = Ms[0][None, :, :] + np.einsum("bp,pij->bij", arr, Ms[1:])
    true_eigs = np.linalg.eigvalsh(M)
    # sort by magnitude
    idx = np.argsort(np.abs(true_eigs), axis=-1)[:, ~1:]
    true_eigs = np.take_along_axis(true_eigs, idx, axis=-1)

    # check the structures
    eigs_struct = jax.tree.structure(eigs)
    orig_struct = jax.tree.structure(data)
    assert eigs_struct == orig_struct
    assert np.allclose(eigs["data"][0][0], true_eigs)
