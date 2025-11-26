import jax
import jax.numpy as np

import parametricmatrixmodels as pmm

jax.config.update("jax_enable_x64", True)


def test_eigenvalues():
    r"""
    Test the Eigenvalues module
    """

    key = jax.random.key(0)
    batch_dim = 2
    N = 5

    # test with real array inputs
    A = jax.random.normal(key, (batch_dim, N, N)).astype(np.float32)
    A = (A + np.swapaxes(A, -1, -2)) / 2  # make symmetric

    m = pmm.modules.Eigenvalues(num_eig=None)

    m.compile(None, (N, N))
    evals, _ = m(A)

    # compare with jax.numpy.linalg.eigvalsh
    expected = np.linalg.eigvalsh(A)

    assert np.allclose(
        evals, expected
    ), "Eigenvalues do not match expected values"

    # test with pytree complex (Hermitian) array inputs
    Ns = [3, 4, 5]

    def create_hermitian_matrix(n, key):
        rkey, ikey = jax.random.split(key)
        A = jax.random.normal(
            rkey, (batch_dim, n, n)
        ) + 1j * jax.random.normal(ikey, (batch_dim, n, n))
        A = (A + np.swapaxes(np.conj(A), -1, -2)) / 2  # make Hermitian
        return A.astype(np.complex64)

    keys = jax.random.split(key, len(Ns))

    A_tree = [create_hermitian_matrix(N, k) for N, k in zip(Ns, keys)]

    m = pmm.modules.Eigenvalues(num_eig=None)
    m.compile(None, jax.tree.map(lambda n: (n, n), Ns))
    evals_tree, _ = m(A_tree)
    expected_tree = jax.tree.map(lambda A: np.linalg.eigvalsh(A), A_tree)

    assert jax.tree.all(
        jax.tree.map(
            lambda evals, expected: np.allclose(evals, expected),
            evals_tree,
            expected_tree,
        )
    ), "Eigenvalues for pytree inputs do not match expected values"

    # repeat previous test but with num_eig = 1, and the module as part of a
    # sequential model
    m = pmm.SequentialModel(
        [
            pmm.modules.Eigenvalues(num_eig=1, which="SA"),
        ]
    )
    m.compile(None, jax.tree.map(lambda n: (n, n), Ns))
    evals_tree = m(A_tree, dtype=np.complex64)
    expected_tree = jax.tree.map(
        lambda A: np.linalg.eigvalsh(A)[..., 0][..., None], A_tree
    )
    assert jax.tree.all(
        jax.tree.map(
            lambda evals, expected: np.allclose(evals, expected),
            evals_tree,
            expected_tree,
        )
    ), "Lowest eigenvalues for pytree inputs do not match expected values"


def test_eigenvectors():
    r"""
    Test the Eigenvectors module. Repeat of the eigenvalues test, but checking
    eigenvectors instead.
    """
    key = jax.random.key(0)
    batch_dim = 2
    N = 5
    # test with real array inputs
    A = jax.random.normal(key, (batch_dim, N, N)).astype(np.float32)
    A = (A + np.swapaxes(A, -1, -2)) / 2  # make symmetric
    m = pmm.modules.Eigenvectors(num_eig=None)
    m.compile
    evecs, _ = m(A)
    # compare with jax.numpy.linalg.eigh
    expected = np.linalg.eigh(A)[1]

    assert np.allclose(
        evecs, expected
    ), "Eigenvectors do not match expected values"

    # test with pytree complex (Hermitian) array inputs
    Ns = [3, 4, 5]

    def create_hermitian_matrix(n, key):
        rkey, ikey = jax.random.split(key)
        A = jax.random.normal(
            rkey, (batch_dim, n, n)
        ) + 1j * jax.random.normal(ikey, (batch_dim, n, n))
        A = (A + np.swapaxes(np.conj(A), -1, -2)) / 2  # make Hermitian
        return A.astype(np.complex64)

    keys = jax.random.split(key, len(Ns))
    A_tree = [create_hermitian_matrix(N, k) for N, k in zip(Ns, keys)]
    m = pmm.modules.Eigenvectors(num_eig=None)
    m.compile(None, jax.tree.map(lambda n: (n, n), Ns))
    evecs_tree, _ = m(A_tree)
    expected_tree = jax.tree.map(lambda A: np.linalg.eigh(A)[1], A_tree)
    assert jax.tree.all(
        jax.tree.map(
            lambda evecs, expected: np.allclose(evecs, expected),
            evecs_tree,
            expected_tree,
        )
    ), "Eigenvectors for pytree inputs do not match expected values"
    # repeat previous test but with num_eig = 1, and the module as part of a
    # sequential model
    m = pmm.SequentialModel(
        [
            pmm.modules.Eigenvectors(num_eig=1, which="SA"),
        ]
    )
    m.compile(None, jax.tree.map(lambda n: (n, n), Ns))
    evecs_tree = m(A_tree, dtype=np.complex64)
    expected_tree = jax.tree.map(
        lambda A: np.linalg.eigh(A)[1][..., 0:1], A_tree
    )
    assert jax.tree.all(
        jax.tree.map(
            lambda evecs, expected: np.allclose(evecs, expected),
            evecs_tree,
            expected_tree,
        )
    ), "Lowest eigenvectors for pytree inputs do not match expected values"


def test_eigensystem():
    r"""
    Test the Eigensystem module. Repeat of the eigenvalues and eigenvectors
    test, but checking both eigenvalues and eigenvectors together.
    """
    key = jax.random.key(0)
    batch_dim = 2
    N = 5
    # test with real array inputs
    A = jax.random.normal(key, (batch_dim, N, N)).astype(np.float32)
    A = (A + np.swapaxes(A, -1, -2)) / 2  # make symmetric
    m = pmm.modules.Eigensystem(num_eig=None)
    m.compile(None, (N, N))
    es, _ = m(A)
    evals, evecs = es["eigenvalues"], es["eigenvectors"]
    # compare with jax.numpy.linalg.eigh
    expected_evals, expected_evecs = np.linalg.eigh(A)
    assert np.allclose(
        evals, expected_evals
    ), "Eigenvalues do not match expected values"
    assert np.allclose(
        evecs, expected_evecs
    ), "Eigenvectors do not match expected values"

    # test with pytree complex (Hermitian) array inputs
    Ns = [3, 4, 5]

    def create_hermitian_matrix(n, key):
        rkey, ikey = jax.random.split(key)
        A = jax.random.normal(
            rkey, (batch_dim, n, n)
        ) + 1j * jax.random.normal(ikey, (batch_dim, n, n))
        A = (A + np.swapaxes(np.conj(A), -1, -2)) / 2  # make Hermitian
        return A.astype(np.complex64)

    keys = jax.random.split(key, len(Ns))
    A_tree = [create_hermitian_matrix(N, k) for N, k in zip(Ns, keys)]
    m = pmm.modules.Eigensystem(num_eig=None)
    m.compile(None, jax.tree.map(lambda n: (n, n), Ns))
    es_tree, _ = m(A_tree)
    evals_tree, evecs_tree = (
        jax.tree.map(
            lambda es: es["eigenvalues"],
            es_tree,
            is_leaf=lambda x: isinstance(x, dict),
        ),
        jax.tree.map(
            lambda es: es["eigenvectors"],
            es_tree,
            is_leaf=lambda x: isinstance(x, dict),
        ),
    )
    expected_evals_tree = jax.tree.map(lambda A: np.linalg.eigvalsh(A), A_tree)
    expected_evecs_tree = jax.tree.map(lambda A: np.linalg.eigh(A)[1], A_tree)
    assert jax.tree.all(
        jax.tree.map(
            lambda evals, expected: np.allclose(evals, expected),
            evals_tree,
            expected_evals_tree,
        )
    ), "Eigenvalues for pytree inputs do not match expected values"
    assert jax.tree.all(
        jax.tree.map(
            lambda evecs, expected: np.allclose(evecs, expected),
            evecs_tree,
            expected_evecs_tree,
        )
    ), "Eigenvectors for pytree inputs do not match expected values"

    # repeat previous test but with num_eig = 1, and the module as part of a
    # sequential model
    m = pmm.SequentialModel(
        [
            pmm.modules.Eigensystem(num_eig=1, which="SA"),
        ]
    )
    m.compile(None, jax.tree.map(lambda n: (n, n), Ns))
    es_tree = m(A_tree, dtype=np.complex64)
    evals_tree = jax.tree.map(
        lambda es: es["eigenvalues"],
        es_tree,
        is_leaf=lambda x: isinstance(x, dict),
    )
    evecs_tree = jax.tree.map(
        lambda es: es["eigenvectors"],
        es_tree,
        is_leaf=lambda x: isinstance(x, dict),
    )
    expected_evals_tree = jax.tree.map(
        lambda A: np.linalg.eigvalsh(A)[..., 0][..., None], A_tree
    )
    expected_evecs_tree = jax.tree.map(
        lambda A: np.linalg.eigh(A)[1][..., 0:1], A_tree
    )
    assert jax.tree.all(
        jax.tree.map(
            lambda evals, expected: np.allclose(evals, expected),
            evals_tree,
            expected_evals_tree,
        )
    ), "Lowest eigenvalues for pytree inputs do not match expected values"
    assert jax.tree.all(
        jax.tree.map(
            lambda evecs, expected: np.allclose(evecs, expected),
            evecs_tree,
            expected_evecs_tree,
        )
    ), "Lowest eigenvectors for pytree inputs do not match expected values"
