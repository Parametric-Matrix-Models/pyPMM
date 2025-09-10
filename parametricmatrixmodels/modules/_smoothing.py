import jax
import jax.numpy as np


def commutator(
    A: np.ndarray,
    B: np.ndarray,
) -> np.ndarray:
    """
    Compute the commutator [A, B] = AB - BA.
    """
    return A @ B - B @ A


def exact_smoothing_matrix(
    Ms: np.ndarray,
) -> np.ndarray:
    """
    Compute the exact smoothing matrix for a set of Hermitian matrices.

    Which is 1j times the sum of all commutators between pairs of matrices.

    C = sqrt(-1) sum_{ij} [M_i, M_j]

    This is done efficiently using the linearity of the commutator and
    cumulative sums.
    """

    Ms_sums = np.cumsum(Ms[::~0], axis=0)[::~0]

    def scan_Ms_comms(acc: np.ndarray, i: int) -> np.ndarray:
        """
        Compute the commutator [M_i, M_j] for j >= i and accumulate.
        """
        # TODO: shouldn't this raise a jax exception since i is dynamic?
        # for some reason it doesn't, but maybe this leads to unexpected
        # behavior in some cases?
        return acc + commutator(Ms[i], Ms_sums[i]), ()

    smoothing_matrix, _ = jax.lax.scan(
        scan_Ms_comms,
        np.zeros_like(Ms[0]),
        np.arange(Ms.shape[0] - 1),  # don't include the last index
    )

    return 1j * smoothing_matrix
