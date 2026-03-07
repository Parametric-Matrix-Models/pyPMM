import jax
import jax.numpy as np
from jaxtyping import Array, Inexact

from ..typing import Tuple


def commutator(
    A: Inexact[Array, "d d"],
    B: Inexact[Array, "d d"],
) -> Inexact[Array, "d d"]:
    """
    Compute the commutator [A, B] = AB - BA.
    """
    return A @ B - B @ A


# define the scan outside of exact_smoothing_matrix to avoid recompiling it
# (breaks persistent compilation cache otherwise)
def scan_Ms_comms(
    acc: Inexact[Array, "d d"],
    Mi_Mcs_im1: Tuple[Inexact[Array, "d d"], Inexact[Array, "d d"]],
) -> Tuple[Inexact[Array, "d d"], None]:
    Mi, Mcs_im1 = Mi_Mcs_im1
    return acc + commutator(Mi, Mcs_im1), None


def exact_smoothing_matrix(
    Ms: Inexact[Array, "n d d"],
) -> Inexact[Array, "d d"]:
    r"""
    Compute the exact smoothing matrix for a set of Hermitian matrices.

    Which is 1j times the sum of all commutators between pairs of matrices.

    C = sqrt(-1) sum_{i>j} [M_i, M_j]

    This is done efficiently using the linearity of the commutator and
    cumulative sums.

    By the linearity of the commutator, we can write this as

    sum_{i>j} [M_i, M_j] = sum_i [M_i, sum_{j<i} M_j]

    B = cumsum(Ms[:-1], axis=0) gives us sum_{j<=i} M_j for each i, except the
    last.

    B[0] = Ms[0]
    B[1] = Ms[0] + Ms[1]
    ...
    B[i] = sum_{j<=i} M_j

    so

    sum_{i>j} [M_i, M_j] = sum_i [M_i, B[i-1]] = sum_{i=1} [M_i, B[i-1]]

    since [M_0, B[-1]] = [M_0, 0] = 0.

    So:

    \sum_{i=1} [M_i, B[i-1]]
        = [M_1, M_0]
        + [M_2, M_0 + M_1]
        + [M_3, M_0 + M_1 + M_2]
        + ...
        + [M_{n-1}, M_0 + M_1 + ... + M_{n-2}]
        + [M_n, M_0 + M_1 + ... + M_{n-1}]
        = sum_{i>j} [M_i, M_j]

    as desired.
    """

    Ms_cumsum = np.cumsum(Ms[:-1], axis=0)

    # pack together the Ms[1:] and Ms_cumsum to scan over
    scan_input = (Ms[1:], Ms_cumsum)

    smoothing_matrix, _ = jax.lax.scan(
        scan_Ms_comms,
        np.zeros_like(Ms[0]),
        scan_input,
    )

    return 1j * smoothing_matrix
