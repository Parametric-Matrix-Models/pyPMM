from typing import Optional

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
    A: np.ndarray,
    Bs: np.ndarray,
) -> np.ndarray:
    # compute the sum of all commutators between pairs of matrices

    # first do A with all Bs
    # since [A, B + C] = [A, B] + [A, C], the most efficient way to do this
    # is by summing all the Bs first, then computing the commutator
    B_sum = np.sum(Bs, axis=0)
    commutator_A_Bs = commutator(A, B_sum)

    # then all pairs of Bs
    # specifically, [B_i, B_j] for i < j

    # again we use [A, B + C] = [A, B] + [A, C]
    # sum all B_j for j >= i, using cumsum
    # its okay to include the i=j case, since [B_i, B_i] = 0
    B_sums = np.cumsum(Bs[::~0], axis=0)[::~0]

    # B_sums[i] is the sum of all B_j for j >= i
    def scan_Bs_comms(acc: np.ndarray, i: int) -> np.ndarray:
        """
        Compute the commutator [B_i, B_j] for j >= i and accumulate.
        """
        return acc + commutator(Bs[i], B_sums[i]), ()

    smoothing_matrix, _ = jax.lax.scan(
        scan_Bs_comms,
        commutator_A_Bs,
        np.arange(Bs.shape[0] - 1),  # don't include the last index
    )

    return smoothing_matrix


def reg_pmm_predict_func(
    A: np.ndarray,
    Bs: np.ndarray,
    Ds: np.ndarray,
    gs: np.ndarray,
    r: int,
    X: np.ndarray,
    smoothing: Optional[float] = None,  # must be traced out before JIT
) -> np.ndarray:
    """
    Parameters:
    A : np.ndarray
        Constant matrix (n, n), must be Hermitian.
    Bs : np.ndarray
        Array of linear matrices (p, n, n) where p is the number of features.
        Each Bs[i] must be Hermitian.
    Ds : np.ndarray
        Array of secondary matrices (q, d, n, n), where q is the number of
        outputs and d is the number of secondary matrices.
        Each Ds[i, j] must be Hermitian.
    gs : np.ndarray
        Array of bias terms (q,). Where q is the number of outputs Must be
        real-valued.
    r : int
        Number of eigenvectors to use for the prediction.
    X : np.ndarray
        Input samples (N, p), where N is the number of samples and p is the
        number of features.

    Returns:
    np.ndarray
        Output predictions (N, q), where q is the number of outputs.
        Each row corresponds to the prediction for each output class for a
        sample.
    """

    return jax.vmap(
        reg_pmm_predict_func_single,
        in_axes=(None, None, None, None, None, 0, None),
        out_axes=0,
    )(A, Bs, Ds, gs, r, X, smoothing)


def reg_pmm_predict_func_single(
    A: np.ndarray,
    Bs: np.ndarray,
    Ds: np.ndarray,
    gs: np.ndarray,
    r: int,
    X: np.ndarray,
    smoothing: Optional[float] = None,  # must be traced out before JIT
) -> np.ndarray:
    """
    Predict function for a single instance using PMM regression. The output is
    given by:

    M(X) = A + sum(Bs[i] * X[i]) =(EVD)=> V E V^H
    V = [v_1, ..., v_n]

    z_k = g_k
        + sum_{ij}^r ( sum_l [ |v_i^H D_{kl} v_j|^2 - 0.5 * ||D_{kl}||^2_2 ] )

        = g_k
        - 0.5 * r^2 * sum_l ||D_{kl}||^2_2
        + sum_{ij}^r sum_l |v_i^H D_{kl} v_j|^2

    Parameters:
    A : np.ndarray
        Constant matrix (n, n), must be Hermitian.
    Bs : np.ndarray
        Array of linear matrices (p, n, n) where p is the number of features.
        Each Bs[i] must be Hermitian.
    Ds : np.ndarray
        Array of secondary matrices (q, d, n, n), where q is the number of
        outputs and d is the number of secondary matrices.
        Each Ds[i, j] must be Hermitian.
    gs : np.ndarray
        Array of bias terms (q,). Where q is the number of outputs Must be
        real-valued.
    r : int
        Number of eigenvectors to use for the prediction.
    X : np.ndarray
        Input features (p,). Each X[i] corresponds to the i-th feature.

    Returns:
    np.ndarray
        Output predictions (q,). Each element corresponds to the prediction for
        each output class.
    """
    # ensure Hermitian matrices
    A = (A + A.conj().T) / 2
    Bs = (Bs + Bs.conj().transpose(0, 2, 1)) / 2
    Ds = (Ds + Ds.conj().transpose(0, 1, 3, 2)) / 2
    gs = gs.real

    # handle different smoothing types

    if smoothing is None or smoothing == 0.0:
        M = A + np.einsum("i,ijk->jk", X.astype(Bs.dtype), Bs)
        E, V = np.linalg.eigh(M)

        # get the r eigenvectors with the largest magnitude eigenvalues
        idx = np.argsort(np.abs(E))[::-1][:r]

    else:
        M = A + np.einsum("i,ijk->jk", X.astype(Bs.dtype), Bs)
        C = exact_smoothing_matrix(A, Bs)

        E, V = np.linalg.eigh(M + smoothing * C)

        # when smoothing we have to sort algebraically, not by magnitude
        # can use either end of the spectrum, so we take the largest r
        idx = np.argsort(E)[::-1][:r]

    V = V[:, idx]

    # Z will be (q,) the output for each class
    # first, just compute all the transition amplitudes
    Z = np.einsum("ai,klab,bj->klij", V.conj(), Ds, V)
    # then abs^2 and sum
    Z = np.sum((Z.real**2 + Z.imag**2), axis=(1, 2, 3))

    # compute the operator norm term
    norm_term = np.sum(np.linalg.norm(Ds, axis=(2, 3), ord=2) ** 2, axis=1)

    # compute the final output
    # TODO: why isn't the norm multiplied by r^2?
    #       when it is, nothing ever learns
    Z = gs - 0.5 * norm_term + Z

    return Z


def reg_pmm_predict_func_legacy(
    A: np.ndarray,
    Bs: np.ndarray,
    Ds: np.ndarray,
    gs: np.ndarray,
    X: np.ndarray,
    smoothing: Optional[float] = None,  # must be traced out before JIT
) -> np.ndarray:
    """
    Parameters:
    A : np.ndarray
        Constant matrix (n, n), must be Hermitian.
    Bs : np.ndarray
        Array of linear matrices (p, n, n) where p is the number of features.
        Each Bs[i] must be Hermitian.
    Ds : np.ndarray
        Array of secondary matrices (q, r, r, n, n), where q is the number of
        outputs and d is the number of secondary matrices.
        Each Ds[i, j] must be Hermitian and
        Ds[:, i, j, :, :] = Ds[:, j, i, :, :].
    gs : np.ndarray
        Array of bias terms (q,). Where q is the number of outputs Must be
        real-valued.
    r : int
        Number of eigenvectors to use for the prediction.
    X : np.ndarray
        Input samples (N, p), where N is the number of samples and p is the
        number of features.

    Returns:
    np.ndarray
        Output predictions (N, q), where q is the number of outputs.
        Each row corresponds to the prediction for each output class for a
        sample.
    """

    return jax.vmap(
        reg_pmm_predict_func_single_legacy,
        in_axes=(
            None,
            None,
            None,
            None,
            0,
            None,
        ),
        out_axes=0,
    )(A, Bs, Ds, gs, X, smoothing)


def reg_pmm_predict_func_single_legacy(
    A: np.ndarray,
    Bs: np.ndarray,
    Ds: np.ndarray,
    gs: np.ndarray,
    X: np.ndarray,
    smoothing: Optional[float] = None,  # must be traced out before JIT
) -> np.ndarray:
    """
    Predict function for a single instance using PMM regression. The output is
    given by:

    M(X) = A + sum(Bs[i] * X[i]) =(EVD)=> V E V^H
    V = [v_1, ..., v_n]

    z_k = g_k
        + sum_{ij}^r ( sum_l [ |v_i^H D_{kl} v_j|^2 - 0.5 * ||D_{kl}||^2_2 ] )

        = g_k
        - 0.5 * r^2 * sum_l ||D_{kl}||^2_2
        + sum_{ij}^r sum_l |v_i^H D_{kl} v_j|^2

    Parameters:
    A : np.ndarray
        Constant matrix (n, n), must be Hermitian.
    Bs : np.ndarray
        Array of linear matrices (p, n, n) where p is the number of features.
        Each Bs[i] must be Hermitian.
    Ds : np.ndarray
        Array of secondary matrices (q, r, r, n, n), where q is the number of
        outputs and d is the number of secondary matrices.
        Each Ds[i, j] must be Hermitian and
        Ds[:, i, j, :, :] = Ds[:, j, i, :, :].
    gs : np.ndarray
        Array of bias terms (q,). Where q is the number of outputs Must be
        real-valued.
    r : int
        Number of eigenvectors to use for the prediction.
    X : np.ndarray
        Input features (p,). Each X[i] corresponds to the i-th feature.

    Returns:
    np.ndarray
        Output predictions (q,). Each element corresponds to the prediction for
        each output class.
    """

    # ensure Hermitian matrices
    A = (A + A.conj().T) / 2
    Bs = (Bs + Bs.conj().transpose(0, 2, 1)) / 2
    Ds = (Ds + Ds.conj().transpose(0, 1, 2, 4, 3)) / 2
    Ds = (Ds + Ds.transpose(0, 2, 1, 3, 4)) / 2
    gs = gs.real
    r = Ds.shape[1]  # r is the number of secondary matrices

    # handle different smoothing types

    if smoothing is None or smoothing == 0.0:
        M = A + np.einsum("i,ijk->jk", X.astype(Bs.dtype), Bs)
        E, V = np.linalg.eigh(M)

        # get the r eigenvectors with the largest magnitude eigenvalues
        idx = np.argsort(np.abs(E))[::-1][:r]

    else:
        M = A + np.einsum("i,ijk->jk", X.astype(Bs.dtype), Bs)
        C = exact_smoothing_matrix(A, Bs)

        E, V = np.linalg.eigh(M + smoothing * C)

        # when smoothing we have to sort algebraically, not by magnitude
        # can use either end of the spectrum, so we take the largest r
        idx = np.argsort(E)[::-1][:r]

    V = V[:, idx]

    # Z will be (q,) the output for each class
    # first, just compute all the transition amplitudes
    Z = np.einsum("ai,kijab,bj->kij", V.conj(), Ds, V)
    # then abs^2 and sum
    Z = np.sum((Z.real**2 + Z.imag**2), axis=(1, 2))

    # compute the operator norm term
    norm_term = np.sum(
        np.linalg.norm(Ds, axis=(3, 4), ord=2) ** 2, axis=(1, 2)
    )

    # compute the final output
    # TODO: why isn't the norm multiplied by r^2?
    #       when it is, nothing ever learns
    Z = gs - 0.5 * norm_term + Z

    return Z
