import jax.numpy as np
from jax import jit, vmap


def reg_pmm_predict_func(
    A: np.ndarray,
    Bs: np.ndarray,
    Ds: np.ndarray,
    gs: np.ndarray,
    r: int,
    X: np.ndarray,
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

    return vmap(
        reg_pmm_predict_func_single,
        in_axes=(None, None, None, None, None, 0),
        out_axes=0,
    )(A, Bs, Ds, gs, r, X)


def reg_pmm_predict_func_single(
    A: np.ndarray,
    Bs: np.ndarray,
    Ds: np.ndarray,
    gs: np.ndarray,
    r: int,
    X: np.ndarray,
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

    M = A + np.einsum("i,ijk->jk", X, Bs)

    E, V = np.linalg.eigh(M)

    # get the r eigenvectors with the largest magnitude eigenvalues
    idx = np.argsort(np.abs(E))[::-1][:r]
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
