import jax.numpy as np
from jax import vmap


def affine_pmm_predict_func(
    A: np.ndarray, Bs: np.ndarray, cs: np.ndarray, k: int, which: str
) -> np.ndarray:
    """
    Ground state eigenvalue PMM model. Output is the selected energy
    eigenvalues of the Hamiltonian:

    H(cs) = A + sum_i cs[i] * Bs[i]

    Parameters
    ----------

    A : np.ndarray
        Constant matrix (n, n), must be Hermitian.
    Bs : np.ndarray
        Array of linear matrices (p, n, n), must be Hermitian.
    cs : np.ndarray
        Array of coefficients (N_samples, p). Must be 2D
    k : int
        Number of eigenvalues to compute.
    which : str
        Which eigenvalues to compute. Options are:
        - 'SA' for smallest algebraic
        - 'LA' for largest algebraic
        - 'SM' for smallest magnitude
        - 'LM' for largest magnitude
        - 'EA' for exterior algebraically
        - 'EM' for exterior by magnitude
        - 'IA' for interior algebraically
        - 'IM' for interior by magnitude

        For algebraic 'which' options, the eigenvalues are returned in
        ascending algebraic order.

        For magnitude 'which' options, the eigenvalues are returned in
        ascending magnitude order.

    Returns
    -------
    Es : np.ndarray
        Array of selected energies (N_samples, k).
    """

    return vmap(
        affine_pmm_predict_func_single, in_axes=(None, None, 0, None, None)
    )(A, Bs, cs, k, which)


def select_eigenvalues(E: np.ndarray, k: int, which: str) -> np.ndarray:
    """
    Selects the k eigenvalues from the array E according to the specified
    'which' option.

    Parameters
    ----------
    E : np.ndarray
        Array of eigenvalues (n,).
    k : int
        Number of eigenvalues to select.
    which : str
        Which eigenvalues to select. Options are:
        - 'SA' for smallest algebraic
        - 'SM' for smallest magnitude
        - 'LA' for largest algebraic
        - 'LM' for largest magnitude
        - 'EA' for exterior algebraically
        - 'EM' for exterior by magnitude
        - 'IA' for interior algebraically
        - 'IM' for interior by magnitude

    Returns
    -------
    selected_E : np.ndarray
        Selected eigenvalues (k,).
    """

    smallest = which.lower().startswith("s")
    largest = which.lower().startswith("l")
    exterior = which.lower().startswith("e")
    interior = which.lower().startswith("i")
    algebraic = which.lower().endswith("a")
    magnitude = which.lower().endswith("m")

    if not (smallest or largest or exterior or interior) or not (
        algebraic or magnitude
    ):
        raise ValueError(
            "Invalid 'which' option. Must start with 'S', 'L', 'E', or 'I' "
            "and end with 'A' or 'M'."
        )

    # sort
    if algebraic:
        idx = np.argsort(E)
    elif magnitude:
        idx = np.argsort(np.abs(E))
    E = E[idx]

    # select
    if smallest:
        return E[:k]
    elif largest:
        return E[-k:]
    elif exterior:
        k_half = k // 2
        k_rem = k - k_half  # to handle odd k
        return np.concatenate((E[:k_half], E[-k_rem:]))
    elif interior:
        n = E.shape[0]
        n_half = n // 2
        k_half = k // 2
        k_rem = k - k_half
        return np.concatenate(
            (E[n_half - k_half : n_half], E[n_half : n_half + k_rem])
        )
    else:
        raise ValueError("Invalid and impossible state reached. Congrats.")


def affine_pmm_predict_func_single(
    A: np.ndarray, Bs: np.ndarray, cs: np.ndarray, k: int, which: str
) -> np.ndarray:
    """
    Eigenvalue PMM model. Output is the selected energy eigenvalues of the
    Hamiltonian:

    H(cs) = A + sum_i cs[i] * Bs[i]

    Parameters
    ----------

    A : np.ndarray
        Constant matrix (n, n), must be Hermitian.
    Bs : np.ndarray
        Array of linear matrices (p, n, n), must be Hermitian.
    cs : np.ndarray
        Array of coefficients (p,). Must be 1D
    k : int
        Number of eigenvalues to compute.
    which : str
        Which eigenvalues to compute. Options are:
        - 'SA' for smallest algebraic
        - 'LA' for largest algebraic
        - 'SM' for smallest magnitude
        - 'LM' for largest magnitude
        - 'EA' for exterior algebraically
        - 'EM' for exterior by magnitude
        - 'IA' for interior algebraically
        - 'IM' for interior by magnitude

        Returns the k selected eigenvalues of the Hamiltonian in
        ascending order according to the specified 'which' option.

    Returns
    -------
    E : np.ndarray
        Selected energies (k,).
    """

    H = A + np.einsum("i,ijk->jk", cs, Bs)
    # Compute the eigenvalues of H
    E = np.linalg.eigvalsh(H)

    return select_eigenvalues(E, k, which)
