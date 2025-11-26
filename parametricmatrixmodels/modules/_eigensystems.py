from __future__ import annotations

import jax
import jax.numpy as np

from .tree_util import is_shape_leaf
from .typing import ArrayDataShape, Callable, DataShape


def validate_eigensystem_input_shape(
    input_shape: DataShape, num_eig: int | None
) -> None:
    def is_valid_shape(shape: ArrayDataShape) -> bool:
        return (
            len(shape) == 2
            and shape[0] == shape[1]
            and (num_eig is None or shape[0] >= num_eig)
        )

    if not jax.tree.all(
        jax.tree.map(is_valid_shape, input_shape, is_leaf=is_shape_leaf)
    ):
        raise ValueError(
            "Input shape must be (..., N, N) with N >= num_eig. Got: "
            f"{input_shape} for num_eig={num_eig}"
        )


def select_eigenpairs_by_fn(
    E: np.ndarray,
    V: np.ndarray | None,
    k: int | None,
    sort_fn: Callable[[np.ndarray, np.ndarray | None], int],
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Selects the k eigenpairs from the arrays E and V according to the specified
    'sort_fn' option.

    Parameters
    ----------
    E
        Array of eigenvalues (n,).
    V
        Array of eigenvectors (n, n) or None.
    k
        Number of eigenpairs to select. If None, selects all.
    sort_fn
        Function that takes in (E, V) and returns the indices that sort E and
        V.

    Returns
    -------
    selected_E
        Selected eigenvalues (k,).
    selected_V
        Selected eigenvectors (k, n) or None.
    """

    k = k if k is not None else E.shape[0]

    idx = sort_fn(E, V)
    E = E[idx]
    if V is not None:
        V = V[:, idx]
        return E[:k], V[:, :k]
    else:
        return E[:k], None


def select_eigenpairs_by_eigenvalue(
    E: np.ndarray, V: np.ndarray | None, k: int | None, which: str
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Selects the k eigenpairs from the arrays E and V according to the specified
    'which' option.

    Parameters
    ----------
    E
        Array of eigenvalues (n,).
    V
        Array of eigenvectors (n, n) or None.
    k
        Number of eigenvalues to select. If None, selects all.
    which
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
    selected_E
        Selected eigenvalues (k,).
    selected_V
        Selected eigenvectors (k, n) or None.
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

    k = k if k is not None else E.shape[0]

    # make sort_fn, first by algebraic or magnitude, then by selection
    if algebraic:
        am_sort_fn = lambda e: np.argsort(e)  # noqa: E731
    elif magnitude:
        am_sort_fn = lambda e: np.argsort(np.abs(e))  # noqa: E731

    # now select by taking subset of sorted indices
    if smallest:
        idx_fn = lambda e: am_sort_fn(e)[:k]  # noqa: E731
    elif largest:
        idx_fn = lambda e: am_sort_fn(e)[-k:]  # noqa: E731
    elif exterior:
        k_half = k // 2
        k_rem = k - k_half

        def idx_fn(e):
            sorted_idx = am_sort_fn(e)
            return np.concatenate((sorted_idx[:k_half], sorted_idx[-k_rem:]))

    elif interior:
        n = E.shape[0]
        n_half = n // 2
        k_half = k // 2
        k_rem = k - k_half

        def idx_fn(e):
            sorted_idx = am_sort_fn(e)
            return np.concatenate(
                (
                    sorted_idx[n_half - k_half : n_half],
                    sorted_idx[n_half : n_half + k_rem],
                )
            )

    else:
        raise ValueError("Invalid and impossible state reached. Congrats.")

    # sort_fn
    def sort_fn(e: np.ndarray, v: np.ndarray | None) -> np.ndarray:
        return idx_fn(e)

    return select_eigenpairs_by_fn(E, V, k, sort_fn)


def select_eigenvalues(E: np.ndarray, k: int | None, which: str) -> np.ndarray:
    """
    Selects the k eigenvalues from the array E according to the specified
    'which' option.

    Parameters
    ----------
    E
        Array of eigenvalues (n,).
    k
        Number of eigenvalues to select. If None, selects all.
    which
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
    selected_E
        Selected eigenvalues (k,).
    """

    selected_E, _ = select_eigenpairs_by_eigenvalue(E, None, k, which)
    return selected_E
