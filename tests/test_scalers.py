import jax.numpy as np

import parametricmatrixmodels as pmm


def test_minmax_scaler(tmp_path):
    r"""
    Test the MinMaxScaler with both arrays and PyTrees
    """

    # array input
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    scaler = pmm.preprocessing.MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)
    X_expected = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]])
    assert np.allclose(X_scaled, X_expected)

    # with feature-dependent ranges
    scaler = pmm.preprocessing.MinMaxScaler(
        feature_range=(np.array([0, 1, 2]), np.array([1, 2, 3]))
    )
    X_scaled = scaler.fit_transform(X)
    X_expected = np.array([[0.0, 1.0, 2.0], [0.5, 1.5, 2.5], [1.0, 2.0, 3.0]])
    assert np.allclose(X_scaled, X_expected)

    # PyTree input
    pytree_X = {
        "part1": np.array([[1.0, 2.0], [4.0, 5.0], [7.0, 8.0]]),
        "part2": np.array([[3.0], [6.0], [9.0]]),
    }
    scaler = pmm.preprocessing.MinMaxScaler(feature_range=(0, 1))
    pytree_X_scaled = scaler.fit_transform(pytree_X)
    pytree_X_expected = {
        "part1": np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]),
        "part2": np.array([[0.0], [0.5], [1.0]]),
    }
    if not pmm.tree_util.all_equal(pytree_X_scaled, pytree_X_expected):
        raise AssertionError("PyTree MinMaxScaler test failed.")

    # PyTree with feature-dependent ranges
    feature_ranges = (
        {"part1": np.array([0, 1]), "part2": np.array([2])},
        {"part1": np.array([1, 2]), "part2": np.array([3])},
    )
    scaler = pmm.preprocessing.MinMaxScaler(feature_range=feature_ranges)
    pytree_X_scaled = scaler.fit_transform(pytree_X)
    pytree_X_expected = {
        "part1": np.array([[0.0, 1.0], [0.5, 1.5], [1.0, 2.0]]),
        "part2": np.array([[2.0], [2.5], [3.0]]),
    }
    if not pmm.tree_util.all_equal(pytree_X_scaled, pytree_X_expected):
        raise AssertionError(
            "PyTree MinMaxScaler with feature-dependent ranges test failed."
        )

    # Test saving and loading
    scaler_path = tmp_path / "minmax_scaler.npz"
    scaler.save(scaler_path)
    loaded_scaler = pmm.preprocessing.MinMaxScaler.load(scaler_path)
    X_scaled_loaded = loaded_scaler.transform(pytree_X)
    if not pmm.tree_util.all_equal(X_scaled_loaded, pytree_X_scaled):
        raise AssertionError("MinMaxScaler save/load test failed.")


def test_standard_scaler(tmp_path):
    r"""
    Test the StandardScaler with both arrays and PyTrees
    """

    # array input
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    scaler = pmm.preprocessing.StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_expected = np.array(
        [
            [-1.22474487, -1.22474487, -1.22474487],
            [0.0, 0.0, 0.0],
            [1.22474487, 1.22474487, 1.22474487],
        ]
    )
    assert np.allclose(X_scaled, X_expected)

    # PyTree input
    pytree_X = {
        "part1": np.array([[1.0, 2.0], [4.0, 5.0], [7.0, 8.0]]),
        "part2": np.array([[3.0], [6.0], [9.0]]),
    }
    scaler = pmm.preprocessing.StandardScaler()
    pytree_X_scaled = scaler.fit_transform(pytree_X)
    pytree_X_expected = {
        "part1": np.array(
            [[-1.22474487, -1.22474487], [0.0, 0.0], [1.22474487, 1.22474487]]
        ),
        "part2": np.array([[-1.22474487], [0.0], [1.22474487]]),
    }
    if not pmm.tree_util.all_close(pytree_X_scaled, pytree_X_expected):
        raise AssertionError("PyTree StandardScaler test failed.")

    # Test saving and loading
    scaler_path = tmp_path / "standard_scaler.npz"
    scaler.save(scaler_path)
    loaded_scaler = pmm.preprocessing.StandardScaler.load(scaler_path)
    X_scaled_loaded = loaded_scaler.transform(pytree_X)
    if not pmm.tree_util.all_close(X_scaled_loaded, pytree_X_scaled):
        raise AssertionError("StandardScaler save/load test failed.")
