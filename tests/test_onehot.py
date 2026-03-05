import jax.numpy as np

import parametricmatrixmodels as pmm


def test_onehot(tmp_path):
    r"""
    Test the OneHot encoder with PyTrees
    """

    # array input
    # PyTree input
    pytree_X = {
        "part1": np.array([[1.1, 2.0], [4.0, 5.0], [1.1, 2.0]]),
        "part2": np.array([[3.0], [6.0], [3.0]]),
    }
    encoder = pmm.preprocessing.OneHot()
    pytree_X_encoded = encoder.fit_transform(pytree_X)
    pytree_X_expected = {
        "part1": np.array([[1, 0], [0, 1], [1, 0]]),
        "part2": np.array([[1, 0], [0, 1], [1, 0]]),
    }
    assert pmm.tree_util.all_equal(pytree_X_encoded, pytree_X_expected)

    pytree_X_unencoded = encoder.inverse_transform(pytree_X_encoded)
    assert pmm.tree_util.all_equal(pytree_X_unencoded, pytree_X)

    # test with unseen category
    pytree_X_unseen = {
        "part1": np.array([[1.1, 2.0], [4.0, 5.0], [7.0, 8.0]]),
        "part2": np.array([[3.0], [6.0], [9.0]]),
    }
    pytree_X_unseen_encoded = encoder.transform(pytree_X_unseen)
    pytree_X_unseen_expected = {
        "part1": np.array([[1, 0], [0, 1], [0, 0]]),
        "part2": np.array([[1, 0], [0, 1], [0, 0]]),
    }
    assert pmm.tree_util.all_equal(
        pytree_X_unseen_encoded, pytree_X_unseen_expected
    )

    pytree_X_unseen_unencoded = encoder.inverse_transform(
        pytree_X_unseen_encoded
    )
    # unseen categories should be encoded as zeros, so inverse transform will
    # return all zeros
    pytree_X_unseen_unencoded_expected = {
        "part1": np.array([[1.1, 2.0], [4.0, 5.0], [0.0, 0.0]]),
        "part2": np.array([[3.0], [6.0], [0.0]]),
    }
    assert pmm.tree_util.all_equal(
        pytree_X_unseen_unencoded, pytree_X_unseen_unencoded_expected
    )

    # test saving and loading
    encoder.save(tmp_path / "encoder.npz")
    loaded_encoder = pmm.preprocessing.OneHot.load(tmp_path / "encoder.npz")
    loaded_pytree_X_encoded = loaded_encoder.transform(pytree_X)
    assert pmm.tree_util.all_equal(loaded_pytree_X_encoded, pytree_X_encoded)
