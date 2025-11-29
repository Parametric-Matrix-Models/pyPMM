import jax
import jax.numpy as np

import parametricmatrixmodels as pmm


def test_concatenate_leaves():
    r"""
    Test the ConcatenateLeaves module
    """

    data = {
        "x": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "y": [
            np.array([[5.0], [6.0]]),
        ],
    }

    # add batch dimension
    data = jax.tree.map(lambda x: x[None, ...], data)

    cl = pmm.modules.ConcatenateLeaves()
    out, _ = cl(data)

    # if axis is None, arrays are flattened first
    expected_out = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])[None, :]
    assert np.allclose(out, expected_out)

    # change order from default leaf order
    cl = pmm.modules.ConcatenateLeaves(path_order=["y.0", "x"])
    out, _ = cl(data)
    expected_out = np.array([5.0, 6.0, 1.0, 2.0, 3.0, 4.0])[None, :]

    assert np.allclose(out, expected_out)

    # specify axis
    cl = pmm.modules.ConcatenateLeaves(axis=1)
    out, _ = cl(data)
    expected_out = np.array([[1.0, 2.0, 5.0], [3.0, 4.0, 6.0]])[None, :, :]
    assert np.allclose(out, expected_out)
