import jax
import jax.numpy as np

import parametricmatrixmodels as pmm


def test_funcbase():

    class AddOverTree(pmm.modules.FuncBase):
        def get_hyperparameters(self):
            return {}

        def f(self, data):
            return jax.tree.reduce(lambda x, y: x + y, data)

    add_over_tree = AddOverTree()
    data = {
        "a": np.array([[1.0, 2.0], [2.0, 3.0]]),
        "b": [
            np.array([[3.0, 4.0], [4.0, 5.0]]),
            np.array([[5.0, 6.0], [6.0, 7.0]]),
        ],
    }
    result, _ = add_over_tree(data)
    expected = np.array([[9, 12], [12, 15]])
    assert np.all(result == expected), f"Expected {expected}, got {result}"

    class Expm(pmm.modules.FuncBase):
        def get_hyperparameters(self):
            return {}

        def f(self, data):
            return jax.tree.map(jax.scipy.linalg.expm, data)

    expm = Expm()
    data = {
        "a": np.array([[[0.0, 1.0], [0.0, 0.0]]]),
        "b": [
            np.array([[[0.0, 2.0], [0.0, 0.0]]]),
            np.array([[[0.0, 3.0], [0.0, 0.0]]]),
        ],
    }
    result, _ = expm(data)
    expected = jax.tree.map(
        jax.scipy.linalg.expm,
        data,
    )

    result_struct = jax.tree.structure(result)
    expected_struct = jax.tree.structure(expected)
    assert (
        result_struct == expected_struct
    ), f"Expected structure {expected_struct}, got {result_struct}"

    def assert_close(a, b):
        assert np.allclose(a, b), f"Expected {b}, got {a}"

    jax.tree.map(assert_close, result, expected)
