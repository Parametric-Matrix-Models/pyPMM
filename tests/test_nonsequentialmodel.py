import jax
import jax.numpy as np
import pytest

import parametricmatrixmodels as pmm

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("execution_number", range(5))
def test_nonsequentialmodel_graph(execution_number: int):
    r"""
    Test NonSequentialModel graph resolution
    """

    C = pmm.modules.Comment

    modules = {"A": C(), "B": (C(), C()), "C": C(), "D": C(), "E": C()}
    connections = {
        "input.0": ["A", "B.0.0"],
        "A": ["B.1", "C"],
        "B.0": ["C"],
        "B.1": ["output"],
        "D": "E",  # disconnected subgraph, will be ignored
        "C": ["output"],
    }

    model = pmm.NonSequentialModel(
        modules=modules,
        connections=connections,
    )

    order = model.get_execution_order()

    # there are many possible valid orders

    valid_orders = [
        ["input", "A", "B.0", "B.1", "C", "output"],
        ["input", "A", "B.1", "B.0", "C", "output"],
        ["input", "A", "B.0", "C", "B.1", "output"],
        ["input", "B.0", "A", "B.1", "C", "output"],
        ["input", "B.0", "A", "C", "B.1", "output"],
    ]
    assert (
        order in valid_orders
    ), f"Execution order {order} not in valid orders {valid_orders}"

    # but the execution order should be deterministic across runs and platforms

    assert order == [
        "input",
        "A",
        "B.0",
        "B.1",
        "C",
        "output",
    ], f"Execution order is not deterministic across runs: got {order} "
    f"on test run {execution_number}"


def test_nonsequentialmodel_output_shape():
    r"""
    Test NonSequentialModel output shape resolution
    """
    C = pmm.modules.Comment

    modules = {"A": C(), "B": (C(), C()), "C": C(), "D": C(), "E": C()}
    connections = {
        "input.0": ["A", "B.0.0"],
        "A": ["B.1", "C"],
        "B.0": ["C"],
        "B.1": ["output"],
        "D": "E",  # disconnected subgraph, will be ignored
        "C": ["output"],
    }

    model = pmm.NonSequentialModel(
        modules=modules,
        connections=connections,
    )

    output_shape = model.get_output_shape(((10, 5), (10, 3, 3, 3)))

    assert output_shape == [
        (10, 5),
        [(10, 5), [(10, 5)]],
    ], f"Output shape resolution failed, got {output_shape}"

    connections = {
        "input": ["A", "B.0"],
        "A": "C",
        "C": "output",
    }
    model = pmm.NonSequentialModel(modules=modules, connections=connections)
    output_shape = model.get_output_shape((10, 5))
    assert output_shape == (
        10,
        5,
    ), f"Output shape resolution failed, got {output_shape}"

    R = pmm.modules.Reshape
    F = pmm.modules.Flatten

    modules = (
        {"R1": R((2, 3)), "F1": F()},
        {"R2": R([(3, 2), (1, 1, 1, 6)]), "F2": F()},
    )
    connections = {
        "input": "0.R1",
        "0.R1": ["0.F1", "0.F1"],
        "0.F1": "1.R2",
        "1.R2": "1.F2",
        "1.F2": "output",
    }
    model = pmm.NonSequentialModel(modules=modules, connections=connections)
    output_shape = model.get_output_shape((6,))
    assert output_shape == [
        (6,),
        (6,),
    ], f"Output shape resolution failed, got {output_shape}"

    modules = {"F": F()}
    connections = {
        "input": "F",
        "F": "output",
    }
    model = pmm.NonSequentialModel(modules=modules, connections=connections)
    output_shape = model.get_output_shape((10, 11))
    assert output_shape == (
        10 * 11,
    ), f"Output shape resolution failed, got {output_shape}"


def test_nonsequentialmodel_invalid_connections():
    r"""
    Test NonSequentialModel invalid connections handling
    """
    C = pmm.modules.Comment
    modules = {"A": C(), "B": (C(), C()), "C": C()}

    # cyclic connection
    connections = {
        "input": "A",
        "A": "B.0",
        "B.0": "C",
        "C": "A",  # cycle here
        "B.1": "output",
    }

    with pytest.raises(RuntimeError):
        m = pmm.NonSequentialModel(modules=modules, connections=connections)
        m.get_execution_order()

    # no valid path from input to output
    connections = {
        "input": "A",
        "A": "B.0",
        "B.0": "C",
        # missing connection to output
    }
    with pytest.raises(RuntimeError):
        m = pmm.NonSequentialModel(modules=modules, connections=connections)
        m.get_execution_order()

    # connection to non-existing module
    connections = {
        "input": "A",
        "A": "B.0",
        "B.0": "C",
        "C": "D",  # D does not exist
        "B.1": "output",
    }
    with pytest.raises(KeyError):
        m = pmm.NonSequentialModel(modules=modules, connections=connections)
        m.get_execution_order()


def test_nonsequentialmodel():
    r"""
    Test NonSequentialModel compilation and execution
    """
    C = pmm.modules.Comment

    key = jax.random.key(0)

    modules = {"A": C(), "B": (C(), C()), "C": C(), "D": C(), "E": C()}
    connections = {
        "input.0": ["A", "B.0.0"],
        "A": ["B.1", "C"],
        "B.0": ["C"],
        "B.1": ["output"],
        "D": "E",  # disconnected subgraph, will be ignored
        "C": ["output"],
    }

    model = pmm.NonSequentialModel(
        modules=modules,
        connections=connections,
    )

    batch_dim = 10

    in_data = (np.ones((batch_dim, 5, 5)), np.ones((batch_dim, 3, 3, 3)))

    model.compile(
        None, jax.tree.map(lambda x: x.shape[1:], in_data)
    )  # remove batch dim

    out = model(in_data)

    assert isinstance(out, list), "Output should be a list"
    assert jax.tree.structure(["*", ["*", ["*"]]]) == jax.tree.structure(
        out
    ), f"Output structure mismatch, got {jax.tree.structure(out)}"
    assert np.allclose(out[0], in_data[0]), "Output[0] mismatch"
    assert np.allclose(out[1][0], in_data[0]), "Output[1][0] mismatch"
    assert np.allclose(out[1][1][0], in_data[0]), "Output[1][1][0] mismatch"

    key = jax.random.key(0)

    modules = {
        "L1": pmm.modules.LinearNN(out_features=8, bias=True),
        "L2": [
            pmm.modules.LinearNN(out_features=4, bias=True),
            pmm.modules.LinearNN(out_features=4, bias=True),
        ],
        "L3": pmm.modules.LinearNN(out_features=2, bias=True),
    }
    connections = {
        "input": "L1",
        "L1": ["L2.0", "L2.1"],
        "L2.0": "L3",
        "L3": "output",
    }

    model = pmm.NonSequentialModel(
        modules=modules,
        connections=connections,
    )

    batch_dim = 10

    in_data = jax.random.normal(key, (batch_dim, 6))

    _, output_shape_prog, output = model._get_shape_progression((6,))

    model.compile(
        key, jax.tree.map(lambda x: x.shape[1:], in_data), verbose=True
    )  # remove batch dim

    out = model(in_data)

    assert isinstance(out, np.ndarray), "Output should be an ndarray"
    assert out.shape == (
        batch_dim,
        2,
    ), f"Output shape mismatch, got {out.shape}"
