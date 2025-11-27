import pytest

import parametricmatrixmodels as pmm


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
