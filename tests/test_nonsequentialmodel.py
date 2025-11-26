import pytest

import parametricmatrixmodels as pmm


@pytest.mark.parametrize("execution_number", range(5))
def test_nonsequentialmodel_graph(execution_number: int):
    r"""
    Test NonSequentialModel graph resolution
    """

    C = pmm.modules.Comment

    modules = {"A": C(), "B": (C(), C()), "C": C()}
    connections = {
        "input": ["A", "B.0"],
        "A": ["B.1", "C"],
        "B.0": ["C"],
        "B.1": ["output"],
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
        "B.1",
        "B.0",
        "C",
        "output",
    ], f"Execution order is not deterministic across runs: got {order} "
    f"on test run {execution_number}"
