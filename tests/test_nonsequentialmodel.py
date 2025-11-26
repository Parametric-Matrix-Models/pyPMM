import parametricmatrixmodels as pmm


def test_nonsequentialmodel_graph():
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

    valid_orders = [
        ["input", "A", "B.0", "B.1", "C", "output"],
        ["input", "A", "B.1", "B.0", "C", "output"],
        ["input", "A", "B.0", "C", "B.1", "output"],
        ["input", "B.0", "A", "B.1", "C", "output"],
    ]
    assert (
        order in valid_orders
    ), f"Execution order {order} not in valid orders {valid_orders}"
