import jax

from parametricmatrixmodels import graph_util


def test_outer_connections():
    r"""
    Test graph_util.get_outer_connections_by_tree
    """

    parent_tree = (
        {"a": "*", "b": ("*", "*")},
        [
            ("*", "*", "*"),
            "*",
        ],
    )

    connections = {
        "input.0": ["0.a.0.0", "0.a.0.1"],
        "0.a.0": ["0.b.0.0.0", "0.b.0.0.1", "0.b.1"],
        "0.b": ["output.0"],
        "input.1.0": ["1.0.0", "1.0.1", "1.0.2"],
        "input.1.1": ["1.1"],
        "1.1.0": ["output.1"],
    }

    expected_outer_connections = {
        "input": ["0.a", "1.0.0", "1.0.1", "1.0.2", "1.1"],
        "0.a": ["0.b.0", "0.b.1"],
        "0.b": ["output"],
        "1.1": ["output"],
    }

    outer_connections = graph_util.get_outer_connections_by_tree(
        connections,
        parent_tree,
        separator=".",
        in_key="input",
        out_key="output",
    )

    assert (
        outer_connections == expected_outer_connections
    ), f"Expected {expected_outer_connections}, but got {outer_connections}"


def test_partition_by_tree():
    r"""
    Test graph_util.partition_connections_by_tree
    """
    parent_tree = (
        {"a": "*", "b": ("*", "*")},
        [
            ("*", "*", "*"),
            "*",
        ],
    )

    connections = {
        "input.0": ["0.a.0.0", "0.a.0.1"],
        "0.a.0": ["0.b.0.0.0", "0.b.0.0.1", "0.b.1"],
        "0.b": ["output.0"],
        "input.1.0": ["1.0.0", "1.0.1", "1.0.2"],
        "input.1.1": ["1.1"],
        "1.1.0": ["output.1"],
    }

    expected = {
        "input..0": ["0.a..0.0", "0.a..0.1"],
        "0.a..0": ["0.b.0..0.0", "0.b.0..0.1", "0.b.1.."],
        "0.b..": ["output..0"],
        "input..1.0": ["1.0.0..", "1.0.1..", "1.0.2.."],
        "input..1.1": ["1.1.."],
        "1.1..0": ["output..1"],
    }

    partitioned_connections = graph_util.partition_connections_by_tree(
        connections,
        parent_tree,
        separator=".",
        in_key="input",
        out_key="output",
    )
    assert (
        partitioned_connections == expected
    ), f"Expected {expected}, but got {partitioned_connections}"


def compare_paths(path1, path2) -> bool:
    if isinstance(path1, str):
        path1 = path1.split(".")
    if isinstance(path2, str):
        path2 = path2.split(".")
    if len(path1) != len(path2):
        return False
    for p1, p2 in zip(path1, path2):
        if p1 == "<uuid>" or p2 == "<uuid>":
            continue
        if p1 != p2:
            return False
    return True


def test_concretize_paths():
    r"""
    Test graph_util.concretize_paths
    """
    paths = [
        "a.b.0.1",
        "a.b.0",
        "a.b",
        "a.b.1",
        "a.1",
        "a.1",
        "a.1",
        "a.1",
        "a",
        "",
    ]
    expected = [
        "a.b.0.1",
        "a.b.0.0",
        "a.b.2",
        "a.b.1",
        "a.1.0",
        "a.1.1",
        "a.1.2",
        "a.1.3",
        "a.<uuid>",
        "<uuid>",
    ]

    concrete_paths = graph_util.concretize_paths(
        paths,
        separator=".",
    )

    # check manually to handle uuid
    expected_l = [p.split(".") for p in expected]
    concrete_paths_l = [p.split(".") for p in concrete_paths]

    for path, conc_path in zip(expected_l, concrete_paths_l):
        assert compare_paths(
            path, conc_path
        ), f"Expected {path}, but got {conc_path}"

    # test that calling again gives same result (idempotency)
    concrete_paths_2 = graph_util.concretize_paths(
        concrete_paths,
        separator=".",
    )

    concrete_paths_2_l = [p.split(".") for p in concrete_paths_2]

    for path1, path2 in zip(concrete_paths_l, concrete_paths_2_l):
        assert path1 == path2, f"Expected {path1}, but got {path2}"


def test_concretize_connections():
    r"""
    Test graph_util.concretize_connections
    """

    tree = {"M1": "*", "M2": "*"}
    connections = {"input": ["M1", "M2"], "M1": "output", "M2": "output"}
    concretized_connections = {
        "input": ["M1", "M2"],
        "M1": ["output.0"],
        "M2": ["output.1"],
    }

    result = graph_util.concretize_connections(
        connections,
        tree,
        separator=".",
        in_key="input",
        out_key="output",
    )

    assert (
        result == concretized_connections
    ), f"Expected {concretized_connections}, but got {result}"

    connections = {
        "input.0": "M1",
        "input.1": "M1",
        "input.2": "M2",
        "M1": "M2",
        "M2": "output",
    }
    concretized_connections = {
        "input.0": ["M1.0"],
        "input.1": ["M1.1"],
        "input.2": ["M2.0"],
        "M1": ["M2.1"],
        "M2": ["output"],
    }
    result = graph_util.concretize_connections(
        connections,
        tree,
        separator=".",
        in_key="input",
        out_key="output",
    )
    assert (
        result == concretized_connections
    ), f"Expected {concretized_connections}, but got {result}"

    connections = {"input": ["M1", "M2"], "M1": "output.0", "M2": "output.0"}
    concretized_connections = {
        "input": ["M1", "M2"],
        "M1": ["output.0.0"],
        "M2": ["output.0.1"],
    }
    result = graph_util.concretize_connections(
        connections,
        tree,
        separator=".",
        in_key="input",
        out_key="output",
    )
    assert (
        result == concretized_connections
    ), f"Expected {concretized_connections}, but got {result}"

    connections = {"input": ["M1", "M2"], "M1": "output.1", "M2": "output"}
    concretized_connections = {
        "input": ["M1", "M2"],
        "M1": ["output.1"],
        "M2": ["output.0"],
    }
    result = graph_util.concretize_connections(
        connections,
        tree,
        separator=".",
        in_key="input",
        out_key="output",
    )
    assert (
        result == concretized_connections
    ), f"Expected {concretized_connections}, but got {result}"

    connections = {"input": ["M1", "M2"], "M1": "output.1.a", "M2": "output"}
    concretized_connections = {
        "input": ["M1", "M2"],
        "M1": ["output.1.a"],
        "M2": ["output.0"],
    }
    result = graph_util.concretize_connections(
        connections,
        tree,
        separator=".",
        in_key="input",
        out_key="output",
    )
    assert (
        result == concretized_connections
    ), f"Expected {concretized_connections}, but got {result}"

    connections = {"input": ["M1", "M2"], "M1": "output.a", "M2": "output"}
    concretized_connections = {
        "input": ["M1", "M2"],
        "M1": ["output.a"],
        "M2": ["output.<uuid>"],
    }
    result = graph_util.concretize_connections(
        connections,
        tree,
        separator=".",
        in_key="input",
        out_key="output",
    )
    comparisons = jax.tree.map(compare_paths, concretized_connections, result)
    assert jax.tree.all(
        comparisons
    ), f"Expected {concretized_connections}, but got {result}"

    # test idempotency with the final test only
    result_2 = graph_util.concretize_connections(
        result,
        tree,
        separator=".",
        in_key="input",
        out_key="output",
    )
    comparisons = jax.tree.map(compare_paths, result, result_2)
    assert jax.tree.all(comparisons), f"Expected {result}, but got {result_2}"
