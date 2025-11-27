import pytest

from parametricmatrixmodels import tree_util


def test_make_mutable():
    r"""
    Test make_mutable
    """

    immutable_tree = (
        {"a": 0, "b": (1, 2)},
        [3, 4, {"c": 5}],
    )
    mutable_tree = tree_util.make_mutable(immutable_tree)
    expected_tree = [
        {"a": 0, "b": [1, 2]},
        [3, 4, {"c": 5}],
    ]
    assert mutable_tree == expected_tree


def test_extend_tree():
    r"""
    Test extend_structure_from_strpaths
    """

    base_tree = {
        "a": 0,
        "b": [0],
    }
    strpaths = [
        "b.1",
        "c.d.e.5",
    ]

    extended_tree = tree_util.extend_structure_from_strpaths(
        base_tree, strpaths, separator="."
    )

    expected_tree = {
        "a": 0,
        "b": [0, None],
        "c": {"d": {"e": [None, None, None, None, None, None]}},
    }

    assert extended_tree == expected_tree

    # ensure that tuples are preserved

    base_tree = ({"a": 0}, (1, 2), ({"b": (3, 0)},))
    strpaths = [
        "1.2",
        "1.1",  # should be ignored
        "2.0.c.4",
        "2.0.b.2",
    ]
    fill_values = [0, None, 2, 3]

    expected_tree = (
        {"a": 0},
        (1, 2, 0),
        ({"b": (3, 0, 3), "c": [None, None, None, None, 2]},),
    )

    extended_tree = tree_util.extend_structure_from_strpaths(
        base_tree, strpaths, fill_values=fill_values, separator="."
    )
    assert extended_tree == expected_tree

    # ensure certain errors are raised
    with pytest.raises(ValueError):
        # if we try to overwrite a node
        fill_values_fail = [0, 1, 2, 3]
        extended_tree = tree_util.extend_structure_from_strpaths(
            base_tree, strpaths, fill_values=fill_values_fail, separator="."
        )
    with pytest.raises(KeyError):
        # if we try to index a list/tuple with a string
        strpaths_fail = [
            "a",
        ]
        extended_tree = tree_util.extend_structure_from_strpaths(
            base_tree, strpaths_fail, separator="."
        )
    with pytest.raises(TypeError):
        # if we try to key a non-dict/list/tuple
        strpaths_fail = [
            "0.a.0",
        ]
        extended_tree = tree_util.extend_structure_from_strpaths(
            base_tree, strpaths_fail, separator="."
        )
