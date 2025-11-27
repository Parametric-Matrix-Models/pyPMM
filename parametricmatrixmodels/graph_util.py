from __future__ import annotations

import uuid

from .tree_util import getitem_by_strpath
from .typing import Any, Dict, List, OrderedSet, PyTree, Tuple


def concretize_paths(paths: List[str], separator: str = ".") -> List[str]:
    r"""
    Concretize a list of keypaths by appending indices or uuids to duplicate
    paths. Respects existing indices or keys.

    Parameters
    ----------
        paths
            List of keypaths to normalize.
        separator
            The string used to separate levels in the keypaths. Default is ".".
    Returns
    -------
        List of concretized keypaths with duplicates resolved.

    Examples
    --------
        >>> paths = [
        ...     "a.b.0.1",
        ...     "a.b.0",
        ...     "a.b",
        ...     "a.1",
        ...     "a.1",
        ...     "a.2",
        ...     "a",
        ...     "",
        ... ]
        >>> concrete_paths = concretize_paths(paths)
        >>> concrete_paths
        [
            "a.b.0.1",
            "a.b.0.0", # added .0 since a.b.0.* exists
            "a.b.1",   # added .1 since a.b.* exists and .0 is taken
            "a.1.0",   # added .0 since a.1.* exists
            "a.1.1",   # added .1 since a.1.* exists and .0 is taken
            "a.2",     # no change since a.2.* doesn't exist
            "a.<uuid>", # added .<uuid> since a.* exists & non-int keys ('b')
            "<uuid>",  # added <uuid> since * exists & non-int keys ('a')
        ]
    """
    paths = [p.split(separator) if p else [] for p in paths]

    # work from the shortest paths to the longest
    cur_len = min(len(p) for p in paths)
    max_len = max(len(p) for p in paths)

    while cur_len <= max_len:
        # group all paths by their prefix up to cur_len
        # save path index with it to reconstruct later (deterministic order)
        prefix_to_paths: Dict[str, List[Tuple[List[str], int]]] = {}
        for i, p in enumerate(paths):
            prefix = separator.join(p[:cur_len])
            if prefix not in prefix_to_paths:
                prefix_to_paths[prefix] = []
            prefix_to_paths[prefix].append((p[cur_len:], i))

        # if a group contains multiple paths and any are []
        # we need to concretize them
        # otherwise, do nothing
        for prefix, group_paths in prefix_to_paths.items():
            if len(group_paths) <= 1 or [] not in [p for p, _ in group_paths]:
                continue
            # infer existing indices/keys at this level
            existing_keys = set([p[0] for p, _ in group_paths if p])
            is_str = any(not k.isdigit() for k in existing_keys)
            if is_str:
                # use uuid to avoid collisions
                for p, i in group_paths:
                    if p == []:
                        new_key = str(uuid.uuid4().hex)
                        paths[i] = paths[i] + [new_key]
            else:
                # get an iterable of available indices
                available_indices = iter(
                    sorted(
                        list(
                            set(range(len(group_paths)))
                            - set(int(k) for k in existing_keys)
                        )
                    )
                )
                for p, i in group_paths:
                    if p == []:
                        new_index = str(next(available_indices))
                        paths[i] = paths[i] + [new_index]
        cur_len += 1

    paths = [separator.join(p) for p in paths]
    return paths


def concretize_connections(
    connections: Dict[str, List[str]],
    tree: PyTree[Any],
    separator: str = ".",
    in_key: str = "input",
    out_key: str = "output",
) -> Dict[str, List[str]]:
    r"""
    Concretize the connections dictionary by explicitly adding paths for
    leaves that have substructures but are not fully
    specified in the connections. The keys in the connections are never
    modified, since omission of substructures in the keys implies that the
    entire substructure is passed. Only the values are modified to explicitly
    specify all substructures. Order is preserved.

    Returns
    -------
        Concretized connections dictionary with all leave substructures
        explicitly specified.

    Examples
    --------

    All examples below assume a tree like
        >>> tree = {
        ...     "M1": "*",
        ...     "M2": "*",
        ... }

    Multiple implicit connections to the output
        >>> connections = {
        ...     "input": ["M1", "M2"],
        ...     "M1": "output",
        ...     "M2": "output"
        ... }
        >>> concretized_connections = {
        ...     "input": ["M1", "M2"],
        ...     "M1": ["output.0"],
        ...     "M2": ["output.1"]
        ... }

    Multiple implicit connections from the input and between modules
        >>> connections = {
        ...     "input.0": "M1",
        ...     "input.1": "M1",
        ...     "input.2": "M2",
        ...     "M1": "M2",
        ...     "M2": "output"
        ... }
        >>> concretized_connections = {
        ...     "input.0": ["M1.0"],
        ...     "input.1": ["M1.1"],
        ...     "input.2": ["M2.0"],
        ...     "M1": ["M2.1"],
        ...     "M2": ["output"]
        ... }

    Beyond depth-1 implicit connections
        >>> connections = {
        ...     "input": ["M1", "M2"],
        ...     "M1": "output.0",
        ...     "M2": "output.0"
        ... }
        >>> concretized_connections = {
        ...     "input": ["M1", "M2"],
        ...     "M1": ["output.0.0"],
        ...     "M2": ["output.0.1"]
        ... }

    Partially implicit connection to the output
        >>> connections = {
        ...     "input": ["M1", "M2"],
        ...     "M1": "output.1",
        ...     "M2": "output"
        ... }
        >>> concretized_connections = {
        ...     "input": ["M1", "M2"],
        ...     "M1": ["output.1"],
        ...     "M2": ["output.0"]
        ... }

    Partially implicit connections across depths
        >>> connections = {
        ...     "input": ["M1", "M2"],
        ...     "M1": "output.1.a",
        ...     "M2": "output"
        ... }
        >>> concretized_connections = {
        ...     "input": ["M1", "M2"],
        ...     "M1": ["output.1.a"],
        ...     "M2": ["output.0"]
        ... }

    Partially implicit connection with a dictionary type, in which case a
    random UUID is used to avoid collisions
        >>> connections = {
        ...     "input": ["M1", "M2"],
        ...     "M1": "output.a",
        ...     "M2": "output"
        ... }
        >>> concretized_connections = {
        ...     "input": ["M1", "M2"],
        ...     "M1": ["output.a"],
        ...     "M2": ["output.<uuid>"]
        ... }
    """

    # first we need a dictionary of outer leaves to substructures, only for
    # values in the connections
    # we include the original key in order to reconstruct the connections later
    outer_to_inner_w_key: Dict[str, List[Tuple[str, str]]] = {}
    partitioned_connections = partition_connections_by_tree(
        connections,
        tree,
        separator=separator,
        in_key=in_key,
        out_key=out_key,
    )
    double_sep = f"{separator}{separator}"
    for key, values in partitioned_connections.items():
        for v in values:
            outer, inner = v.split(double_sep)
            if outer not in outer_to_inner_w_key:
                outer_to_inner_w_key[outer] = []
            outer_to_inner_w_key[outer].append((inner, key))

    # make a copy without the keys
    outer_to_inner = {
        k: [t[0] for t in v] for k, v in outer_to_inner_w_key.items()
    }

    # handle mixed depths
    outer_to_inner = {
        outer: concretize_paths(inner_paths, separator=separator)
        for outer, inner_paths in outer_to_inner.items()
    }

    # handle duplicates by adding indices
    concretized_outer_to_inner: Dict[str, List[str]] = {}
    for outer, inner_paths in outer_to_inner.items():
        # first, find duplicates
        path_counts: Dict[str, int] = {}
        for p in inner_paths:
            if p not in path_counts:
                path_counts[p] = 0
            path_counts[p] += 1

        # now concretize duplicates by adding indices
        concretized_paths: List[str] = []
        path_indices: Dict[str, int] = {}
        for p in inner_paths:
            if path_counts[p] > 1:
                if p not in path_indices:
                    path_indices[p] = 0
                concretized_p = (
                    str(path_indices[p])
                    if not p
                    else f"{p}{separator}{path_indices[p]}"
                )
                path_indices[p] += 1
                concretized_paths.append(concretized_p)
            else:
                concretized_paths.append(p)
        concretized_outer_to_inner[outer] = concretized_paths

    # readd keys
    concretized_outer_to_inner_w_key: Dict[str, List[Tuple[str, str]]] = {}
    for outer, inner_paths in concretized_outer_to_inner.items():
        original_tuples = outer_to_inner_w_key[outer]
        concretized_outer_to_inner_w_key[outer] = list(
            zip(inner_paths, [t[1] for t in original_tuples])
        )

    # finally, reconstruct the connections dictionary
    concretized_connections: Dict[str, List[str]] = {}
    for outer, inner_tuples in concretized_outer_to_inner_w_key.items():
        for inner, key in inner_tuples:
            new_value = (
                outer + double_sep + inner if inner else outer + double_sep
            )
            if key not in concretized_connections:
                concretized_connections[key] = []
            concretized_connections[key].append(new_value)

    # remove double separators
    # if they're at the begin or end, just remove them,
    # otherwise replace with single separator
    return {
        k.replace(double_sep, separator).strip(separator): [
            v.replace(double_sep, separator).strip(separator) for v in vs
        ]
        for k, vs in concretized_connections.items()
    }


def get_outer_connections_by_tree(
    connections: Dict[str, List[str]],
    tree: PyTree[Any],
    separator: str = ".",
    in_key: str = "input",
    out_key: str = "output",
) -> Dict[str, List[str]]:
    r"""
    Get the connections dictionary showing only the tree structure,
    not including any substructure.

    Returns
    -------
        Connections dictionary with only tree structure.

    Examples
    --------
    Given a tree like
        >>> tree = {
        ...     "block1": {
        ...         "M1": "*"
        ...     },
        ...     "block2": {
        ...         "M2": "*"
        ...     },
        ...     "block3": {
        ...         "M3": "*"
        ...     }
        ... }

    and a connections dictionary like

        >>> connections = {
        ...     "block1.M1.a": ["block2.M2.0", "block3.M3.input"],
        ...     "block1.M1.b": "output",
        ...     "input": "block1.M1"
        ... }

    The outer tree-only connections will be

        >>> outer_connections = {
        ...     "block1.M1": ["block2.M2", "block3.M3"],
        ...     "block1.M1": ["output"],
        ...     "input": ["block1.M1"]
        ... }

    """
    conn_separated = partition_connections_by_tree(
        connections,
        tree,
        separator=separator,
        in_key=in_key,
        out_key=out_key,
    )

    double_sep = f"{separator}{separator}"

    # now we have connections in the form
    # { 'input.<path>..': ['<mod_path>..<io_path>', ...], ... }

    # we need to be careful, since now there can be multiple identical keys
    # we handle this in the values by using a set, and always add to the
    # value instead of overwriting
    # we use OrderedSet to preserve order
    conn_stripped: Dict[str, OrderedSet[str]] = {}
    for key, value in conn_separated.items():
        # special case for 'input' and 'output'
        if key.startswith("input") or key.startswith("output"):
            # remove all IO paths here too
            stripped_key = key.split(separator)[0]
        else:
            stripped_key, _ = key.split(double_sep)
        if stripped_key not in conn_stripped:
            conn_stripped[stripped_key] = OrderedSet()
        for v in value:
            stripped_v, _ = v.split(double_sep)
            conn_stripped[stripped_key].add(stripped_v)

    # convert OrderedSets back to lists
    return {key: list(value) for key, value in conn_stripped.items()}


def partition_connections_by_tree(
    connections: Dict[str, List[str]],
    tree: PyTree[Any],
    separator: str = ".",
    in_key: str = "input",
    out_key: str = "output",
) -> Dict[str, List[str]]:
    r"""
    Process the connections dictionary to separate the tree structure of 'tree'
    from the remaining structure.

    Parameters
    ----------
        connections
            Dictionary defining connections between leaves in the tree and
            their sub-structures. Keys and values are strings representing
            the keypaths to the leaves or sub-structures in the tree.
        tree
            A PyTree representing the outer structure to separate.
        separator
            The string used to separate levels in the keypaths. Default is ".".
        in_key
            The reserved key representing the model input. Default is "input".
        out_key
            The reserved key representing the model output. Default is
            "output".

    Returns
    -------
        Processed connections dictionary with separated structures
        by a double separator (e.g., ".." if the separator is ".").

    Raises
    ------
        ValueError
            If the connections contain invalid keys or values, or if the
            separator appears consecutively in any key or value.

    Examples
    --------
    Given a tree like
        >>> tree = {
        ...     "block1": {
        ...         "M1": "*"
        ...     },
        ...     "block2": {
        ...         "M2": "*"
        ...     },
        ...     "block3": {
        ...         "M3": "*"
        ...     }
        ... }

    and connections dictionary like
        >>> connections = {
        ...     "block1.M1.a": ["block2.M2.0", "block3.M3.input"],
        ...     "block1.M1.b": "output",
        ...     "input": "block1.M1"
        ... }

    The partitioned connections will be
        >>> partitioned_connections = {
        ...     "block1..M1.a": ["block2..M2.0", "block3..M3.input"],
        ...     "block1..M1.b": ["output.."],
        ...     "input..": ["block1..M1"]
        ... }


    """
    # the zeroth step is to convert the connections dictionary mixed value
    # types to uniform list types
    conn: Dict[str, List[str]] = {}
    for key, value in connections.items():
        if isinstance(value, (list, tuple)):
            conn[key] = list(value)
        else:
            conn[key] = [value]

    # first we need to verify that the separator is not already doubled
    # anywhere
    double_sep = f"{separator}{separator}"
    for key, value in conn.items():
        if double_sep in key:
            raise ValueError(
                f"Separator '{separator}' cannot appear "
                "consecutively in connection keys. Found in key '{key}'."
            )
        for v in value:
            if double_sep in v:
                raise ValueError(
                    f"Separator '{separator}' cannot appear "
                    "consecutively in connection values. Found in value "
                    f"'{v}'."
                )

    # now we use getitem_by_strpath with allow_early_return and
    # return_remainder to separate the tree structure from the remaining
    # structure
    conn_separated: Dict[str, List[str]] = {}
    for key, value in conn.items():
        if key.startswith(in_key + separator) or key == in_key:
            new_key = (
                in_key
                + double_sep
                + key.removeprefix(in_key).lstrip(separator)
            )
        elif key.startswith(out_key + separator) or key == out_key:
            raise ValueError(
                f"'{out_key}' cannot be used as a key in the connections "
                "dictionary since it is reserved for model output."
            )
        else:
            _, key_remainder = getitem_by_strpath(
                tree,
                key,
                separator=separator,
                allow_early_return=True,
                return_remainder=True,
            )
            new_key = (
                key.removesuffix(key_remainder).rstrip(separator)
                + double_sep
                + key_remainder
            )
        new_values = []
        for v in value:
            if v.startswith(out_key + separator) or v == out_key:
                new_v = (
                    out_key
                    + double_sep
                    + v.removeprefix(out_key).lstrip(separator)
                )
                new_values.append(new_v)
                continue
            elif v.startswith(in_key + separator) or v == in_key:
                raise ValueError(
                    f"'{in_key}' cannot be used as a value in the "
                    "connections dictionary since it is reserved for "
                    "model input."
                )
            else:
                _, v_remainder = getitem_by_strpath(
                    tree,
                    v,
                    separator=separator,
                    allow_early_return=True,
                    return_remainder=True,
                )
                new_v = (
                    v.removesuffix(v_remainder).removesuffix(separator)
                    + double_sep
                    + v_remainder
                )
                new_values.append(new_v)
        conn_separated[new_key] = new_values

    return conn_separated


def resolve_connections(
    incoming_edges,
    start_key="input",
    end_key="output",
    max_recursion_depth=100,
):

    # this is a special case of a topological sort, since we can ignore
    # all nodes that are not on a path from start_key to end_key,
    # but we need to make sure that all nodes on all such paths are
    # included
    visited: OrderedSet[str] = OrderedSet()
    topological_order: List[str] = []

    def dfs(
        node: str, depth: int = 0, path: OrderedSet[str] | None = None
    ) -> None:
        if depth > max_recursion_depth:
            raise ValueError(
                "Maximum recursion depth exceeded while resolving connections."
                " Possible uncaught cycle in connections?"
            )
        if path is not None and node in path:
            raise ValueError(f"Cycle detected in connections at node '{node}'")
        depth += 1
        if node in visited:
            depth -= 1
            return
        visited.add(node)
        if node in incoming_edges:
            for parent in incoming_edges[node]:
                dfs(
                    parent,
                    depth,
                    path={node} if path is None else path | {node},
                )
        topological_order.append(node)
        depth -= 1

    dfs(end_key)

    try:
        validate_resolution(topological_order, start_key, end_key)
    except RuntimeError as e:
        raise RuntimeError(
            "Failed to resolve connections into a valid topological order."
        ) from e

    return topological_order, visited


def validate_resolution(
    order: List[str], start_key="input", end_key="output"
) -> None:
    # ensure that order[0] is start_key, order[-1] is end_key,
    # and all nodes appear exactly once
    if order[0] != start_key:
        raise RuntimeError(
            f"Topological order does not start with '{start_key}'"
        )
    if order[-1] != end_key:
        raise RuntimeError("Topological order does not end with '{end_key}'")
    if len(order) != len(set(order)):
        raise RuntimeError("Topological order contains duplicate nodes")
