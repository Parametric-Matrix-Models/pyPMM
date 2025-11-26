from .typing import List, Set


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
    visited: Set[str] = set()
    topological_order: List[str] = []

    def dfs(node: str, depth: int = 0, path: Set[str] | None = None) -> None:
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
