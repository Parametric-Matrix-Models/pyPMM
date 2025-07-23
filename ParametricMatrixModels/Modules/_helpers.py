def subsets_to_string(subset_tuple):
    """
    Convert a tuple of slice objects to __getitem__ notation string.

    Args:
        subset_tuple: A tuple containing slice objects or integer arrays

    Returns:
        str: String representation in [start:stop:step] format
    """

    def format_slice(s):
        if isinstance(s, slice):
            # Handle start
            start = "" if s.start is None else str(s.start)
            # Handle stop
            stop = "" if s.stop is None else str(s.stop)
            # Handle step
            if s.step is None:
                return f"{start}:{stop}"
            else:
                return f"{start}:{stop}:{s.step}"
        else:
            # Handle integer array slices
            return str(s)

    # Handle single slice or tuple of slices
    if isinstance(subset_tuple, slice):
        return f"[{format_slice(subset_tuple)}]"
    elif isinstance(subset_tuple, tuple):
        parts = [format_slice(s) for s in subset_tuple]
        return f"[{', '.join(parts)}]"
    else:
        return f"[{format_slice(subset_tuple)}]"


if __name__ == "__main__":
    # Test cases
    test_cases = [
        (slice(2, None), slice(2, None, 2)),
        slice(1, 5),  # Single slice
        (slice(None, 10), slice(0, None, 3)),  # Multiple slices
        (5, slice(2, 8)),  # Mixed integer and slice
        slice(None, None, -1),  # Reverse slice
        slice(None),  # Full slice
        (slice(1, None), 3, slice(None, 5)),  # Multiple mixed indices
    ]

    for case in test_cases:
        result = slice_to_string(case)
        print(f"{case} -> {result}")
