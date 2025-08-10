from __future__ import annotations

from .reshape import Reshape


class Flatten(Reshape):
    """
    Module that flattens the input to 1D. Ignores the batch dimension.
    """

    def __init__(self) -> None:
        super().__init__(shape=(-1,))

    def name(self) -> str:
        return "Flatten"
