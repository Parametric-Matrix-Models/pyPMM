from .reshape import Reshape


class Flatten(Reshape):

    def __init__(self) -> None:
        super().__init__(shape=(-1,))

    def name(self) -> str:
        return "Flatten"
