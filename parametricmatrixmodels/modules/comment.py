from parametricmatrixmodels.typing import (
    Any,
    DataShape,
    HyperParams,
    ModuleCallable,
    Params,
    State,
)

from .basemodule import BaseModule


class Comment(BaseModule):
    """
    A module that allows adding comments to ``Model`` summaries.
    """

    def __init__(self, comment: str = None) -> None:
        """
        Create a ``Comment`` module.

        Parameters
        ----------
        comment
            Comment text to be shown in the ``Model`` summary where this module
            is placed.

        """
        self.comment = comment

    @property
    def name(self) -> str:
        return f"# {self.comment}" if self.comment else "#"

    def is_ready(self) -> bool:
        return True

    def get_num_trainable_floats(self) -> int | None:
        return 0

    def _get_callable(
        self,
    ) -> ModuleCallable:
        return lambda params, data, training, state, rng: (
            data,  # output is the same as input
            state,  # state is unchanged
        )

    def compile(self, rng: Any, input_shape: DataShape) -> None:
        pass

    def get_output_shape(self, input_shape: DataShape) -> DataShape:
        return input_shape  # output shape is the same as input shape

    def get_hyperparameters(self) -> HyperParams:
        return {"comment": self.comment}

    def get_params(self) -> Params:
        return ()

    def set_params(self, params: Params) -> None:
        pass

    def set_state(self, state: State) -> None:
        pass
