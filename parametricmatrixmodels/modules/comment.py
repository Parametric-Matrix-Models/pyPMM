from __future__ import annotations

from typing import Any, Callable

import jax.numpy as np

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

    def name(self) -> str:
        return f"# {self.comment}" if self.comment else "#"

    def is_ready(self) -> bool:
        return True

    def get_num_trainable_floats(self) -> int | None:
        return 0

    def _get_callable(
        self,
    ) -> Callable[
        [
            tuple[np.ndarray, ...],
            np.ndarray,
            bool,
            tuple[np.ndarray, ...],
            Any,
        ],
        tuple[np.ndarray, tuple[np.ndarray, ...]],
    ]:
        return lambda params, input_NF, training, state, rng: (
            input_NF,  # output is the same as input
            state,  # state is unchanged
        )

    def compile(self, rng: Any, input_shape: tuple[int, ...]) -> None:
        pass

    def get_output_shape(
        self, input_shape: tuple[int, ...]
    ) -> tuple[int, ...]:
        return input_shape  # output shape is the same as input shape

    def get_hyperparameters(self) -> dict[str, Any]:
        return {"comment": self.comment}

    def get_params(self) -> tuple[np.ndarray, ...]:
        return ()

    def set_params(self, params: tuple[np.ndarray, ...]) -> None:
        pass

    def get_state(self) -> tuple[np.ndarray, ...]:
        return ()

    def set_state(self, state: tuple[np.ndarray, ...]) -> None:
        pass
