from typing import Any, Callable, Dict, Optional, Tuple

import jax.numpy as np

from .basemodule import BaseModule


class Reshape(BaseModule):
    """
    Modules that reshapes the input array to a specified shape. Ignores the
    batch dimension.
    """

    def __init__(self, shape: Optional[Tuple[int, ...]] = None) -> None:
        """
        Parameters
        ----------
        shape : Optional[Tuple[int, ...]], optional
            The target shape to reshape the input to, by default None.
            If None, the input shape will remain unchanged.
            Does not include the batch dimension.
        """
        self.shape = shape

    def name(self) -> str:
        return f"Reshape(shape={self.shape})"

    def is_ready(self) -> bool:
        return True

    def get_num_trainable_floats(self) -> Optional[int]:
        return 0

    def _get_callable(
        self,
    ) -> Callable[
        [
            Tuple[np.ndarray, ...],
            np.ndarray,
            bool,
            Tuple[np.ndarray, ...],
            Any,
        ],
        Tuple[np.ndarray, Tuple[np.ndarray, ...]],
    ]:
        return lambda params, input_NF, training, state, rng: (
            (
                input_NF.reshape(input_NF.shape[0], *self.shape)
                if self.shape
                else input_NF
            ),
            state,  # state is unchanged
        )

    def compile(self, rng: Any, input_shape: Tuple[int, ...]) -> None:
        pass

    def get_output_shape(
        self, input_shape: Tuple[int, ...]
    ) -> Tuple[int, ...]:
        # handle the special cases where self.shape is None or (-1,)
        if self.shape is None:
            return input_shape
        elif self.shape == (-1,):
            return (np.prod(np.array(input_shape)).item(),)
        else:
            return self.shape

    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            "shape": self.shape,
        }

    def get_params(self) -> Tuple[np.ndarray, ...]:
        return ()

    def set_params(self, params: Tuple[np.ndarray, ...]) -> None:
        pass
