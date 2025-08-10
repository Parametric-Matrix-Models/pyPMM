from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as np

from ._affine_backing_funcs import select_eigenvalues
from .basemodule import BaseModule


class Eigenvalues(BaseModule):

    def __init__(
        self,
        num_eig: int = 1,
        which: str = "SA",
    ) -> None:
        if num_eig <= 0 or not isinstance(num_eig, int):
            raise ValueError("num_eig must be a positive integer")
        if which.lower() not in [
            "sa",
            "la",
            "sm",
            "lm",
            "ea",
            "em",
            "ia",
            "im",
        ]:
            raise ValueError(
                "which must be one of: 'SA', 'LA', 'SM', 'LM', 'EA', 'EM', "
                f"'IA', 'IM'. Got: {which}"
            )

        self.num_eig = num_eig
        self.which = which.lower()

    def name(self) -> str:
        if self.num_eig == 1 and self.which == "sa":
            return "Eigenvalues(ground state)"
        else:
            return (
                f"Eigenvalues(num_eig={self.num_eig},"
                f" which={self.which.upper()})"
            )

    def is_ready(self) -> bool:
        return True

    def get_num_trainable_floats(self) -> Optional[int]:
        return 0

    def _get_callable(self) -> Callable:
        return lambda params, input_NF, training, state, rng: (
            jax.vmap(select_eigenvalues, in_axes=(0, None, None))(
                np.linalg.eigvalsh(input_NF), self.num_eig, self.which
            ),
            state,  # state is not used in this module, return it unchanged
        )

    def compile(self, rng: Any, input_shape: Tuple[int, ...]) -> None:
        # ensure input shape is valid
        if len(input_shape) != 2 or input_shape[0] != input_shape[1]:
            raise ValueError(
                f"Input shape must be a square matrix, got {input_shape}"
            )

    def get_output_shape(
        self, input_shape: Tuple[int, ...]
    ) -> Tuple[int, ...]:
        return (self.num_eig,)

    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            "num_eig": self.num_eig,
            "which": self.which,
        }

    def set_hyperparameters(self, hyperparams: Dict[str, Any]) -> None:
        super(Eigenvalues, self).set_hyperparameters(hyperparams)

    def get_params(self) -> Tuple[np.ndarray, ...]:
        return ()

    def set_params(self, params: Tuple[np.ndarray, ...]) -> None:
        return
