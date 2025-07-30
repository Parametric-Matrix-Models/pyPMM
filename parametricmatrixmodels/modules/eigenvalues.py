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
        """
        Returns the name of the module
        """

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
        """
        This method must return a jax-jittable and jax-gradable callable in the
        form of
        ```
        (
            params: Tuple[np.ndarray, ...],
            input_NF: np.ndarray[num_samples, num_features],
            training: bool,
            state: Tuple[np.ndarray, ...],
            rng: key<fry>
        ) -> (
                output_NF: np.ndarray[num_samples, num_output_features],
                new_state: Tuple[np.ndarray, ...]
            )
        ```
        That is, all hyperparameters are traced out and the callable depends
        explicitly only on a Tuple of parameter numpy arrays, the input array,
        the training flag, a state Tuple of numpy arrays, and a JAX rng key.

        The training flag will be traced out, so it doesn't need to be jittable
        """
        return lambda params, input_NF, training, state, rng: (
            jax.vmap(select_eigenvalues, in_axes=(0, None, None))(
                np.linalg.eigvalsh(input_NF), self.num_eig, self.which
            ),
            state,  # state is not used in this module, return it unchanged
        )

    def compile(self, rng: Any, input_shape: Tuple[int, ...]) -> None:
        """
        Compile the module to be used with the given input shape.

        This method should initialize the module's parameters and state based
        on the input shape and random key.

        This is needed since Models are built before the input data is given,
        so before training or inference can be done, the module needs to be
        compiled and each Module passes its output shape to the next Module's
        compile method.

        The rng key is used to initialize random parameters, if needed.

        Parameters
        ----------
        rng : Any
            JAX random key.
        input_shape : Tuple[int, ...]
            Shape of the input data, e.g. (num_features,).
        """

        # ensure input shape is valid
        if len(input_shape) != 2 or input_shape[0] != input_shape[1]:
            raise ValueError(
                f"Input shape must be a square matrix, got {input_shape}"
            )

    def get_output_shape(
        self, input_shape: Tuple[int, ...]
    ) -> Tuple[int, ...]:
        """
        Get the output shape of the module given the input shape.

        Parameters
        ----------
        input_shape : Tuple[int, ...]
            Shape of the input data, e.g. (num_features,).

        Returns
        Tuple[int, ...]
            Shape of the output data, e.g. (num_output_features,).
        """
        return (self.num_eig,)

    def get_hyperparameters(self) -> Dict[str, Any]:
        """
        Get the hyperparameters of the module.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the hyperparameters of the module.
        """
        return {
            "num_eig": self.num_eig,
            "which": self.which,
        }

    def set_hyperparameters(self, hyperparams: Dict[str, Any]) -> None:
        """
        Set the hyperparameters of the module, using the default
        implementation. Just to input validation.

        Parameters
        ----------
        hyperparams : Dict[str, Any]
            Dictionary containing the hyperparameters to set.
        """

        super(Eigenvalues, self).set_hyperparameters(hyperparams)

    def get_params(self) -> Tuple[np.ndarray, ...]:
        """
        Get the current trainable parameters of the module. If the module has
        no trainable parameters, this method should return an empty tuple.

        Returns
        -------
        Tuple[np.ndarray, ...]
            Tuple of numpy arrays representing the module's parameters.
        """
        return ()

    def set_params(self, params: Tuple[np.ndarray, ...]) -> None:
        """
        Set the parameters of the module.

        Parameters
        ----------
        params : Tuple[np.ndarray, ...]
            Tuple of numpy arrays representing the new parameters.
        """
        return
