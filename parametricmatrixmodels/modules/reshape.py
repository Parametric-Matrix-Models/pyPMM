from typing import Any, Callable, Dict, Optional, Tuple

import jax.numpy as np

from .basemodule import BaseModule


class Reshape(BaseModule):

    def __init__(self, shape: Optional[Tuple[int, ...]] = None) -> None:
        """
        Reshape module that reshapes the input array to the specified shape.

        Parameters
        ----------
        shape : Optional[Tuple[int, ...]], optional
            The target shape to reshape the input to, by default None.
            If None, the input shape will remain unchanged.
            Does not include the batch dimension.
        """
        self.shape = shape

    def name(self) -> str:
        """
        Returns the name of the module
        """
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
            (
                input_NF.reshape(input_NF.shape[0], *self.shape)
                if self.shape
                else input_NF
            ),
            state,  # state is unchanged
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
        pass

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

        # handle the special cases where self.shape is None or (-1,)
        if self.shape is None:
            return input_shape
        elif self.shape == (-1,):
            return (np.prod(np.array(input_shape)).item(),)
        else:
            return self.shape

    def get_hyperparameters(self) -> Dict[str, Any]:
        """
        Get the hyperparameters of the module.

        Hyperparameters are used to configure the module and are not trainable.
        They can be set via `set_hyperparameters`.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the hyperparameters of the module.
        """
        return {
            "shape": self.shape,
        }

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
        pass
