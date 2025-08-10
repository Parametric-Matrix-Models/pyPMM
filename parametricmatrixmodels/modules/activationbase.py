from typing import Any, Callable, Dict, Tuple

import jax.numpy as np

from .basemodule import BaseModule


class ActivationBase(BaseModule):
    """
    Base class for activation function modules. Not to be instantiated
    directly.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the elementwise activation function module.

        Parameters
        ----------
        args : tuple
            Positional arguments for the activation function, starts with the
            second argument, as the first is the input array.
        kwargs : dict
            Keyword arguments for the activation function.
        """
        self.args = args
        self.kwargs = kwargs

    def name(self) -> str:
        raise NotImplementedError("Subclasses must implement the name method.")

    def is_ready(self) -> bool:
        """
        Check if the module is ready to be used. Activation functions are
        always ready.

        Returns
        -------
        bool
            Always returns True.
        """
        return True

    def get_num_trainable_floats(self) -> int:
        """
        Get the number of trainable floats in the module. Activation functions
        do not have trainable parameters.

        Returns
        -------
        int
            Always returns 0.
        """
        return 0

    def func(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the activation function to the input array.

        Parameters
        ----------
        x : np.ndarray
            Input array to the activation function.

        Returns
        -------
        np.ndarray
            Output array after applying the activation function.
        """
        raise NotImplementedError("Subclasses must implement the func method.")

    def _get_callable(self) -> Callable:
        """
        Get the callable for the activation function.

        Returns
        -------
        Callable
            The activation function callable in the form the PMM library
            expects
        """

        return lambda params, input_NF, training, state, rng: (
            self.func(input_NF),
            state,
        )

    def compile(self, rng: Any, input_shape: Tuple[int, ...]) -> None:
        """
        Compile the activation function module. This method is a no-op for
        activation functions.

        Parameters
        ----------
        rng : Any
            Random number generator state.
        input_shape : Tuple[int, ...]
            Shape of the input array.
        """
        pass

    def get_output_shape(
        self, input_shape: Tuple[int, ...]
    ) -> Tuple[int, ...]:
        """
        Get the output shape of the activation function given the input shape.

        Parameters
        ----------
        input_shape : Tuple[int, ...]
            Shape of the input array.

        Returns
        -------
        Tuple[int, ...]
            Output shape after applying the activation function.
        """
        return input_shape

    def get_hyperparameters(self) -> Dict[str, Any]:
        """
        Get the hyperparameters of the activation function module.

        Returns
        -------
        Dict[str, Any]
            Hyperparameters of the activation function.
        """
        return {
            "args": self.args,
            "kwargs": self.kwargs,
        }

    def get_params(self) -> Tuple[np.ndarray, ...]:
        """
        Get the parameters of the activation function module.

        Returns
        -------
        Tuple[np.ndarray, ...]
            An empty tuple, as activation functions do not have parameters.
        """
        return ()

    def set_params(self, params: Tuple[np.ndarray, ...]) -> None:
        return
