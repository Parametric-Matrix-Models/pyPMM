from abc import abstractmethod

import jax
import jax.numpy as np
from beartype import beartype
from jaxtyping import jaxtyped

from parametricmatrixmodels.typing import (
    Any,
    Data,
    DataShape,
    HyperParams,
    ModuleCallable,
    Params,
    State,
    Tuple,
)

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
        args
            Positional arguments for the activation function, starts with the
            second argument, as the first is the input array.
        kwargs
            Keyword arguments for the activation function.
        """
        self.args = args
        self.kwargs = kwargs

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError("Subclasses must implement the name method.")

    def is_ready(self) -> bool:
        """
        Check if the module is ready to be used. Activation functions are
        always ready.

        Returns
        -------
            Always returns True.
        """
        return True

    def get_num_trainable_floats(self) -> int | None:
        """
        Get the number of trainable floats in the module. Activation functions
        do not have trainable parameters.

        Returns
        -------
            Always returns 0.
        """
        return 0

    @abstractmethod
    def func(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the activation function to the input array.

        Parameters
        ----------
        x
            Input array to the activation function.

        Returns
        -------
            Output array after applying the activation function.
        """
        raise NotImplementedError("Subclasses must implement the func method.")

    def _get_callable(self) -> ModuleCallable:
        """
        Get the callable for the activation function.

        Returns
        -------
            The activation function callable in the form the PMM library
            expects
        """

        @jaxtyped(typechecker=beartype)
        def callable(
            params: Params,
            data: Data,
            training: bool,
            state: State,
            rng: Any,
        ) -> Tuple[Data, State]:
            return jax.tree.map(self.func, data), state

        return callable

    def compile(self, rng: Any, input_shape: DataShape) -> None:
        """
        Compile the activation function module. This method is a no-op for
        activation functions.

        Parameters
        ----------
        rng
            Random number generator state.
        input_shape
            Shape of the input arrays.
        """
        pass

    def get_output_shape(self, input_shape: DataShape) -> DataShape:
        """
        Get the output shape of the activation function given the input shape.

        Parameters
        ----------
        input_shape
            Shape of the input arrays.

        Returns
        -------
            Output shapes after applying the activation function.
        """
        return input_shape

    def get_hyperparameters(self) -> HyperParams:
        """
        Get the hyperparameters of the activation function module.

        Returns
        -------
            Hyperparameters of the activation function.
        """
        return {
            "args": self.args,
            "kwargs": self.kwargs,
        }

    def get_params(self) -> Params:
        """
        Get the parameters of the activation function module.

        Returns
        -------
            An empty tuple, as activation functions do not have parameters.
        """
        return ()

    def set_params(self, params: Params) -> None:
        return
