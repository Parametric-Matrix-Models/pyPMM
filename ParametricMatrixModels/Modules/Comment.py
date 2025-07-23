import jax
import jax.numpy as np
from jax import jit, vmap, lax
from typing import Callable, Tuple, Any, Union, Optional, Dict
from .BaseModule import BaseModule


class Comment(BaseModule):

    def __init__(self, comment: Optional[str] = None) -> None:
        self.comment = comment

    def name(self) -> str:
        """
        Returns the name of the module
        """
        return f"# {self.comment}" if self.comment else "#"

    def __repr__(self) -> str:
        """
        Returns a string representation of the module
        """
        return f"{self.name()}"

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
            input_NF,  # output is the same as input
            state,  # state is unchanged
        )

    def __call__(
        self,
        input_NF: np.ndarray,
        training: bool = False,
        state: Tuple[np.ndarray, ...] = (),
        rng: Any = None,
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, ...]]:
        """
        Call the module with the current parameters and given input, state, and
        rng.

        Parameters
        ----------
        input_NF : np.ndarray
            Input array of shape (num_samples, num_features).
        training : bool, optional
            Whether the module is in training mode, by default False.
        state : Tuple[np.ndarray, ...], optional
            State of the module, by default ().
        rng : Any, optional
            JAX random key, by default None.

        Returns
        -------
        Tuple[np.ndarray, Tuple[np.ndarray, ...]]
            Output array of shape (num_samples, num_output_features) and new
            state.
        """
        if not self.is_ready():
            raise ValueError("Module is not ready, call compile() first")

        # get the callable
        func = self._get_callable()

        # call the function with the current parameters, input, training flag,
        # state, and rng
        return func(
            self.get_params(),
            input_NF,
            training,
            state,
            rng,
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
        return input_shape  # output shape is the same as input shape

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
        return {"comment": self.comment}

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

    def get_state(self) -> Tuple[np.ndarray, ...]:
        """
        Get the current state of the module.

        States are used to store "memory" or other information that is not
        passed between modules, is not trainable, but may be updated during
        either training or inference. e.g. batch normalization state.

        The state is optional, in which case this method should return the
        empty tuple.

        Returns
        -------
        Tuple[np.ndarray, ...]
            Tuple of numpy arrays representing the module's state.
        """
        return ()

    def set_state(self, state: Tuple[np.ndarray, ...]) -> None:
        """
        Set the state of the module.

        This method is optional.

        Parameters
        ----------
        state : Tuple[np.ndarray, ...]
            Tuple of numpy arrays representing the new state.
        """
        pass
