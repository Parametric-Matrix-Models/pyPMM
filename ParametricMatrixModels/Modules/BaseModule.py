"""
Base module for JAX-based PMM models

The base module can be used to implement various PMM models, NN models, and
other (optionally stateful and trainable) operations in JAX.

Modules can be combined to create Models.
"""

import jax
import jax.numpy as np
from jax import jit, vmap, lax
from typing import Callable, Tuple, Any, Union, Optional, Dict


class BaseModule(object):
    """
    Base class for all Modules
    """

    def __init__(self) -> None:
        """
        All modules MUST be able to be initialized without any parameters in
        order for Model saving and loading to work correctly.

        __init__ can take optional parameters, but all aspects of the module
        must be able to be set by `set_hyperparameters`, `set_params`, and
        `set_state`.

        Always raises NotImplementedError when called.

        BaseModule is not meant to be instantiated directly.
        """
        raise NotImplementedError(
            "BaseModule is an abstract class and cannot be instantiated directly."
        )

    def name(self) -> str:
        """
        Returns the name of the module
        """
        return self.__class__.__name__

    def __repr__(self) -> str:
        """
        Returns a string representation of the module
        """

        param_count = self.get_num_trainable_floats()
        ready = self.is_ready()
        if param_count is not None and ready:
            return f"{self.name()} (trainable floats: {param_count:,})"
        elif not ready:
            return f"{self.name()} (uninitialized)"
        else:
            return f"{self.name()}"

    def is_ready(self) -> bool:
        """
        Return True if the module is initialized and ready for training or inference.
        """
        raise NotImplementedError(
            "is_ready method must be implemented in subclasses"
        )

    def get_num_trainable_floats(self) -> Optional[int]:
        """
        Returns the number of trainable floats in the module.
        If the module does not have trainable parameters, returns 0.
        If the module is not ready, returns None.
        """
        raise NotImplementedError(
            "get_num_trainable_floats method must be implemented in subclasses"
        )

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
        raise NotImplementedError(
            "_get_callable method must be implemented in subclasses"
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
        raise NotImplementedError(
            "compile method must be implemented in subclasses"
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
        raise NotImplementedError(
            "get_output_shape method must be implemented in subclasses"
        )

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
        raise NotImplementedError(
            "get_hyperparameters method must be implemented in subclasses"
        )

    def set_hyperparameters(self, hyperparameters: Dict[str, Any]) -> None:
        """
        Set the hyperparameters of the module.

        Hyperparameters are used to configure the module and are not trainable.
        They can be set via this method.

        The default implementation uses setattr to set the hyperparameters as
        attributes of the class instance.

        Parameters
        ----------
        hyperparameters : Dict[str, Any]
            Dictionary containing the hyperparameters to set.
        """
        if not isinstance(hyperparameters, dict):
            raise TypeError(
                "Hyperparameters must be provided as a dictionary."
            )
        for key, value in hyperparameters.items():
            setattr(self, key, value)

    def get_params(self) -> Tuple[np.ndarray, ...]:
        """
        Get the current trainable parameters of the module. If the module has
        no trainable parameters, this method should return an empty tuple.

        Returns
        -------
        Tuple[np.ndarray, ...]
            Tuple of numpy arrays representing the module's parameters.
        """
        raise NotImplementedError(
            "get_params method must be implemented in subclasses"
        )

    def set_params(self, params: Tuple[np.ndarray, ...]) -> None:
        """
        Set the parameters of the module.

        Parameters
        ----------
        params : Tuple[np.ndarray, ...]
            Tuple of numpy arrays representing the new parameters.
        """
        raise NotImplementedError(
            "set_params method must be implemented in subclasses"
        )

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

    def set_precision(self, prec: Union[np.dtype, str, int]) -> None:
        """
        Set the precision of the module parameters and state.

        Parameters
        ----------
            prec : Union[np.dtype, str, int]
                Precision to set for the module parameters.
                Valid options are:
                    [for 32-bit precision (all options are equivalent)]
                    - np.float32, np.complex64, "float32", "complex64"
                    - "single", "f32", "c64", 32
                    [for 64-bit precision (all options are equivalent)]
                    - np.float64, np.complex128, "float64", "complex128"
                    - "double", "f64", "c128", 64
        """
        if not self.is_ready():
            raise RuntimeError("Module is not ready. Call compile() first.")

        # convert precision to 32 or 64
        if prec in [
            np.float32,
            np.complex64,
            "float32",
            "complex64",
            "single",
            "f32",
            "c64",
            32,
        ]:
            prec = 32
        elif prec in [
            np.float64,
            np.complex128,
            "float64",
            "complex128",
            "double",
            "f64",
            "c128",
            64,
        ]:
            prec = 64
        else:
            raise ValueError(
                "Invalid precision. Valid options are:\n"
                "[for 32-bit precision] np.float32, np.complex64, 'float32', "
                "'complex64', 'single', 'f32', 'c64', 32;\n"
                "[for 64-bit precision] np.float64, np.complex128, 'float64', "
                "'complex128', 'double', 'f64', 'c128', 64."
            )

        # check if dtype is supported
        if not jax.config.read("jax_enable_x64") and prec == 64:
            raise ValueError(
                "JAX_ENABLE_X64 is not set. "
                "Please set it to True to use double precision float64 or "
                "complex128 data types."
            )

        def set_param_prec(p: np.ndarray) -> np.ndarray:
            """
            Set the precision of a single parameter array, choosing real or
            complex precision based on the original dtype.
            """
            if np.iscomplexobj(p):
                return p.astype(np.complex64 if prec == 32 else np.complex128)
            else:
                return p.astype(np.float32 if prec == 32 else np.float64)

        self.set_params(tuple(set_param_prec(p) for p in self.get_params()))
        self.set_state(tuple(set_param_prec(s) for s in self.get_state()))

    def astype(self, dtype: Union[np.dtype, str]) -> "BaseModule":
        """
        Convenience wrapper to set_precision using the dtype argument, returns
        self.
        """
        self.set_precision(dtype)
        return self

    def serialize(self) -> Dict[str, Any]:
        """
        Serialize the module to a dictionary.

        This method should return a dictionary representation of the module,
        including its parameters and state.

        The default implementation serializes the module's name,
        hyperparameters, trainable parameters, and state via a simple
        dictionary.

        This only works if the module's hyperparameters are auto-serializable.
        This includes basic types as well as numpy arrays.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the serialized module data.
        """

        return {
            "name": self.name(),
            "hyperparameters": self.get_hyperparameters(),
            **{f"p{i}": p for i, p in enumerate(self.get_params())},
            **{f"s{i}": s for i, s in enumerate(self.get_state())},
        }

    def deserialize(self, data: Dict[str, Any]) -> None:
        """
        Deserialize the module from a dictionary.

        This method should set the module's parameters and state based on the
        provided dictionary.

        The default implementation expects the dictionary to contain
        the module's name, trainable parameters, and state.

        Parameters
        ----------
        data : Dict[str, Union[np.ndarray, str]]
            Dictionary containing the serialized module data.
        """

        # set the hyperparameters
        self.set_hyperparameters(data.get("hyperparameters", {}))

        # if there are trainable parameters, set them
        if "p0" in data:
            # get the number of parameter arrays
            num_params = len([k for k in data.keys() if k.startswith("p")])
            params = tuple(data[f"p{i}"] for i in range(num_params))
            self.set_params(params)
        # if there are states, set them
        if "s0" in data:
            # get the number of state arrays
            num_states = len([k for k in data.keys() if k.startswith("s")])
            state = tuple(data[f"s{i}"] for i in range(num_states))
            self.set_state(state)
