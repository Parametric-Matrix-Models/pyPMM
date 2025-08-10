"""
Base module for JAX-based PMM models
The base module can be used to implement various PMM models, NN models, and
other (optionally stateful and trainable) operations in JAX.
Modules can be combined to create Models.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as np
from packaging.version import parse

import parametricmatrixmodels as pmm


class BaseModule(object):
    """
    Base class for all Modules. Custom modules should inherit from this class.
    """

    def __init__(self) -> None:
        """
        BaseModule constructor, must be overridden by subclasses.

        All modules **must** be able to be initialized without any arguments in
        order for Model saving and loading to work correctly.

        ``__init__`` can take optional parameters, but all aspects of the
        module must be able to be set by ``set_hyperparameters``,
        ``set_params``, and ``set_state``.

        Always raises ``NotImplementedError`` when called on ``BaseModule`` s.

        ``BaseModule`` is not meant to be instantiated directly.
        """
        raise NotImplementedError(
            "BaseModule is an abstract class and cannot be instantiated "
            "directly."
        )

    def name(self) -> str:
        """
        Returns the name of the module, unless overridden, this is the class
        name.

        Returns
        -------
            Name of the module.
        """
        return self.__class__.__name__

    def __repr__(self) -> str:
        """
        Returns a string representation of the module. Unless overridden,
        this includes the module name, number of trainable floats (if any),
        and whether the module is initialized (ready) or not.

        Returns
        -------
            String representation of the module.
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
        Return True if the module is initialized and ready for training or
        inference.

        Returns
        -------
            ``True`` if the module is ready, ``False`` otherwise.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the subclass.
        """
        raise NotImplementedError(
            "is_ready method must be implemented in subclasses"
        )

    def get_num_trainable_floats(self) -> int | None:
        """
        Returns the number of trainable floats in the module.
        If the module does not have trainable parameters, returns 0.
        If the module is not ready, returns None.

        Returns
        -------
            Number of trainable floats in the module, or None if the module
            is not ready.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the subclass.
        """
        raise NotImplementedError(
            "get_num_trainable_floats method must be implemented in subclasses"
        )

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
        """
        Returns a ``jax.jit``-able and ``jax.grad``-able callable that
        represents the module's forward pass.

        This method must be implemented by all subclasses and must return a
        ``jax-jit``-able and ``jax-grad``-able callable in the form of

        .. code-block:: python

            module_callable(
                params: tuple[np.ndarray, ...],
                input_NF: np.ndarray[num_samples, num_features],
                training: bool,
                state: tuple[np.ndarray, ...],
                rng: Any
            ) -> (
                    output_NF: np.ndarray[num_samples, num_output_features],
                    new_state: tuple[np.ndarray, ...]
                )

        That is, all hyperparameters are traced out and the callable depends
        explicitly only on a ``tuple`` of parameter ``jax.numpy`` arrays,
        the input array, the training flag, a state ``tuple`` of ``jax.numpy``
        arrays, and a JAX rng key.

        The training flag will be traced out, so it doesn't need to be jittable

        Returns
        -------
            A callable that takes the module's parameters, input data,
            training flag, state, and rng key and returns the output data and
            new state.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the subclass.

        See Also
        --------
        __call__ : Calls the module with the current parameters and
            given input, state, and rng.
        """
        raise NotImplementedError(
            "_get_callable method must be implemented in subclasses"
        )

    def __call__(
        self,
        input_NF: np.ndarray,
        training: bool = False,
        state: tuple[np.ndarray, ...] = (),
        rng: Any = None,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        """
        Call the module with the current parameters and given input, state, and
        rng.

        Parameters
        ----------
        input_NF
            Input array of shape (num_samples, num_features).
        training
            Whether the module is in training mode, by default False.
        state
            State of the module, by default ``()``.
        rng
            JAX random key, by default None.

        Returns
        -------
            Output array of shape (num_samples, num_output_features) and new
            state.

        Raises
        ------
        ValueError
            If the module is not ready (i.e., `compile()` has not been called).

        See Also
        --------
        _get_callable : Returns a callable that can be used to
            compute the output and new state given the parameters, input,
            training flag, state, and rng.
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

    def compile(self, rng: Any, input_shape: tuple[int, ...]) -> None:
        """
        Compile the module to be used with the given input shape.

        This method should initialize the module's parameters and state based
        on the input shape and random key.

        This is needed since Models are built before the input data is given,
        so before training or inference can be done, the module needs to be
        compiled and each Module passes its output shape to the next Module's
        compile method.

        The rng key is used to initialize random parameters, if needed.

        This is **not** used to trace or jit the module's callable, that is
        done automatically later.

        Parameters
        ----------
        rng
            JAX random key.
        input_shape
            Shape of the input data, e.g. ``(num_features,)``.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the subclass.
        """
        raise NotImplementedError(
            "compile method must be implemented in subclasses"
        )

    def get_output_shape(
        self, input_shape: tuple[int, ...]
    ) -> tuple[int, ...]:
        """
        Get the output shape of the module given the input shape.

        Parameters
        ----------
        input_shape
            Shape of the input data, e.g. ``(num_features,)``.

        Returns
        -------
            Shape of the output data, e.g. ``(num_output_features,)``.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the subclass.
        """
        raise NotImplementedError(
            "get_output_shape method must be implemented in subclasses"
        )

    def get_hyperparameters(self) -> dict[str, Any]:
        """
        Get the hyperparameters of the module.

        Hyperparameters are used to configure the module and are not trainable.
        They can be set via `set_hyperparameters`.

        Returns
        -------
            Dictionary containing the hyperparameters of the module.
        """
        raise NotImplementedError(
            "get_hyperparameters method must be implemented in subclasses"
        )

    def set_hyperparameters(self, hyperparameters: dict[str, Any]) -> None:
        """
        Set the hyperparameters of the module.

        Hyperparameters are used to configure the module and are not trainable.
        They can be set via this method.

        The default implementation uses setattr to set the hyperparameters as
        attributes of the class instance.

        Parameters
        ----------
        hyperparameters
            Dictionary containing the hyperparameters to set.

        Raises
        ------
        TypeError
            If hyperparameters is not a dictionary.
        """
        if not isinstance(hyperparameters, dict):
            raise TypeError(
                "Hyperparameters must be provided as a dictionary."
            )
        for key, value in hyperparameters.items():
            setattr(self, key, value)

    def get_params(self) -> tuple[np.ndarray, ...]:
        """
        Get the current trainable parameters of the module. If the module has
        no trainable parameters, this method should return an empty tuple.

        Returns
        -------
            Tuple of numpy arrays representing the module's parameters.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the subclass.
        """
        raise NotImplementedError(
            "get_params method must be implemented in subclasses"
        )

    def set_params(self, params: tuple[np.ndarray, ...]) -> None:
        """
        Set the trainable parameters of the module.

        Parameters
        ----------
        params
            Tuple of numpy arrays representing the new parameters.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the subclass.
        """
        raise NotImplementedError(
            "set_params method must be implemented in subclasses"
        )

    def get_state(self) -> tuple[np.ndarray, ...]:
        """
        Get the current state of the module.

        States are used to store "memory" or other information that is not
        passed between modules, is not trainable, but may be updated during
        either training or inference. e.g. batch normalization state.

        The state is optional, in which case this method should return the
        empty tuple.

        Returns
        -------
            Tuple of numpy arrays representing the module's state.
        """
        return ()

    def set_state(self, state: tuple[np.ndarray, ...]) -> None:
        """
        Set the state of the module.

        This method is optional.

        Parameters
        ----------
        state
            Tuple of numpy arrays representing the new state.
        """
        pass

    def set_precision(self, prec: np.dtype | str | int) -> None:
        """
        Set the precision of the module parameters and state.

        Parameters
        ----------
            prec
                Precision to set for the module parameters.
                Valid options are:
                *For 32-bit precision (all options are equivalent)*
                ``np.float32``, ``np.complex64``, ``"float32"``,
                ``"complex64"``, ``"single"``, ``"f32"``, ``"c64"``, ``32``.
                *For 64-bit precision (all options are equivalent)*
                ``np.float64``, ``np.complex128``, ``"float64"``,
                ``"complex128"``, ``"double"``, ``"f64"``, ``"c128"``, ``64``.

        Raises
        ------
        ValueError
            If the precision is invalid or if 64-bit precision is requested
            but ``JAX_ENABLE_X64`` is not set.
        RuntimeError
            If the module is not ready (i.e., `compile()` has not been called).

        See Also
        --------
        astype
            Convenience wrapper to set_precision using the dtype argument,
            returns self.
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

    def astype(self, dtype: np.dtype | str) -> "BaseModule":
        """
        Convenience wrapper to set_precision using the dtype argument, returns
        self.

        Parameters
        ----------
        dtype
            Precision to set for the module parameters.
            Valid options are:
            *For 32-bit precision (all options are equivalent)*
            ``np.float32``, ``np.complex64``, ``"float32"``,
            ``"complex64"``, ``"single"``, ``"f32"``, ``"c64"``, ``32``
            *For 64-bit precision (all options are equivalent)*
            ``np.float64``, ``np.complex128``, ``"float64"``,
            ``"complex128"``, ``"double"``, ``"f64"``, ``"c128"``, ``64``

        Returns
        -------
        BaseModule
            The module itself, with updated precision.

        Raises
        ------
        ValueError
            If the precision is invalid or if 64-bit precision is requested
            but ``JAX_ENABLE_X64`` is not set.
        RuntimeError
            If the module is not ready (i.e., `compile()` has not been called).

        See Also
        --------
        set_precision
            Sets the precision of the module parameters and state.
        """
        self.set_precision(dtype)
        return self

    def serialize(self) -> dict[str, Any]:
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
            Dictionary containing the serialized module data.
        """

        return {
            "name": self.name(),
            "hyperparameters": self.get_hyperparameters(),
            "params": {f"p{i}": p for i, p in enumerate(self.get_params())},
            "state": {f"s{i}": s for i, s in enumerate(self.get_state())},
            "package_version": pmm.__version__,
        }

    def deserialize(self, data: dict[str, Any]) -> None:
        """
        Deserialize the module from a dictionary.

        This method should set the module's parameters and state based on the
        provided dictionary.

        The default implementation expects the dictionary to contain
        the module's name, trainable parameters, and state.

        Parameters
        ----------
        data
            Dictionary containing the serialized module data.
        """

        # read the version of the package this module was serialized with
        current_version = parse(pmm.__version__)
        package_version = parse(str(data["package_version"]))

        if current_version != package_version:
            # in the future, we will issue DeprecationWarnings or Errors if the
            # version is unsupported
            # or possibly handle version-specific deserialization
            pass

        # set the hyperparameters
        self.set_hyperparameters(data.get("hyperparameters", {}))

        # if there are trainable parameters, set them
        params_dict = data.get("params", {})
        params = tuple(params_dict[f"p{i}"] for i in range(len(params_dict)))
        if len(params) > 0:
            self.set_params(params)

        # if there are states, set them
        state_dict = data.get("state", {})
        state = tuple(state_dict[f"s{i}"] for i in range(len(state_dict)))
        if len(state) > 0:
            self.set_state(state)
