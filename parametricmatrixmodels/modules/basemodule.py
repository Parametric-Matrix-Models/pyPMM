"""
Base module for JAX-based PMM models
The base module can be used to implement various PMM models, NN models, and
other (optionally stateful and trainable) operations in JAX.
Modules can be combined to create Models.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod

import jax
import jax.numpy as np
from beartype import beartype
from jaxtyping import jaxtyped
from packaging.version import InvalidVersion, Version, parse

import parametricmatrixmodels as pmm
from parametricmatrixmodels.typing import (
    Any,
    Data,
    DataShape,
    Dict,
    HyperParams,
    ModuleCallable,
    Params,
    State,
    Tuple,
)


class BaseModule(ABC):
    """
    Base class for all Modules. Custom modules should inherit from this class.
    """

    # version of the module class, used for serialization, must be implemented
    # by subclasses
    __version__: str

    @abstractmethod
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

    def __init_subclass__(cls, **kwargs):
        r"""
        Ensures that all requirements for concrete subclasses are met:

        1. That all methods of all subclasses of BaseModule are also
        decorated with ``@jaxtyped(typechecker=beartype)``. This includes
        "private" methods (those starting with an underscore).

        2. That the ``__version__`` attribute is set and is a
        valid version string.

        3. That __init__ has no required arguments.
        """
        super().__init_subclass__(**kwargs)

        # only continue if the subclass is concrete
        if inspect.isabstract(cls):
            return

        for name, method in cls.__dict__.items():
            if callable(method) and not hasattr(method, "__jaxtyped__"):
                setattr(cls, name, jaxtyped(typechecker=beartype)(method))
                # set the __jaxtyped__ attribute to avoid re-wrapping
                getattr(cls, name).__jaxtyped__ = True

        # ensure that __version__ is set
        if not hasattr(cls, "__version__"):
            raise NotImplementedError(
                f"Subclass {cls.__name__} must define a __version__ attribute."
            )
        # ensure that __version__ is a valid version string
        try:
            Version(cls.__version__)
        except InvalidVersion as e:
            raise ValueError(
                f"Invalid version string '{cls.__version__}' in subclass "
                f"{cls.__name__}. Version strings must follow PEP 440. See "
                "https://peps.python.org/pep-0440/ for more information."
            ) from e

        # Check if any parameters (other than 'self') are required
        sig = inspect.signature(cls.__init__)
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            # If a parameter has no default value, it's required
            if (
                param.default == inspect.Parameter.empty
                and param.kind
                not in (
                    inspect.Parameter.VAR_POSITIONAL,  # *args
                    inspect.Parameter.VAR_KEYWORD,  # **kwargs
                )
            ):
                raise TypeError(
                    f"{cls.__name__}.__init__() has required parameter"
                    f" '{param_name}'. Subclasses of"
                    f" {BaseModule.__name__} must have __init__ methods"
                    " with no required arguments."
                )

    @property
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
            if param_count == 0:
                return f"{self.name}"
            else:
                return f"{self.name} (trainable floats: {param_count:,})"
        elif not ready:
            return f"{self.name} (uninitialized)"
        else:
            return f"{self.name}"

    @abstractmethod
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
        If the module does not have trainable parameters, returns ``0``.
        If the module is not ready, returns ``None``.

        Returns
        -------
            Number of trainable floats in the module, or None if the module
            is not ready.
        """
        if not self.is_ready():
            return None
        try:
            params = self.get_params()
            if params is None or len(jax.tree.leaves(params)) == 0:
                return 0
            # params is a PyTree, so we need to reduce over it
            return jax.tree.reduce(
                lambda s, p: s + (2 if np.iscomplexobj(p) else 1) * p.size,
                params,
                0,
            )
        except Exception as e:
            # reraise
            raise RuntimeError(
                "Error while counting trainable floats: " + str(e)
            ) from e

    @abstractmethod
    def _get_callable(
        self,
    ) -> ModuleCallable:
        """
        Returns a ``jax.jit``-able and ``jax.grad``-able callable that
        represents the module's forward pass.

        This method must be implemented by all subclasses and must return a
        ``jax-jit``-able and ``jax-grad``-able callable in the form of

        .. code-block:: python

            module_callable(
                params: parametricmatrixmodels.typing.Params,
                data: parametricmatrixmodels.typing.Data,
                training: bool,
                state: parametricmatrixmodels.typing.State,
                rng: Any,
            ) -> (
                output: parametricmatrixmodels.typing.Data,
                new_state: parametricmatrixmodels.typing.State,
                )


        That is, all hyperparameters are traced out and the callable depends
        explicitly only on

        * the module's parameters, as a PyTree with leaf nodes as JAX arrays,
        * the input data, as a PyTree with leaf nodes as JAX arrays, each of
            which has shape (num_samples, ...),
        * the training flag, as a boolean,
        * the module's state, as a PyTree with leaf nodes as JAX arrays

        and returns

        * the output data, as a PyTree with leaf nodes as JAX arrays, each of
            which has shape (num_samples, ...),
        * the new module state, as a PyTree with leaf nodes as JAX arrays. The
            PyTree structure must match that of the input state and
            additionally all leaf nodes must have the same shape as the input
            state leaf nodes.

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
        ModuleCallable : Typing for the callable returned by this method.
        Params : Typing for the module parameters.
        Data : Typing for the input and output data.
        State : Typing for the module state.
        """
        raise NotImplementedError(
            "_get_callable method must be implemented in subclasses"
        )

    @jaxtyped(typechecker=beartype)
    def __call__(
        self,
        data: Data,
        /,
        *,
        training: bool = False,
        state: State = (),
        rng: Any = None,
    ) -> Tuple[Data, State]:
        """
        Call the module with the current parameters and given input, state, and
        rng.

        Parameters
        ----------
        data
            PyTree of input arrays of shape (num_samples, ...). Only the first
            dimension (num_samples) is guaranteed to be the same for all input
            arrays.
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
        Params : Typing for the module parameters.
        Data : Typing for the input and output data.
        State : Typing for the module state.
        """
        if not self.is_ready():
            raise ValueError("Module is not ready, call compile() first")

        # get the callable
        func = self._get_callable()

        # call the function with the current parameters, input, training flag,
        # state, and rng
        return func(
            self.get_params(),
            data,
            training,
            state,
            rng,
        )

    @jaxtyped(typechecker=beartype)
    @abstractmethod
    def compile(self, rng: Any, input_shape: DataShape, /) -> None:
        """
        Compile the module to be used with the given input shape.

        This method initializes the module's parameters and state based
        on the input shape and random key.

        This is needed since ``Model`` s are built before the input data is
        given, so before training or inference can be done, the module
        needs to be compiled and each module passes its output shape to the
        next module's ``compile`` method.

        The RNG key is used to initialize random parameters, if needed.

        This is **not** used to trace or jit the module's callable, that is
        done automatically later.

        Parameters
        ----------
        rng
            JAX random key.
        input_shape
            PyTree of input shape tuples, e.g. ``((num_features,),)``, to
            compile the module for. All data passed to the module later must
            have the same PyTree structure and shape in all leaf array
            dimensions except the leading batch dimension.


        Raises
        ------
        NotImplementedError
            If the method is not implemented in the subclass.

        See Also
        --------
        DataShape : Typing for the input shape.
        get_output_shape : Get the output shape of the module
        """
        raise NotImplementedError(
            "compile method must be implemented in subclasses"
        )

    @jaxtyped(typechecker=beartype)
    @abstractmethod
    def get_output_shape(self, input_shape: DataShape, /) -> DataShape:
        """
        Get the output shape of the module given the input shape.

        Parameters
        ----------
        input_shape
            PyTree of input shape tuples, e.g. ``((num_features,),)``, to
            get the output shape for.

        Returns
        -------
            PyTree of output shape tuples, e.g. ``((num_output_features,),)``,
            corresponding to the output shape of the module for the given
            input shape.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the subclass.

        See Also
        --------
        DataShape : Typing for the input and output shape.
        """
        raise NotImplementedError(
            "get_output_shape method must be implemented in subclasses"
        )

    @abstractmethod
    def get_hyperparameters(self) -> HyperParams:
        """
        Get the hyperparameters of the module.

        Hyperparameters are used to configure the module and are not trainable.
        They can be set via `set_hyperparameters`.

        Returns
        -------
            Dictionary containing the hyperparameters of the module.

        See Also
        --------
        set_hyperparameters : Set the hyperparameters of the module.
        HyperParams : Typing for the hyperparameters. Simply an alias for
            Dict[str, Any].

        """
        raise NotImplementedError(
            "get_hyperparameters method must be implemented in subclasses"
        )

    @jaxtyped(typechecker=beartype)
    def set_hyperparameters(self, hyperparameters: HyperParams, /) -> None:
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

        See Also
        --------
        get_hyperparameters : Get the hyperparameters of the module.
        HyperParams : Typing for the hyperparameters. Simply an alias for
            Dict[str, Any].
        """
        if not isinstance(hyperparameters, dict):
            raise TypeError(
                "Hyperparameters must be provided as a dictionary."
            )
        for key, value in hyperparameters.items():
            setattr(self, key, value)

    @abstractmethod
    def get_params(self) -> Params:
        """
        Get the current trainable parameters of the module. If the module has
        no trainable parameters, this method should return an empty tuple.

        Returns
        -------
            PyTree with leaf nodes as JAX arrays representing the module's
            trainable parameters.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the subclass.

        See Also
        --------
        set_params : Set the trainable parameters of the module.
        Params : Typing for the module parameters.

        """
        raise NotImplementedError(
            "get_params method must be implemented in subclasses"
        )

    @jaxtyped(typechecker=beartype)
    @abstractmethod
    def set_params(self, params: Params, /) -> None:
        """
        Set the trainable parameters of the module.

        Parameters
        ----------
        params
            PyTree with leaf nodes as JAX arrays representing the new
            trainable parameters of the module.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the subclass.

        See Also
        --------
        get_params : Get the trainable parameters of the module.
        Params : Typing for the module parameters.
        """
        raise NotImplementedError(
            "set_params method must be implemented in subclasses"
        )

    def get_state(self) -> State:
        """
        Get the current state of the module.

        States are used to store "memory" or other information that is not
        passed between modules, is not trainable, but may be updated during
        either training or inference. e.g. batch normalization state.

        The state is optional, in which case this method should return the
        empty tuple.

        Returns
        -------
            PyTree with leaf nodes as JAX arrays representing the module's
            state.

        See Also
        --------
        set_state : Set the state of the module.
        State : Typing for the module state.
        """
        return ()

    @jaxtyped(typechecker=beartype)
    def set_state(self, state: State, /) -> None:
        """
        Set the state of the module.

        This method is optional.

        Parameters
        ----------
        state
            PyTree with leaf nodes as JAX arrays representing the new
            state of the module.

        See Also
        --------
        get_state : Get the state of the module.
        State : Typing for the module state.
        """
        pass

    @jaxtyped(typechecker=beartype)
    def set_precision(self, prec: Any | str | int, /) -> None:
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

        self.set_params(jax.tree.map(set_param_prec, self.get_params()))
        self.set_state(jax.tree.map(set_param_prec, self.get_state()))

    @jaxtyped(typechecker=beartype)
    def astype(self, dtype: jax.typing.DTypeLike, /) -> "BaseModule":
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

    def serialize(self) -> Dict[str, Any]:
        """
        Serialize the module to a dictionary.

        This method returns a dictionary representation of the module,
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
            "name": self.name,
            "hyperparameters": self.get_hyperparameters(),
            "params": self.get_params(),
            "state": self.get_state(),
            "package_version": pmm.__version__,
            "module_version": self.__version__,
        }

    @jaxtyped(typechecker=beartype)
    def deserialize(
        self, data: Dict[str, Any], /, *, strict_package_version=False
    ) -> None:
        """
        Deserialize the module from a dictionary.

        This method sets the module's parameters and state based on the
        provided dictionary.

        The default implementation expects the dictionary to contain
        the module's name, trainable parameters, and state.

        Parameters
        ----------
        data
            Dictionary containing the serialized module data.

        strict_package_version
            If True, raises an error if the package version used to
            serialize the model does not match the current package version.
            Default is False.

        Raises
        ------
        ValueError
            If the serialized data does not contain the expected keys or
            if the version of the serialized data is not compatible with
            with the current package version.
        """

        # read the version of the package this module was serialized with
        current_version = parse(pmm.__version__)
        package_version = parse(str(data["package_version"]))

        if current_version != package_version:
            if strict_package_version:
                raise ValueError(
                    "Version mismatch when deserializing module "
                    f"'{self.name}': serialized with version "
                    f"{package_version}, current version is "
                    f"{current_version}."
                )

        module_version = parse(str(data["module_version"]))
        current_module_version = parse(self.__version__)
        if module_version != current_module_version:
            # upgrade the data to the current version
            data = self.upgrade(data)

        # set the hyperparameters
        self.set_hyperparameters(data.get("hyperparameters", {}))

        # if there are trainable parameters, set them
        params = data.get("params", None)
        if params is not None:
            self.set_params(params)

        # if there are states, set them
        state = data.get("state", None)
        if state is not None:
            self.set_state(state)

    @jaxtyped(typechecker=beartype)
    def upgrade(self, data: Dict[str, Any], /) -> Dict[str, Any]:
        """
        Upgrade serialized module data to the current version.

        This method can be overridden by subclasses to implement custom
        upgrade logic when the module's serialization format changes between
        versions.

        The default implementation simply returns the input data unchanged.

        Parameters
        ----------
        data
            Dictionary containing the serialized module data.

        Returns
        -------
            Upgraded dictionary containing the serialized module data.
        """
        return data

    @jaxtyped(typechecker=beartype)
    def copy(self) -> "BaseModule":
        """
        Create a deep copy of the module.

        Returns
        -------
            A deep copy of the module.
        """
        # serialize and deserialize to create a deep copy
        data = self.serialize()
        new_module = self.__class__()
        new_module.deserialize(data)
        return new_module

    deepcopy = copy  # alias for copy
