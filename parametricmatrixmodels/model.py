import random
import sys
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as np
import numpy as onp
from packaging.version import parse

import parametricmatrixmodels as pmm

from .modules import BaseModule
from .training import make_loss_fn, train


class Model(object):
    """
    Model class built from a list of modules.
    """

    def __repr__(self) -> str:

        trainable_floats_num = self.get_num_trainable_floats()

        # get number of modules in order to reserve whitespace
        num_modules = len(self.modules)
        mod_idx_width = len(str(num_modules - 1))

        if trainable_floats_num is None:
            num_trainable_floats = "(uninitialized)"
        else:
            num_trainable_floats = (
                f"(trainable floats: {trainable_floats_num:,})"
            )

        rep = (
            f"Model(input_shape={self.input_shape}, "
            f"output_shape={self.output_shape}, ready={self.ready}) "
            f"{num_trainable_floats}\n"
        )
        input_shape = self.input_shape if self.input_shape else None
        for i, module in enumerate(self.modules):
            input_shape = (
                module.get_output_shape(input_shape) if input_shape else None
            )
            comment = module.name().startswith("#")
            rep += f"\n{i:>{mod_idx_width}}: {module}" + (
                f" -> {input_shape}" if input_shape and not comment else ""
            )
        return rep

    def __init__(
        self, modules: List[BaseModule] = [], rng: Any = None
    ) -> None:
        """
        Initialize the model with the input shape and a list of modules.

        Parameters
        ----------
            modules : List[BaseModule], optional
                List of modules to initialize the model with. Default is an
                empty list.
            rng : Any, optional
                Initial random key for the model. Default is None. If None, a
                new random key will be generated using JAX's random.PRNGKey. If
                an integer is provided, it will be used as the seed to create
                the key.
        """
        self.modules = modules
        if rng is None:
            self.rng = jax.random.key(random.randint(0, 2**32 - 1))
        elif isinstance(rng, int):
            self.rng = jax.random.key(rng)
        else:
            self.rng = rng
        self.reset()

    def get_num_trainable_floats(self) -> Optional[int]:
        num_trainable_floats = [
            module.get_num_trainable_floats() for module in self.modules
        ]
        if None in num_trainable_floats:
            return None
        else:
            return sum(num_trainable_floats)

    def reset(self) -> None:
        self.input_shape = None
        self.output_shape = None
        self.ready = False
        self.parameter_counts = None
        self.state_counts = None
        self.callable = None

    def append_module(self, module: BaseModule) -> None:
        """
        Append a module to the model.

        Parameters
        ----------
            module : BaseModule
                Module to append to the model.
        """
        self.modules.append(module)
        self.reset()

    def prepend_module(self, module: BaseModule) -> None:
        """
        Prepend a module to the model.

        Parameters
        ----------
            module : BaseModule
                Module to prepend to the model.
        """
        self.modules.insert(0, module)
        self.reset()

    def insert_module(self, module: BaseModule, index: int) -> None:
        """
        Insert a module at the given index in the model.

        Parameters
        ----------
            module : BaseModule
                Module to insert into the model.
            index : int
                Index at which to insert the module.
        """
        self.modules.insert(index, module)
        self.reset()

    add = append_module
    put = prepend_module
    insert = insert_module

    def remove_module(self, index: int) -> None:
        """
        Remove a module from the model at the given index.

        Parameters
        ----------
            index : int
                Index of the module to remove.
        """
        if index < 0 or index >= len(self.modules):
            raise IndexError("Index out of range.")
        del self.modules[index]
        self.reset()

    def pop_module(self) -> BaseModule:
        """
        Pop the last module from the model.

        Returns
        -------
            BaseModule
                The last module in the model
        """
        if not self.modules:
            raise IndexError("No modules to pop.")
        module = self.modules.pop()
        self.reset()
        return module

    def __getitem__(
        self, key: Union[int, np.ndarray, slice]
    ) -> Union[List[BaseModule], BaseModule]:
        """
        Get the module at the given index.

        Parameters
        ----------
            index : int
                Index of the module to retrieve.

        Returns
        -------
            BaseModule
                The module at the specified index.
        """
        if isinstance(key, np.ndarray):
            if key.ndim > 1:
                raise ValueError(
                    "Index array must be 1D. Use a boolean mask or a 1D array."
                )
            # the key can either be an index array or a boolean mask
            if key.dtype == bool:
                if len(key) != len(self.modules):
                    raise ValueError(
                        "Boolean mask length must match the number of modules."
                    )
                indices = np.where(key)[0]
                return [self.modules[i] for i in indices]
            elif key.dtype == int:
                indices = key.flatten()
                return [self.modules[i] for i in indices]
            else:
                raise ValueError(
                    "Index array must be of type int or bool. "
                    f"Got {key.dtype}."
                )
        elif isinstance(key, slice):
            # return a slice of the modules
            return self.modules[key]
        elif isinstance(key, int):
            if key < 0 or key >= len(self.modules):
                raise IndexError("Index out of range.")
            return self.modules[key]
        else:
            raise TypeError(
                "Index must be an integer, a slice, or a 1D numpy array. "
                f"Got {type(key)}."
            )

    def compile(
        self,
        rngkey: Optional[Union[Any, int]],
        input_shape: Tuple[int, ...],
        verbose: bool = False,
    ) -> None:
        """
        Compile the model for training by compiling each module.

        Parameters
        ----------
            rngkey : Union[Any, int]
                Random key for initializing the model parameters. JAX PRNGKey
                or integer seed.
            input_shape : Tuple[int, ...]
                Shape of the input array, excluding the batch size.
                For example, (input_features,) for a 1D input or
                (input_height, input_width, input_channels) for a 3D input.
            verbose : bool, optional
                Print debug information during compilation. Default is False.
        """

        if rngkey is None:
            rngkey = random.randint(0, 2**32 - 1)

        if isinstance(rngkey, int):
            rngkey = jax.random.key(rngkey)

        if verbose:
            print(
                f"Compiling model with input shape {input_shape} and "
                f"{len(self.modules)} modules."
            )

        self.input_shape = input_shape
        for i, module in enumerate(self.modules):
            rngkey, modrng = jax.random.split(rngkey)
            module.compile(modrng, input_shape)
            input_shape = module.get_output_shape(input_shape)
            if verbose:
                print(f"({i}) {module.name()} output shape: {input_shape}")
        self.output_shape = input_shape

        # get number of parameter arrays for each module
        self.parameter_counts = [
            len(module.get_params()) for module in self.modules
        ]
        self.state_counts = [
            len(module.get_state()) for module in self.modules
        ]

        self.ready = True

    def get_output_shape(
        self, input_shape: Tuple[int, ...]
    ) -> Tuple[int, ...]:
        """
        Get the output shape of the model given an input shape.

        Parameters
        ----------
            input_shape : Tuple[int, ...]
                Shape of the input array, excluding the batch size.
                For example, (input_features,) for a 1D input or
                (input_height, input_width, input_channels) for a 3D input.

        Returns
        -------
            Tuple[int, ...]
                Shape of the output array after passing through the model.
        """
        for module in self.modules:
            input_shape = module.get_output_shape(input_shape)

        return input_shape

    def get_params(self) -> Tuple[np.ndarray, ...]:
        """
        Get the parameters of the model as a Tuple of numpy arrays.

        Returns
        -------
            Tuple[np.ndarray, ...]
                numpy arrays representing the parameters of the model. The
                order of the parameters should match the order in
                which they are used in the _get_callable method.
        """
        if not self.ready:
            raise RuntimeError("Model is not ready. Call compile() first.")

        # parameter tuple must be flat
        return tuple(
            param for module in self.modules for param in module.get_params()
        )

    def set_params(self, params: Tuple[np.ndarray, ...]) -> None:
        """
        Set the parameters of the model from a Tuple of numpy arrays.

        Parameters
        ----------
            params: Tuple[np.ndarray, ...]
                numpy arrays representing the parameters of the model. The
                order of the parameters should match the order in which
                they are used in the _get_callable method.
        """
        if not self.ready:
            raise RuntimeError("Model is not ready. Call compile() first.")

        if len(params) != sum(self.parameter_counts):
            raise ValueError(
                f"Expected {sum(self.parameter_counts)} parameters, "
                f"but got {len(params)}."
            )

        # set parameters for each module
        param_index = 0
        for module in self.modules:
            count = len(module.get_params())
            module.set_params(params[param_index : param_index + count])
            param_index += count

    def get_state(self) -> Tuple[np.ndarray, ...]:
        """
        Get the state of the model as a Tuple of numpy arrays.

        Returns
        -------
            Tuple[np.ndarray, ...]
                numpy arrays representing the state of the model. The order of
                the states should match the order in which they are used in
                the _get_callable method.
        """
        if not self.ready:
            raise RuntimeError("Model is not ready. Call compile() first.")

        # state tuple must be flat
        return tuple(
            state for module in self.modules for state in module.get_state()
        )

    def set_state(self, state: Tuple[np.ndarray, ...]) -> None:
        """
        Set the state of the model from a Tuple of numpy arrays.

        Parameters
        ----------
            state: Tuple[np.ndarray, ...]
                numpy arrays representing the state of the model. The order of
                the states should match the order in which they are used in
                the _get_callable method.
        """
        if not self.ready:
            raise RuntimeError("Model is not ready. Call compile() first.")

        if len(state) != sum(self.state_counts):
            raise ValueError(
                f"Expected {sum(self.state_counts)} states, "
                f"but got {len(state)}."
            )

        # set state for each module
        state_index = 0
        for module in self.modules:
            count = len(module.get_state())
            module.set_state(state[state_index : state_index + count])
            state_index += count

    def get_rng(self) -> Any:
        return self.rng

    def set_rng(self, rng: Any) -> None:
        """
        Set the random key for the model.

        Parameters
        ----------
            rng : Any
                Random key to set for the model. JAX PRNGKey or an integer seed
        """
        if isinstance(rng, int):
            self.rng = jax.random.key(rng)
        else:
            self.rng = rng

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
        if not self.ready:
            raise RuntimeError("Model is not ready. Call compile() first.")

        # get the callables for each module
        module_callables = [module._get_callable() for module in self.modules]

        # parameter tuple must be flattened, so we'll need to iterate over the
        # parameter counts
        # state tuple must also be flattened, so we'll need to iterate over the
        # state counts
        # jax will unroll this loop
        def model_callable(
            params: Tuple[np.ndarray],
            X: np.ndarray,
            training: bool = False,
            states: Tuple[np.ndarray, ...] = (),
            rng: Any = None,  # absolutely not optonal
        ) -> np.ndarray:
            param_index = 0
            state_index = 0
            # split rng key into a key for each module
            rngs = jax.random.split(rng, len(self.modules))
            for idx, param_count in enumerate(self.parameter_counts):
                state_count = self.state_counts[idx]
                module_params = tuple(
                    params[param_index : param_index + param_count]
                )
                module_states = tuple(
                    states[state_index : state_index + state_count]
                )
                module_rng = rngs[idx]

                X, new_module_states = module_callables[idx](
                    module_params, X, training, module_states, module_rng
                )
                # update the states
                states = (
                    states[:state_index]
                    + new_module_states
                    + states[state_index + state_count :]
                )

                # increment indices
                param_index += param_count
                state_index += state_count

            return X, states

        return model_callable

    def __call__(
        self,
        X: np.ndarray,
        dtype: Optional[Any] = np.float64,
        rng: Any = None,
        return_state: bool = False,
        update_state: bool = False,
    ) -> np.ndarray:
        """
        Call the model with the input array.

        Parameters
        ----------
            X : np.ndarray
                Input array of shape (batch_size, <input feature axes>).
                For example, (batch_size, input_features) for a 1D input or
                (batch_size, input_height, input_width, input_channels) for a
                3D input.
            dtype : Optional[Any], optional
                Data type of the output array. Default is jax.numpy.float64.
                It is strongly recommended to perform training in single
                precision (float32 and complex64) and inference with double
                precision inputs (float64, the default here) with single
                precision weights.
            rng : Any, optional
                JAX random key for stochastic modules. Default is None.
                If None, the saved rng key will be used if it exists, which
                would be the final rng key from the last training run. If an
                integer is provided, it will be used as the seed to create a
                new JAX random key.
            return_state : bool, optional
                If True, the model will return the state of the model after
                evaluation. Default is False.
            update_state : bool, optional
                If True, the model will update the state of the model after
                evaluation. Default is False.

        Returns
        -------
            np.ndarray
                Output array of shape (batch_size, <output feature axes>).
                For example, (batch_size, output_features) for a 1D output or
                (batch_size, output_height, output_width, output_channels) for
                a 3D output.
            Tuple[np.ndarray, ...], optional
                If return_state is True, the model will also return the state
                of the model as a Tuple of numpy arrays. The order of the
                states will match the order in which they are used in the
                _get_callable method.
        """
        if not self.ready:
            raise RuntimeError("Model is not ready. Call compile() first.")

        if self.callable is None:
            self.callable = jax.jit(
                self._get_callable(), static_argnames=["training"]
            )

        X_ = X.astype(dtype)

        # make sure the dtype was converted, issue a warning if not
        if X_.dtype != dtype:
            warnings.warn(
                "While performing inference with model: "
                f"Requested dtype ({dtype}) was not successfully applied. "
                "This is most likely due to JAX_ENABLE_X64 not being set. "
                "See accompanying JAX warning for more details.",
                UserWarning,
            )

        if rng is None:
            rng = self.get_rng()
        elif isinstance(rng, int):
            rng = jax.random.key(rng)

        out, new_state = self.callable(
            self.get_params(), X_, False, self.get_state(), rng
        )

        if update_state:
            warnings.warn(
                "update_state is True. This is an uncommon use case, make "
                "sure you know what you are doing.",
                UserWarning,
            )
            self.set_state(new_state)
        if return_state:
            return out, new_state
        else:
            return out

    # alias for __call__ method
    predict = __call__

    def set_precision(self, prec: Union[np.dtype, str, int]) -> None:
        """
        Set the precision of the model parameters and states.

        Parameters
        ----------
            prec : Union[np.dtype, str, int]
                Precision to set for the model parameters and states.
                Valid options are:
                    [for 32-bit precision (all options are equivalent)]
                    - np.float32, np.complex64, "float32", "complex64"
                    - "single", "f32", "c64", 32
                    [for 64-bit precision (all options are equivalent)]
                    - np.float64, np.complex128, "float64", "complex128"
                    - "double", "f64", "c128", 64
        """
        if not self.ready:
            raise RuntimeError("Model is not ready. Call compile() first.")

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

        for module in self.modules:
            module.set_precision(prec)

    # alias for set_precision method that returns self
    def astype(self, dtype: Union[np.dtype, str]) -> "Model":
        """
        Convenience wrapper to set_precision using the dtype argument, returns
        self.
        """
        self.set_precision(dtype)
        return self

    def train(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        Y_unc: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        Y_val: Optional[np.ndarray] = None,
        Y_val_unc: Optional[np.ndarray] = None,
        loss_fn: Union[str, Callable] = "mse",
        lr: float = 1e-3,
        batch_size: int = 32,
        num_epochs: int = 100,
        convergence_threshold: float = 1e-12,
        early_stopping_patience: int = 10,
        early_stopping_tolerance: float = 1e-6,
        # advanced options
        initialization_seed: Optional[int] = None,
        callback: Optional[Callable] = None,
        unroll: Optional[int] = None,
        verbose: bool = True,
        batch_seed: Optional[int] = None,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        clip: float = 1e3,
    ) -> None:

        # check if the model is ready
        if not self.ready:
            initialization_seed = initialization_seed or random.randint(
                0, 2**32 - 1
            )
            self.compile(jax.random.key(initialization_seed), X.shape[1:])

        # check if any of the model parameters are double precision and give a
        # warning if so
        if any(
            (
                np.issubdtype(np.asarray(param).dtype, np.float64)
                or np.issubdtype(np.asarray(param).dtype, np.complex128)
            )
            for param in self.get_params()
        ):
            warnings.warn(
                "Some parameters are double precision. "
                "This may lead to significantly slower training on certain "
                "backends. It is strongly recommended to use single precision "
                "(float32/complex64) parameters for training. Set the "
                "precision of the model with Model.set_precision.",
                UserWarning,
            )

        # check dimensions
        input_shape = X.shape[1:]
        if input_shape != self.input_shape:
            raise ValueError(
                f"Input shape {input_shape} does not match model input shape "
                f"{self.input_shape}."
            )
        if Y is not None and Y.shape[1:] != self.output_shape:
            raise ValueError(
                f"Output shape {Y.shape[1:]} does not match model output "
                f"shape {self.output_shape}."
            )
        if Y is not None and X_val is not None and Y_val is None:
            raise ValueError(
                "Validation data Y_val must be provided if validation input "
                "X_val is provided for supervised training."
            )

        # get callable, not jitted since the training function will
        # handle that
        callable_ = self._get_callable()

        # make the loss function
        if isinstance(loss_fn, str):
            loss_fn_ = make_loss_fn(
                loss_fn, lambda x, p, t, s, r: callable_(p, x, t, s, r)
            )
        else:
            # if the loss function is already a callable, we wrap it with the
            # model callable
            # whether or not Y and Y_unc are provided changes the signature
            # of the loss function
            if Y is not None and Y_unc is not None:
                # the loss function should be
                # loss_fn(X, Y, Y_unc, Y_pred) -> err
                def loss_fn_(X, Y, Y_unc, params, training, states, rng):
                    Y_pred, new_states = callable_(
                        params, X, training, states, rng
                    )
                    err = loss_fn(X, Y, Y_unc, Y_pred)
                    return err, new_states

            elif Y is not None and Y_unc is None:
                # the loss function should be
                # loss_fn(X, Y, Y_pred) -> err
                def loss_fn_(X, Y, params, training, states, rng):
                    Y_pred, new_states = callable_(
                        params, X, training, states, rng
                    )
                    err = loss_fn(X, Y, Y_pred)
                    return err, new_states

            elif Y is None and Y_unc is None:
                # the loss function should be
                # loss_fn(X, pred) -> err
                # (unsupervised training)
                def loss_fn_(X, params, training, states, rng):
                    pred, new_states = callable_(
                        params, X, training, states, rng
                    )
                    err = loss_fn(X, pred)
                    return err, new_states

            else:
                raise ValueError(
                    "Invalid loss function signature. "
                    "If Y and Y_unc are provided, the loss function should be "
                    "loss_fn(X, Y, Y_unc, Y_pred) -> err. "
                    "If only Y is provided, it should be "
                    "loss_fn(X, Y, Y_pred) -> err. "
                    "If neither are provided, it should be "
                    "loss_fn(X, pred) -> err."
                )

        # train the model
        (
            final_params,
            final_model_states,
            final_model_rng,
            final_epoch,
            final_adam_states,
        ) = train(
            init_params=self.get_params(),
            init_states=self.get_state(),
            init_rng=self.get_rng(),
            loss_fn=loss_fn_,
            X=X,
            Y=Y,
            Y_unc=Y_unc,
            X_val=X_val,
            Y_val=Y_val,
            Y_val_unc=Y_val_unc,
            lr=lr,
            batch_size=batch_size,
            num_epochs=num_epochs,
            convergence_threshold=convergence_threshold,
            early_stopping_patience=early_stopping_patience,
            early_stopping_tolerance=early_stopping_tolerance,
            callback=callback,
            unroll=unroll,
            verbose=verbose,
            batch_seed=batch_seed,
            b1=b1,
            b2=b2,
            eps=eps,
            clip=clip,
            real=False,
        )

        # set the final parameters
        self.set_params(final_params)
        # set the final state
        self.set_state(final_model_states)
        # set the final rng
        self.set_rng(final_model_rng)

    def serialize(self) -> Dict[str, Union[Any, Dict[str, Any]]]:
        """
        Serialize the model to a dictionary. This is done by serializing the
        model's parameters/metadata and then serializing each module.

        Returns
        -------
            Dict[str, Union[Any, Dict[str, Any]]]
        """

        module_fulltypenames = [str(type(module)) for module in self.modules]
        module_typenames = [
            module.__class__.__name__ for module in self.modules
        ]
        module_modules = [module.__module__ for module in self.modules]
        module_names = [module.name() for module in self.modules]

        serialized_modules = [module.serialize() for module in self.modules]

        # serialize rng key
        key_data = jax.random.key_data(self.get_rng())

        return {
            "module_typenames": module_typenames,
            "module_modules": module_modules,
            "module_fulltypenames": module_fulltypenames,
            "module_names": module_names,
            "serialized_modules": serialized_modules,
            "key_data": key_data,
            "package_version": pmm.__version__,
        }

    def deserialize(self, data: Dict[str, Any]) -> None:
        """
        Deserialize the model from a dictionary. This is done by deserializing
        the model's parameters/metadata and then deserializing each module.

        Parameters
        ----------
            data : Dict[str, Any]
                Dictionary containing the serialized model data.
        """
        self.reset()

        # read the version of the package this model was serialized with
        current_version = parse(pmm.__version__)
        package_version = parse(str(data["package_version"]))

        if current_version != package_version:
            # in the future, we will issue DeprecationWarnings or Errors if the
            # version is unsupported
            # or possibly handle version-specific deserialization
            pass

        module_typenames = data["module_typenames"]
        module_modules = data["module_modules"]

        # initialize the modules
        self.modules = [
            getattr(sys.modules[module_module], module_typename)()
            for module_typename, module_module in zip(
                module_typenames, module_modules
            )
        ]

        # deserialize the modules
        for module, serialized_module in zip(
            self.modules, data["serialized_modules"]
        ):
            module.deserialize(serialized_module)

        # deserialize the rng key
        key = jax.random.wrap_key_data(data["key_data"])
        self.set_rng(key)

    def save(self, filename: str) -> None:
        """
        Save the model to a file.

        Parameters
        ----------
            filename : str
                Name of the file to save the model to.
        """

        # if everything serializes correctly, we can save the model with just
        # savez
        data = self.serialize()

        filename = filename if filename.endswith(".npz") else filename + ".npz"
        np.savez(filename, **data)

    def save_compressed(self, filename: str) -> None:
        """
        Save the model to a compressed file.

        Parameters
        ----------
            filename : str
                Name of the file to save the model to.
        """
        # if everything serializes correctly, we can save the model with just
        # savez_compressed
        data = self.serialize()

        filename = filename if filename.endswith(".npz") else filename + ".npz"

        # jax.numpy doesn't have savez_compressed, so we use numpy
        onp.savez_compressed(filename, **data)

    def load(self, filename: str) -> None:
        """
        Load the model from a file. Supports both compressed and uncompressed

        Parameters
        ----------
            filename : str
                Name of the file to load the model from.
        """
        filename = filename if filename.endswith(".npz") else filename + ".npz"
        # jax numpy load supports both compressed and uncompressed npz files
        data = np.load(filename, allow_pickle=True)

        # deserialize the model
        self.deserialize(data)

    @classmethod
    def from_file(cls, filename: str) -> "Model":
        """
        Load a model from a file and return an instance of the Model class.

        Parameters
        ----------
            filename : str
                Name of the file to load the model from.

        Returns
        -------
            Model
                An instance of the Model class with the loaded parameters.
        """
        model = cls()
        model.load(filename)
        return model
