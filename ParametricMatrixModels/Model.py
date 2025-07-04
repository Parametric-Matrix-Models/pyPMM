import jax.numpy as np
from jax import jit, vmap, random
from .Training import train, make_loss_fn
from .Modules import BaseModule
from typing import Callable, List, Optional, Tuple, Any, Dict, Union
import sys


class Model(object):
    """
    Model class built from a list of modules.
    """

    def __repr__(self) -> str:

        num_trainable_floats = [
            module.get_num_trainable_floats() for module in self.modules
        ]
        if None in num_trainable_floats:
            num_trainable_floats = "(uninitialized)"
        else:
            num_trainable_floats = (
                f"(trainable floats: " f"{sum(num_trainable_floats)})"
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
            rep += f"\nModule {i}: {module}" + (
                f" -> {input_shape}" if input_shape else ""
            )
        return rep

    def __init__(self, modules: List[BaseModule] = []) -> None:
        """
        Initialize the model with the input shape and a list of modules.

        Parameters
        ----------
            modules : List[BaseModule], optional
                List of modules to initialize the model with. Default is an
                empty list.
        """
        self.modules = modules
        self.reset()

    def reset(self) -> None:
        self.input_shape = None
        self.output_shape = None
        self.ready = False
        self.parameter_counts = None
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

    def __getitem__(self, index: int) -> BaseModule:
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
        if index < 0 or index >= len(self.modules):
            raise IndexError("Index out of range.")
        return self.modules[index]

    def compile(self, rngkey, input_shape: Tuple[int, ...]) -> None:
        """
        Compile the model for training by compiling each module.

        Parameters
        ----------
            rngkey : Any
                Random key for initializing the model parameters. JAX PRNGKey
            input_shape : Tuple[int, ...]
                Shape of the input array, excluding the batch size.
                For example, (input_features,) for a 1D input or
                (input_height, input_width, input_channels) for a 3D input.
        """
        self.input_shape = input_shape
        for module in self.modules:
            module.compile(rngkey, input_shape)
            input_shape = module.get_output_shape(input_shape)
        self.output_shape = input_shape

        # get number of parameter arrays for each module
        self.parameter_counts = [
            len(module.get_params()) for module in self.modules
        ]

        self.ready = True

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
        np.ndarray,
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

        # TODO:
        # add training, state, and rng parameters to the callables
        # and therefore to the training

        # parameter tuple must be flattened, so we'll need to iterate over the
        # parameter counts
        # jax will unroll this loop
        def model_callable(
            params: Tuple[np.ndarray],
            X: np.ndarray,
        ) -> np.ndarray:
            param_index = 0
            for idx, count in enumerate(self.parameter_counts):
                module_params = tuple(
                    params[param_index : param_index + count]
                )
                param_index += count
                X, _ = module_callables[idx](module_params, X, False, (), None)

            return X

        return model_callable

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Call the model with the input array.

        Parameters
        ----------
            X : np.ndarray
                Input array of shape (batch_size, <input feature axes>).
                For example, (batch_size, input_features) for a 1D input or
                (batch_size, input_height, input_width, input_channels) for a
                3D input.

        Returns
        -------
            np.ndarray
                Output array of shape (batch_size, <output feature axes>).
                For example, (batch_size, output_features) for a 1D output or
                (batch_size, output_height, output_width, output_channels) for
                a 3D output.
        """
        if not self.ready:
            raise RuntimeError("Model is not ready. Call compile() first.")

        if self.callable is None:
            self.callable = jit(self._get_callable())

        return self.callable(self.get_params(), X)

    def train(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        Y_unc: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        Y_val: Optional[np.ndarray] = None,
        Y_val_unc: Optional[np.ndarray] = None,
        loss_fn: str = "mse",
        lr: float = 1e-3,
        batch_size: int = 32,
        num_epochs: int = 100,
        convergence_threshold: float = 1e-12,
        early_stopping_patience: int = 10,
        early_stopping_tolerance: float = 1e-6,
        # advanced options
        callback: Optional[Callable] = None,
        unroll: Optional[int] = None,
        verbose: bool = True,
        seed: Optional[int] = None,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        clip: float = 1e3,
    ) -> None:

        # check if the model is ready
        if not self.ready:
            self.compile(random.PRNGKey(seed or 0), X.shape[1:])

        # check dimensions
        input_shape = X.shape[1:]
        if input_shape != self.input_shape:
            raise ValueError(
                f"Input shape {input_shape} does not match model input shape "
                f"{self.input_shape}."
            )
        if Y is not None and Y.shape[1:] != self.output_shape:
            raise ValueError(
                f"Output shape {Y.shape[1:]} does not match model output shape "
                f"{self.output_shape}."
            )
        if Y is not None and X_val is not None and Y_val is None:
            raise ValueError(
                "Validation data Y_val must be provided if validation input "
                "X_val is provided for supervised training."
            )

        # get callable, not jitted since the training function will
        # handle that
        if self.callable is None:
            self.callable = self._get_callable()

        # make the loss function
        loss_fn = make_loss_fn(loss_fn, lambda x, p: self.callable(p, x))

        # train the model
        final_params, final_epoch, final_states = train(
            init_params=self.get_params(),
            loss_fn=loss_fn,
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
            seed=seed,
            b1=b1,
            b2=b2,
            eps=eps,
            clip=clip,
            real=False,
        )

        # set the final parameters
        self.set_params(final_params)

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

        return {
            "module_typenames": module_typenames,
            "module_modules": module_modules,
            "module_fulltypenames": module_fulltypenames,
            "module_names": module_names,
            "serialized_modules": serialized_modules,
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

        filename = filename if filename.endswith(".pmm") else filename + ".pmm"
        np.savez(filename, **data)

    def load(self, filename: str) -> None:
        """
        Load the model from a file.

        Parameters
        ----------
            filename : str
                Name of the file to load the model from.
        """
        filename = filename if filename.endswith(".pmm") else filename + ".pmm"
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
