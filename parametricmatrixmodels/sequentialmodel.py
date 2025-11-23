from __future__ import annotations

import random
import uuid

import jax
from beartype import beartype
from jaxtyping import jaxtyped

from .model import Model
from .model_util import (
    ModelCallable,
    ModelModules,
    ModelParams,
    ModelState,
)
from .modules import BaseModule
from .typing import (
    Any,
    Data,
    DataShape,
    Dict,
    List,
    ModuleCallable,
    Params,
    PyTree,
    State,
    Tuple,
)


class SequentialModel(Model):
    r"""
    A simple sequential model that chains modules (or other models) together
    sequentially.

    For confidence intervals or uncertainty quantification, wrap a trained
    model with ``ConformalModel``.

    See Also
    --------
    jax.tree
        PyTree utilities and concepts in JAX.
    Model
        Abstract base class for all models.
    NonsequentialModel
        A model that allows for non-sequential connections between modules.
    ConformalModel
        Wrap a trained model to produce confidence intervals.
    """

    def __init__(
        self,
        modules: ModelModules | BaseModule | None = None,
        rng: Any | int | None = None,
    ) -> None:
        r"""
        Initialize a sequential model with a PyTree of modules and a
        random key.

        For sequential models, the modules are applied in the order they appear
        in the flattened PyTree.

        Since insertion order is preserved in dictionaries since Python 3.7,
        using a dictionary to specify modules is a convenient way to name
        modules while controlling their application order.

        If a sequential model is initialized with a dictionary, the
        ``append``/``prepend``-style methods will will use UUIDs to name new
        modules if the optional key argument is not provided.

        Parameters
        ----------
            modules
                module(s) to initialize the model with. Default is None, which
                will become an empty list.
            rng
                Initial random key for the model. Default is None. If None, a
                new random key will be generated using JAX's ``random.key``. If
                an integer is provided, it will be used as the seed to create
                the key.

        See Also
        --------
        ModelModules
            Type alias for a PyTree of modules in a model.
        jax.random.key
            JAX function to create a random key.
        jax.tree.flatten
            JAX function to flatten a PyTree, which determines the order of
            module application in a sequential model.
        jax.tree.leaves
            JAX function to flatten a PyTree without returning the structure.
            Equivalent to ``jax.tree.flatten(x)[0]``.
        """
        # no custom initialization needed for sequential model
        super().__init__(modules=modules, rng=rng)

    def compile(
        self,
        rng: Any | int | None,
        input_shape: DataShape,
        verbose: bool = False,
    ) -> None:
        r"""
        Compile the model for training by compiling each module. Must be
        implemented by all subclasses.

        Parameters
        ----------
            rng
                Random key for initializing the model parameters. JAX PRNGKey
                or integer seed.
            input_shape
                Shape of the input array, excluding the batch size.
                For example, (input_features,) for a 1D input or
                (input_height, input_width, input_channels) for a 3D input.
            verbose
                Print debug information during compilation. Default is False.
        """

        if rng is None:
            rng = jax.random.key(random.randint(0, 2**32 - 1))
        elif isinstance(rng, int):
            rng = jax.random.key(rng)

        if verbose:
            print(f"Compiling {self.name()} for input shape {input_shape}.")

        self.input_shape = input_shape

        for i, module in enumerate(jax.tree.leaves(self.modules)):
            rng, modrng = jax.random.split(rng)
            module.compile(modrng, input_shape)
            input_shape = module.get_output_shape(input_shape)
            if verbose:
                print(f"  {i}: {module.name()} output shape: {input_shape}")

        self.output_shape = input_shape

    def get_output_shape(self, input_shape: DataShape) -> DataShape:
        r"""
        Get the output shape of the model given an input shape. Must be
        implemented by all subclasses.

        Parameters
        ----------
            input_shape
                Shape of the input, excluding the batch dimension.
                For example, (input_features,) for 1D bare-array input, or
                (input_height, input_width, input_channels) for 3D bare-array
                input, [(input_features1,), (input_features2,)] for a List
                (PyTree) of 1D arrays, etc.

        Returns
        -------
            output_shape
                Shape of the output after passing through the model.
        """
        if self.is_ready():
            return self.output_shape
        else:
            shape = input_shape
            for module in jax.tree.leaves(self.modules):
                shape = module.get_output_shape(shape)
            return shape

    def _get_callable(
        self,
    ) -> ModelCallable:
        r"""
        Returns a ``jax.jit``-able and ``jax.grad``-able callable that
        represents the model's forward pass.

        This must be implemented by all subclasses.

        This method must be implemented by all subclasses and must return a
        ``jax-jit``-able and ``jax-grad``-able callable in the form of

        .. code-block:: python

            model_callable(
                params: parametricmatrixmodels.model_util.ModelParams,
                data: parametricmatrixmodels.typing.Data,
                training: bool,
                state: parametricmatrixmodels.model_util.ModelState,
                rng: Any,
            ) -> (
                output: parametricmatrixmodels.typing.Data,
                new_state: parametricmatrixmodels.model_util.ModelState,
                )


        That is, all hyperparameters are traced out and the callable depends
        explicitly only on

        * the model's parameters, as a PyTree with leaf nodes as JAX arrays,
        * the input data, as a PyTree with leaf nodes as JAX arrays, each of
            which has shape (num_samples, ...),
        * the training flag, as a boolean,
        * the model's state, as a PyTree with leaf nodes as JAX arrays

        and returns

        * the output data, as a PyTree with leaf nodes as JAX arrays, each of
            which has shape (num_samples, ...),
        * the new model state, as a PyTree with leaf nodes as JAX arrays. The
            PyTree structure must match that of the input state and
            additionally all leaf nodes must have the same shape as the input
            state leaf nodes.

        The training flag will be traced out, so it doesn't need to be jittable

        Returns
        -------
            A callable that takes the model's parameters, input data,
            training flag, state, and rng key and returns the output data and
            new state.

        See Also
        --------
        __call__ : Calls the model with the current parameters and
            given input, state, and rng.
        ModelCallable : Typing for the callable returned by this method.
        Params : Typing for the model parameters.
        Data : Typing for the input and output data.
        State : Typing for the model state.
        """

        if not self.is_ready():
            raise RuntimeError(
                f"{self.name()} is not ready. Call compile() first."
            )

        # get the callables for each module and put them in a PyTree with the
        # same structure
        module_callables = jax.tree.map(
            lambda m: m._get_callable(), self.modules
        )
        modules_structure = jax.tree.structure(self.modules)

        @jaxtyped(typechecker=beartype)
        def sequential(
            carry: Tuple[Data, ModelState],
            module_data: Tuple[ModuleCallable, Params, State, Any],
            training: bool,
        ) -> Tuple[Data, ModelState]:
            # carry is (data, [flattened model state])
            # module_data is (module_callable, module_params,
            #                 module_state, module_rng)
            (
                module_callable,
                module_params,
                module_state,
                module_rng,
            ) = module_data
            data, modelstate_flat = carry
            output, new_module_state = module_callable(
                module_params,
                data,
                training,
                module_state,
                module_rng,
            )
            return output, modelstate_flat + [new_module_state]

        @jaxtyped(typechecker=beartype)
        def model_callable(
            params: ModelParams,
            data: Data,
            training: bool,
            state: ModelState,
            rng: Any,
        ) -> Tuple[Data, ModelState]:

            # params, state, and module_callables are PyTrees with the same
            # structure as self.modules

            # split rng for each module, put in a PyTree with same structure
            rngs = jax.random.split(rng, len(jax.tree.leaves(self.modules)))
            rngs = jax.tree.unflatten(modules_structure, rngs)

            # use jax.tree.reduce to sequentially apply each module
            # then reconstruct the new state PyTree
            # we apply reduce over the zipped module_callables, params,
            # state, and rngs PyTrees

            module_data = jax.tree.map(
                lambda mc, mp, ms, mr: (mc, mp, ms, mr),
                module_callables,
                params,
                state,
                rngs,
            )

            output, new_state_flat = jax.tree.reduce(
                lambda ds, md: sequential(ds, md, training),
                module_data,
                initializer=(data, []),
                is_leaf=lambda x: isinstance(x, tuple) and len(x) == 4,
            )

            # reconstruct new state PyTree
            new_state = jax.tree.unflatten(modules_structure, new_state_flat)

            return output, new_state

        return model_callable

    # methods to modify the module list
    def __getitem__(
        self,
        key: jax.tree_util.KeyPath | str | int | slice | None,
    ) -> BaseModule | PyTree[BaseModule]:
        self.reset()
        if key is None:
            return self.modules

        elif isinstance(key, tuple):
            # KeyPath is just Tuple[Any, ...]
            curr = self.modules
            for k in key:
                curr = curr[k]
            return curr

        elif isinstance(key, (str, int, slice)):
            if isinstance(key, str) and not isinstance(self.modules, Dict):
                raise KeyError(
                    f"Cannot access module '{key}' by name since "
                    "SequentialModel modules are not stored in a "
                    "dictionary."
                )
            elif isinstance(key, (int, slice)) and not isinstance(
                self.modules,
                (List, Tuple),
            ):
                raise KeyError(
                    f"Cannot access module '{key}' by index since "
                    "SequentialModel modules are not stored in a "
                    "list or tuple."
                )
            return self.modules[key]

        else:
            raise TypeError(
                f"Invalid key type {type(key)} for SequentialModel "
                "module access."
            )

    def __setitem__(
        self,
        key: jax.tree_util.KeyPath | str | int | slice | None,
        module: BaseModule | List[BaseModule] | Tuple[BaseModule, ...],
    ) -> None:
        self.reset()
        if key is None:
            if isinstance(module, BaseModule):
                self.modules = [module]
            else:
                self.modules = module

        elif isinstance(key, tuple):
            # KeyPath is just Tuple[Any, ...]
            curr = self.modules
            for k in key[:~0]:
                curr = curr[k]
            curr[key[~0]] = module

        elif isinstance(key, str):
            if not isinstance(self.modules, Dict):
                raise KeyError(
                    f"Cannot set module '{key}' by name since "
                    "SequentialModel modules are not stored in a "
                    "dictionary."
                )
            self.modules[key] = module
        elif isinstance(key, (int, slice)):
            if not isinstance(self.modules, (List, Tuple)):
                raise KeyError(
                    f"Cannot set module '{key}' by index since "
                    "SequentialModel modules are not stored in a "
                    "list or tuple."
                )
            self.modules[key] = module
        else:
            raise TypeError(
                f"Invalid key type {type(key)} for SequentialModel "
                "module access."
            )

    def insert_module(
        self,
        index: int,
        module: BaseModule,
        key: jax.tree_util.KeyPath | str | int | None = None,
    ) -> None:
        r"""
        Insert a module at a specific index in the model.

        Parameters
        ----------
            index
                Index to insert the module at.
            module
                Module to insert.
            key
                Key to name the module if modules are stored in a dictionary.
                If None and modules are stored in a dictionary, a UUID will be
                generated and used as the key. Default is None. Ignored if
                modules are stored in a list, tuple, or other structure.
        """
        self.reset()
        if isinstance(self.modules, Dict):
            if key is None:
                key = str(uuid.uuid4().hex)
            # create new dict with new module at index
            items = list(self.modules.items())
            items.insert(index, (key, module))
            self.modules = Dict(items)
        elif isinstance(self.modules, List):
            self.modules.insert(index, module)
        elif isinstance(self.modules, Tuple):
            self.modules = (
                self.modules[:index] + (module,) + self.modules[index:]
            )
        else:
            raise TypeError(
                "Cannot insert module to SequentialModel since "
                "modules are not stored in a list, tuple, or "
                "dictionary."
            )

    def append_module(
        self,
        module: BaseModule,
        key: jax.tree_util.KeyPath | str | int | None = None,
    ) -> None:
        r"""
        Append a module to the end of the model.

        Parameters
        ----------
            module
                Module to append.
            key
                Key to name the module if modules are stored in a dictionary.
                If None and modules are stored in a dictionary, a UUID will be
                generated and used as the key. Default is None. Ignored if
                modules are stored in a list, tuple, or other structure.
        """
        self.insert_module(
            index=len(jax.tree.leaves(self.modules)),
            module=module,
            key=key,
        )

    def prepend_module(
        self,
        module: BaseModule,
        key: jax.tree_util.KeyPath | str | int | None = None,
    ) -> None:
        r"""
        Prepend a module to the beginning of the model.

        Parameters
        ----------
            module
                Module to prepend.
            key
                Key to name the module if modules are stored in a dictionary.
                If None and modules are stored in a dictionary, a UUID will be
                generated and used as the key. Default is None. Ignored if
                modules are stored in a list, tuple, or other structure.
        """
        self.insert_module(
            index=0,
            module=module,
            key=key,
        )

    def pop_module_by_index(
        self,
        index: int,
    ) -> BaseModule:
        r"""
        Remove and return a module at a specific index in the model.
        Parameters
        ----------
            index
                Index of the module to remove.
        Returns
            The removed module.
        """
        self.reset()
        if isinstance(self.modules, Dict):
            # create new dict without the module at index
            items = list(self.modules.items())
            key, module = items.pop(index)
            self.modules = Dict(items)
            return module
        elif isinstance(self.modules, List):
            return self.modules.pop(index)
        elif isinstance(self.modules, Tuple):
            module = self.modules[index]
            self.modules = self.modules[:index] + self.modules[index + 1 :]
            return module
        else:
            raise TypeError(
                "Cannot pop module from SequentialModel since "
                "modules are not stored in a list, tuple, or "
                "dictionary."
            )

    def pop_module_by_key(
        self,
        key: jax.tree_util.KeyPath | str | int,
    ) -> BaseModule:
        r"""
        Remove and return a module by key or index in the model.
        Parameters
        ----------
            key
                Key or index of the module to remove.
        Returns
            The removed module.
        """
        self.reset()
        if isinstance(key, str):
            if not isinstance(self.modules, Dict):
                raise KeyError(
                    f"Cannot pop module '{key}' by name since "
                    "SequentialModel modules are not stored in a "
                    "dictionary."
                )
            return self.modules.pop(key)
        elif isinstance(key, int):
            if not isinstance(self.modules, (List, Tuple)):
                raise KeyError(
                    f"Cannot pop module '{key}' by index since "
                    "SequentialModel modules are not stored in a "
                    "list or tuple."
                )
            return self.pop_module_by_index(key)
        else:
            raise TypeError(
                f"Invalid key type {type(key)} for SequentialModel "
                "module access."
            )

    def __add__(self, other: BaseModule) -> SequentialModel:
        r"""
        Overload the + operator to append a module or model to the current
        model.

        Parameters
        ----------
            other
                Module or model to append.

        Returns
        -------
            New SequentialModel with the other module or model appended.
        """
        new_model = SequentialModel(modules=self.modules)
        new_model.append_module(other)
        return new_model

    # aliases
    append = append_module
    prepend = prepend_module
    insert = insert_module
    add_module = append_module
    add = append_module
    pop = pop_module_by_key
