from __future__ import annotations

import random
import uuid

import jax
from beartype import beartype
from jaxtyping import jaxtyped

from .graph_util import (
    get_outer_connections_by_tree,
    partition_connections_by_tree,
    resolve_connections,
)
from .model import Model
from .model_util import (
    ModelCallable,
    ModelModules,
    ModelParams,
    ModelState,
)
from .modules import BaseModule
from .tree_util import (
    extend_structure_from_strpaths,
    getitem_by_strpath,
    is_shape_leaf,
    setitem_by_strpath,
)
from .typing import (
    Any,
    Data,
    DataShape,
    Dict,
    List,
    ModuleCallable,
    OrderedSet,
    Params,
    PyTree,
    State,
    Tuple,
)


class NonSequentialModel(Model):
    r"""
    A nonsequential model that chains modules (or other models) together with
    directed connections.

    For confidence intervals or uncertainty quantification, wrap a trained
    model with ``ConformalModel``.

    See Also
    --------
    jax.tree
        PyTree utilities and concepts in JAX.
    Model
        Abstract base class for all models.
    SequentialModel
        A model that applies modules in sequence.
    ConformalModel
        Wrap a trained model to produce confidence intervals.
    """

    def __init__(
        self,
        modules: ModelModules | BaseModule | None = None,
        connections: (
            Dict[str, str | List[str] | Tuple[str, ...]] | None
        ) = None,
        rng: Any | int | None = None,
        separator: str = ".",
        max_recursion_depth: int = 100,
    ) -> None:
        r"""
        Initialize a nonsequential model with a PyTree of modules and a
        random key.

        Parameters
        ----------
            modules
                module(s) to initialize the model with. Default is None, which
                will become an empty dictionary. Can be a single module, which
                will be wrapped in a list, or a PyTree of modules (e.g., nested
                lists, tuples, or dictionaries).
            connections
                Directed connections between module input and outputs in the
                model. Keys are period-separated paths of module outputs, and
                values are lists or tuples of period-separated paths of module
                inputs that receive the output. The reserved keys "input" and
                "output" refer to the model input and output, respectively. The
                separator can be changed from the default period using the
                ``separator`` argument. Default is None, which will become an
                empty dictionary.
            rng
                Initial random key for the model. Default is None. If None, a
                new random key will be generated using JAX's ``random.key``. If
                an integer is provided, it will be used as the seed to create
                the key.
            separator
                Separator string to use for denoting paths in the connections
                dictionary. Default is ".".
            max_recursion_depth
                Maximum recursion depth for resolving the directed graph.
                Default is 100.

        Examples
        --------

        To denote a sequential model where all modules expect a bare array and
        produce a bare array:

            >>> modules = [Module1(), Module2(), Module3()]
            >>> connections = {
            ...     "input": "0",
            ...     "0": "1",
            ...     "1": "2",
            ...     "2": "output"
            ... }
            >>> model = NonSequentialModel(modules, connections)

        or equivalently to name the modules:

            >>> modules = {"M0": Module1(), "M1": Module2(), "M2": Module3()}
            >>> connections = {
            ...     "input": "M0",
            ...     "M0": "M1",
            ...     "M1": "M2",
            ...     "M2": "output"
            ... }
            >>> model = NonSequentialModel(modules, connections)

        or equivalently to use nested structures:
            >>> modules = {
            ...     "block1": [Module1(), Module2()],
            ...     "block2": {"M3": Module3()}
            ... }
            >>> connections = {
            ...     "input": "block1.0",
            ...     "block1.0": "block1.1",
            ...     "block1.1": "block2.M3",
            ...     "block2.M3": "output"
            ... }
            >>> model = NonSequentialModel(modules, connections)

        All three of the above will produce a model that applies the same
        three modules sequentially.

        If a module outputs a PyTree of arrays, or expects a PyTree of arrays
        as input, the connections can specify the leaf nodes using the same
        period-separated path syntax. For example, if Module1 outputs a dict
        with keys "a" and "b", and Module2 expects a tuple of two arrays as
        input, the connections can be specified as:

            >>> modules = {"M1": Module1(), "M2": Module2()}
            >>> connections = {
            ...     "input": "M1",
            ...     "M1.a": "M2.1",
            ...     "M1.b": "M2.0",
            ...     "M2": "output"
            ... }
            >>> model = NonSequentialModel(modules, connections)

        This will send the "a" output of Module1 to the second input of
        Module2, and the "b" output of Module1 to the first input of Module2.

        .. note::

            If a module expects a Tuple or List as input, it is best to write
            the module to accept both Tuple and List types, since the specific
            input type between List and Tuple cannot be inferred at compile
            time.

        If the entire model input or output is a PyTree of arrays, the
        connections use the same period-separated path syntax with the reserved
        keys "input" and "output". For example, if the model input is a dict
        with keys "x1" and "x2", and the model output is a Tuple of two arrays,
        the connections can be specified as:

            >>> modules = {"M1": Module1(), "M2": Module2()}
            >>> connections = {
            ...     "input.x1": "M1",
            ...     "M1": "M2",
            ...     "M2": "output.1",
            ...     "input.x2": "output.0"
            ... }
            >>> model = NonSequentialModel(modules, connections)

        This will perform a sequential model on the "x1" input through Module1
        and Module2, sending the output to the second output of the model, and
        will send the "x2" input directly to the first output of the model
        unchanged.

        Modules that output PyTrees need not be fully traversed if entire
        subtrees are to be passed between modules. For example, if Module1
        outputs a dict with keys "a" and "b", and Module2 expects a dict
        with keys "a" and "b" as input, the connections can be specified as:

            >>> modules = {"M1": Module1(), "M2": Module2()}
            >>> connections = {
            ...     "input": "M1",
            ...     "M1": "M2",
            ...     "M2": "output"
            ... }
            >>> model = NonSequentialModel(modules, connections)

        or equivalently:

            >>> modules = {"M1": Module1(), "M2": Module2()}
            >>> connections = {
            ...     "input": "M1",
            ...     "M1.a": "M2.a",
            ...     "M1.b": "M2.b",
            ...     "M2": "output"
            ... }
            >>> model = NonSequentialModel(modules, connections)

        Both ways will pass the entire output dict of Module1 to Module2.

        Module ouputs can be sent to multiple module inputs by specifying a
        list or tuple of input paths in the connections dictionary. For
        example, to send the output of Module1 to both Module2 and Module3:

            >>> modules = {"M1": Module1(), "M2": Module2(), "M3": Module3()}
            >>> connections = {
            ...     "input": "M1",
            ...     "M1": ["M2", "M3"],
            ...     "M2": "output.0",
            ...     "M3": "output.1"
            ... }
            >>> model = NonSequentialModel(modules, connections)

        This will create a model that sends the output of Module1 to both
        Module2 and Module3 in parallel, and collects their outputs as a Tuple
        as the model output.

        The order of the connections in the dictionary does not matter, as long
        as the connections form a valid directed acyclic graph from the
        model input to the model output. It is not necessary to use all parts
        of the model input, or all modules. However, this will raise a warning.
        It is not necessary and will not raise a warning if some parts of the
        outputs of some modules are not used, but all inputs of all present
        modules must be connected.

        See Also
        --------
        ModelModules
            Type alias for a PyTree of modules in a model.
        jax.random.key
            JAX function to create a random key.
        jax.tree_util.keystr
            JAX function to create string paths for PyTree KeyPaths in the
            format expected by the connections dictionary.
        """
        super().__init__(
            modules=modules if modules is not None else {}, rng=rng
        )
        self.connections = connections if connections is not None else {}

        # if "input" or "output" is in the module keys (if modules is a dict),
        # then raise an error
        if isinstance(self.modules, dict):
            if "input" in self.modules:
                raise ValueError(
                    "Module key 'input' is reserved for model input."
                )
            if "output" in self.modules:
                raise ValueError(
                    "Module key 'output' is reserved for model output."
                )

        self.execution_order: List[str] = None
        self.separator = separator
        self.max_recursion_depth = max_recursion_depth

    def get_execution_order(self) -> List[str]:
        r"""
        Resolve the connections dictionary to find the execution order of
        module execution.

        Raises
        ------
            ValueError
                If the connections do not form a valid directed acyclic graph
                from the model input to the model output.
        """

        module_connections = get_outer_connections_by_tree(
            self.connections,
            self.modules,
            separator=self.separator,
            in_key="input",
            out_key="output",
        )

        # now we have connections in the form
        # { 'input': {'<mod_path>', ...}, ... }

        # now is a good time to verify that 'input' is present as a key
        if "input" not in module_connections:
            raise ValueError(
                "Connections must include 'input' as a key "
                "denoting the model input."
            )

        # we want to reverse the connections to get the incoming edges for
        # each node, so we can traverse the graph from output to input
        incoming_edges: Dict[str, OrderedSet[str]] = {}
        for key, value in module_connections.items():
            for v in value:
                if v not in incoming_edges:
                    incoming_edges[v] = OrderedSet()
                incoming_edges[v].add(key)

        # now is a good time to verify that 'output' is present as a key in the
        # reversed connections
        if "output" not in incoming_edges:
            raise ValueError(
                "Connections must include 'output' as a value "
                "denoting the model output."
            )

        # convert OrderedSets back to lists
        incoming_edges = {
            key: list(value) for key, value in incoming_edges.items()
        }

        # now we resolve the execution order
        topo_order, visited = resolve_connections(
            incoming_edges,
            start_key="input",
            end_key="output",
            max_recursion_depth=self.max_recursion_depth,
        )

        return topo_order

    def reset(self) -> None:
        r"""
        Reset the compiled state of the model. This will require recompilation
        before the model can be used again.
        """
        self.execution_order = None
        super().reset()

    def is_ready(self) -> bool:
        r"""
        Check if the model is compiled and ready for use. Overrides the base
        implementation since not all modules need to be ready, as some may
        not appear in the execution order.

        Returns
        -------
            True if the model is compiled and ready, False otherwise.
        """

        # if the execution order is not set, the model is not ready
        if self.execution_order is None:
            return False

        # if any module in the execution order is not ready, the model is not
        # ready
        for module_path in self.execution_order:
            module = getitem_by_strpath(
                self.modules,
                module_path,
                separator=self.separator,
                allow_early_return=False,
                return_remainder=False,
            )
            if not module.is_ready():
                return False

        # if the input or output shapes are not set, the model is not ready
        return self.input_shape is not None and self.output_shape is not None

    def compile(
        self,
        rng: Any | int | None,
        input_shape: DataShape,
        verbose: bool = False,
    ) -> None:
        r"""
        Compile the model for training by finding the execution order of the
        directed graph defined by the connections, and compiling each module
        in that order.

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

        Raises
        ------
            ValueError
                If the connections do not form a valid directed acyclic graph
                from the model input to the model output.
        """

        if rng is None:
            rng = jax.random.key(random.randint(0, 2**32 - 1))
        elif isinstance(rng, int):
            rng = jax.random.key(rng)

        if self.is_ready():
            # just validate that the input shape matches the compiled one
            assert jax.tree.all(
                jax.tree.map(
                    lambda a, b: a == b,
                    self.input_shape,
                    input_shape,
                    is_leaf=is_shape_leaf,
                )
            ), (
                f"{self.name} is already compiled with input shape "
                f"{self.input_shape}, cannot recompile with different "
                f"input shape {input_shape}."
            )
            return

        # resolve the execution order
        self.execution_order = self.get_execution_order()

        # for all modules that appear in the execution order, compile them

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

        # get the execution order
        execution_order = self.get_execution_order()
        split_connections = partition_connections_by_tree(
            self.connections,
            self.modules,
            separator=self.separator,
            in_key="input",
            out_key="output",
        )
        module_connections = get_outer_connections_by_tree(
            self.connections,
            self.modules,
            separator=self.separator,
            in_key="input",
            out_key="output",
        )

        double_sep = f"{self.separator}{self.separator}"

        # turn split connections into a mapping from modules to their input and
        # output paths
        module_input_paths: Dict[str, List[str]] = {}
        module_output_paths: Dict[str, List[str]] = {}
        for key, values in split_connections.items():
            # the keys in split_connections are show the output paths
            # the values show the input paths
            # it is possible for the io path to be empty, in which case the
            # entire module output or input is used
            mod_out_path, out_io_path = key.split(double_sep)
            module_output_paths.setdefault(mod_out_path, []).append(
                out_io_path
            )

            for v in values:
                mod_in_path, in_io_path = v.split(double_sep)
                module_input_paths.setdefault(mod_in_path, []).append(
                    in_io_path
                )

        r"""
            Example
            -------

            Given modules like

                >>> modules = {
                ...     "M1": Module1(),
                ...     "block": {
                ...         "M2": Module2(),
                ...         "M3": Module3()
                ...     }
                ... }

            And connections like

                >>> connections = {
                ...     "input": ["M1", "block.M2.0", "block.M3.y"],
                ...     "M1.a": ["block.M2.1", "block.M3.x"],
                ...     "M1.b": "output",
                ...     "block.M2.0": "output",
                ...     "block.M3": "output"
                ... }

            The split connections will be

                >>> split_connections = {
                ...     "input..": ["M1..", "block.M2..0", "block.M3..y"],
                ...     "M1..a": ["block.M2..1", "block.M3..x"],
                ...     "M1..b": ["output.."],
                ...     "block.M2..0": ["output.."],
                ...     "block.M3..": ["output.."]
                ... }

            The module connections will be

                >>> module_connections = {
                ...     "input": ["M1", "block.M2", "block.M3"],
                ...     "M1": ["block.M2", "block.M3", "output"],
                ...     "block.M2": ["output"],
                ...     "block.M3": ["output"]
                ... }

            The module input paths will be

                >>> module_input_paths = {
                ...     "M1": [""],
                ...     "block.M2": ["0", "1"],
                ...     "block.M3": ["y", "x"]
                ...     "output": []
                ... }







        """

        # the modules execute sequentially in the execution order and each
        # return an arbitrary PyTree of arrays.
        # since the inputs to each module can be arbitrary compositions of
        # previous module outputs, we need to track the shapes of all
        # intermediate outputs and assemble the inputs to each module

        # first we make two empty PyTrees in the same shape as the modules, but
        # only the ones in the execution order
        input_shapes: PyTree[DataShape | None] = (
            extend_structure_from_strpaths(
                None,
                execution_order,
                separator=self.separator,
            )
        )
        output_shapes: PyTree[DataShape | None] = (
            extend_structure_from_strpaths(
                None,
                execution_order,
                separator=self.separator,
            )
        )

        # fill in the return shape from the "input" node
        setitem_by_strpath(
            input_shapes,
            "input",
            None,  # input node doesn't have any input
            separator=self.separator,
        )
        setitem_by_strpath(
            output_shapes,
            "input",
            input_shape,  # input node returns the model input shape
            separator=self.separator,
        )
        # now we iterate through the execution order and build the shapes
        for module_path in execution_order:
            if module_path == "input" or module_path == "output":
                continue

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
                f"{self.name} is not ready. Call compile() first."
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
                    "NonSequentialModel modules are not stored in a "
                    "dictionary."
                )
            elif isinstance(key, (int, slice)) and not isinstance(
                self.modules,
                (List, Tuple),
            ):
                raise KeyError(
                    f"Cannot access module '{key}' by index since "
                    "NonSequentialModel modules are not stored in a "
                    "list or tuple."
                )
            return self.modules[key]

        else:
            raise TypeError(
                f"Invalid key type {type(key)} for NonSequentialModel "
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
                    "NonSequentialModel modules are not stored in a "
                    "dictionary."
                )
            self.modules[key] = module
        elif isinstance(key, (int, slice)):
            if not isinstance(self.modules, (List, Tuple)):
                raise KeyError(
                    f"Cannot set module '{key}' by index since "
                    "NonSequentialModel modules are not stored in a "
                    "list or tuple."
                )
            self.modules[key] = module
        else:
            raise TypeError(
                f"Invalid key type {type(key)} for NonSequentialModel "
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
                "Cannot insert module to NonSequentialModel since "
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
                "Cannot pop module from NonSequentialModel since "
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
                    "NonSequentialModel modules are not stored in a "
                    "dictionary."
                )
            return self.modules.pop(key)
        elif isinstance(key, int):
            if not isinstance(self.modules, (List, Tuple)):
                raise KeyError(
                    f"Cannot pop module '{key}' by index since "
                    "NonSequentialModel modules are not stored in a "
                    "list or tuple."
                )
            return self.pop_module_by_index(key)
        else:
            raise TypeError(
                f"Invalid key type {type(key)} for NonSequentialModel "
                "module access."
            )

    def __add__(self, other: BaseModule) -> NonSequentialModel:
        r"""
        Overload the + operator to append a module or model to the current
        model.

        Parameters
        ----------
            other
                Module or model to append.

        Returns
        -------
            New NonSequentialModel with the other module or model appended.
        """
        new_model = NonSequentialModel(modules=self.modules)
        new_model.append_module(other)
        return new_model

    # aliases
    append = append_module
    prepend = prepend_module
    insert = insert_module
    add_module = append_module
    add = append_module
    pop = pop_module_by_key
