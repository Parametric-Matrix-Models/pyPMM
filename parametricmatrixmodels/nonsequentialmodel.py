from __future__ import annotations

import random

import jax
from beartype import beartype
from jaxtyping import jaxtyped

from .graph_util import (
    get_outer_connections_by_tree,
    place_connections_in_tree,
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
    make_mutable,
    setitem_by_strpath,
)
from .typing import (
    Any,
    Data,
    DataShape,
    Dict,
    HyperParams,
    List,
    ModuleCallable,
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

    __version__ = "0.0.0"

    def __init__(
        self,
        modules: ModelModules | BaseModule | None = None,
        connections: (
            Dict[str, str | List[str] | Tuple[str, ...]] | None
        ) = None,
        /,
        *,
        rng: Any | int | None = None,
        separator: str = ".",
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
        modules = make_mutable(modules)

        super().__init__(modules if modules is not None else {}, rng=rng)
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

    def __repr__(self) -> str:
        modules_repr = super().__repr__()

        # add the connections if present in the format
        # A -> B
        if self.connections is None or len(self.connections) == 0:
            connections_repr = "No connections defined."
            return f"{modules_repr}\nConnections:\n{connections_repr}"

        connections_lines = []
        for out_path, in_paths in self.connections.items():
            if isinstance(in_paths, (list, tuple)):
                for in_path in in_paths:
                    connections_lines.append(f"{out_path} -> {in_path}")
            else:
                connections_lines.append(f"{out_path} -> {in_paths}")
        connections_repr = "\n".join(connections_lines)
        return f"{modules_repr}\nConnections:\n{connections_repr}"

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

        # now we resolve the execution order
        topo_order, visited = resolve_connections(
            module_connections,
            start_key="input",
            end_key="output",
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
            if module_path == "input" or module_path == "output":
                continue
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
        /,
        *,
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
            assert jax.tree.structure(self.input_shape) == jax.tree.structure(
                input_shape
            ), (
                f"{self.name} is already compiled with input shape "
                f"{self.input_shape}, cannot recompile with different "
                f"input shape {input_shape}."
            )
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

        if verbose:
            print(f"{self.name} execution order:")
            for i, module_path in enumerate(self.execution_order):
                print(f"  {i}: {module_path}")

        # set the input and output shapes
        self.input_shape = input_shape

        input_progression, output_progression, self.output_shape = (
            self._get_shape_progression(input_shape)
        )

        if verbose:
            print(f"{self.name} input shape: {self.input_shape}")
            # print progression of shapes through the execution order
            print(f"{self.name} shape progression:")
            for module_path in self.execution_order:
                if module_path == "input" or module_path == "output":
                    continue
                module_input_shape = getitem_by_strpath(
                    input_progression,
                    module_path,
                    separator=self.separator,
                    allow_early_return=False,
                    return_remainder=False,
                    is_leaf=is_shape_leaf,
                )
                module_output_shape = getitem_by_strpath(
                    output_progression,
                    module_path,
                    separator=self.separator,
                    allow_early_return=False,
                    return_remainder=False,
                    is_leaf=is_shape_leaf,
                )
                print(
                    f"  {module_path}: {module_input_shape} -> "
                    f"{module_output_shape}"
                )
            print(f"{self.name} output shape: {self.output_shape}")

        # for all modules that appear in the execution order, compile them
        for module_path in self.execution_order:
            if module_path == "input" or module_path == "output":
                continue
            rng, module_rng = jax.random.split(rng)
            module = getitem_by_strpath(
                self.modules,
                module_path,
                separator=self.separator,
                allow_early_return=False,
                return_remainder=False,
            )

            module_input_shape = getitem_by_strpath(
                input_progression,
                module_path,
                separator=self.separator,
                allow_early_return=False,
                return_remainder=False,
                is_leaf=is_shape_leaf,
            )

            if verbose:
                print(
                    f"Compiling module '{module_path}' with input shape "
                    f"{module_input_shape}."
                )

            try:
                module.compile(
                    module_rng,
                    module_input_shape,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Error compiling module '{module_path}' ({module.name}) "
                    f"with input shape {module_input_shape}: {e}"
                ) from e

    def _get_module_input_dependencies(
        self,
    ) -> Tuple[List[PyTree[str]], PyTree[str]]:
        r"""
        Get the input dependencies for each module in the execution order.

        Returns
        -------
            module_input_dependencies
                List of PyTrees of str paths to the required inputs for each
                module in the execution order. The first entry corresponds to
                the "input" node and is None.
            output_input_dependencies
                PyTree of str paths to the required inputs for the "output"
                node.
        """

        execution_order = self.get_execution_order()

        # place the connections into the module tree
        placed_conn, out_placed_conn = place_connections_in_tree(
            self.connections,
            self.modules,
            separator=self.separator,
            in_key="input",
            out_key="output",
        )

        module_inputs = [None] + [
            getitem_by_strpath(
                placed_conn,
                mod_path,
                separator=self.separator,
                allow_early_return=False,
                return_remainder=False,
            )
            for mod_path in execution_order
            if mod_path != "input" and mod_path != "output"
        ]

        return module_inputs, out_placed_conn

    def _get_shape_progression(
        self,
        input_shape: DataShape,
        /,
    ) -> Tuple[PyTree[DataShape | None], PyTree[DataShape | None], DataShape]:
        r"""
        Get the progression of output shapes through the model given an input
        shape. The first entry is the model input shape, and the last entry is
        the model output shape.

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
            input_shapes
                PyTree of input shapes at each module in the execution order,
                with the same structure as the modules in the execution order.
            output_shapes
                PyTree of output shapes at each module in the execution order,
                with the same structure as the modules in the execution order.
            output_shape
                Shape of the output after passing through the model.

        """

        # get the execution order
        execution_order = self.get_execution_order()

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
            output_shapes,
            "input",
            input_shape,  # input node returns the model input shape
            separator=self.separator,
            is_leaf=is_shape_leaf,
        )

        module_input_deps, out_input_deps = (
            self._get_module_input_dependencies()
        )

        # now we build the input and output shapes in execution order
        for mod_path, req_in_paths in zip(execution_order, module_input_deps):
            if mod_path == "input" or mod_path == "output":
                continue
            # mod_path is the path to the current module in self.modules
            # req_in_paths is either a str or a list of str paths to the
            # required inputs for the current module
            # since we are in execution order, all required inputs will be
            # available in output_shapes
            in_shapes = jax.tree.map(
                lambda p: getitem_by_strpath(
                    output_shapes,
                    p,
                    separator=self.separator,
                    allow_early_return=False,
                    return_remainder=False,
                    is_leaf=is_shape_leaf,
                ),
                req_in_paths,
            )

            setitem_by_strpath(
                input_shapes,
                mod_path,
                in_shapes,
                separator=self.separator,
                is_leaf=is_shape_leaf,
            )

            module = getitem_by_strpath(
                self.modules,
                mod_path,
                separator=self.separator,
                allow_early_return=False,
                return_remainder=False,
            )
            try:
                out_shape = module.get_output_shape(in_shapes)
            except Exception as e:
                raise RuntimeError(
                    f"Error getting output shape for module '{mod_path}' "
                    f"({module.name}) with input shape {in_shapes}: {e}"
                    "\nCurrent input shapes progression:"
                    f"\n{input_shapes}"
                    "\nCurrent output shapes progression:"
                    f"\n{output_shapes}"
                ) from e
            setitem_by_strpath(
                output_shapes,
                mod_path,
                out_shape,
                separator=self.separator,
                is_leaf=is_shape_leaf,
            )

        # finally, get the output shape from the "output" node
        out_shapes = jax.tree.map(
            lambda p: getitem_by_strpath(
                output_shapes,
                p,
                separator=self.separator,
                allow_early_return=False,
                return_remainder=False,
                is_leaf=is_shape_leaf,
            ),
            out_input_deps,
        )

        return input_shapes, output_shapes, out_shapes

    def get_output_shape(self, input_shape: DataShape, /) -> DataShape:
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

        _, _, output_shape = self._get_shape_progression(input_shape)
        return output_shape

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

        self.modules = make_mutable(self.modules)

        # get the callables for each module in the execution order and put them
        # in a PyTree with the same structure
        module_callables: List[ModuleCallable | None] = [None] + [
            getitem_by_strpath(
                self.modules,
                mod_path,
                separator=self.separator,
                allow_early_return=False,
                return_remainder=False,
            )._get_callable()
            for mod_path in self.execution_order
            if mod_path != "input" and mod_path != "output"
        ]

        # get the input dependencies for each module
        module_input_deps, out_input_deps = (
            self._get_module_input_dependencies()
        )

        @jaxtyped(typechecker=beartype)
        def nonseq_callable(
            params: ModelParams,
            data: Data,
            training: bool,
            state: ModelState,
            rng: Any,
        ) -> Tuple[Data, ModelState]:

            # initialize the intermediate outputs PyTree
            intermediate_outputs: PyTree[Data] = (
                extend_structure_from_strpaths(
                    None,
                    self.execution_order,
                    separator=self.separator,
                )
            )
            # set the model input
            setitem_by_strpath(
                intermediate_outputs,
                "input",
                data,
                separator=self.separator,
            )
            new_state = state
            # now execute each module in the execution order, assembling the
            # inputs from the intermediate outputs
            for mod_path, module_callable, req_in_paths in zip(
                self.execution_order,
                module_callables,
                module_input_deps,
            ):
                if mod_path == "input" or mod_path == "output":
                    continue
                # assemble the inputs for the module

                in_data = jax.tree.map(
                    lambda p: getitem_by_strpath(
                        intermediate_outputs,
                        p,
                        separator=self.separator,
                        allow_early_return=False,
                        return_remainder=False,
                        is_leaf=is_shape_leaf,
                    ),
                    req_in_paths,
                )

                module_params = getitem_by_strpath(
                    params,
                    mod_path,
                    separator=self.separator,
                    allow_early_return=False,
                    return_remainder=False,
                )
                module_state = getitem_by_strpath(
                    new_state,
                    mod_path,
                    separator=self.separator,
                    allow_early_return=False,
                    return_remainder=False,
                )

                # call the module
                out_data, new_module_state = module_callable(
                    module_params,
                    in_data,
                    training,
                    module_state,
                    rng,
                )

                # store the output
                setitem_by_strpath(
                    intermediate_outputs,
                    mod_path,
                    out_data,
                    separator=self.separator,
                )
                # update the state
                setitem_by_strpath(
                    new_state,
                    mod_path,
                    new_module_state,
                    separator=self.separator,
                )
            # assemble the model output from the intermediate outputs
            out_data = jax.tree.map(
                lambda p: getitem_by_strpath(
                    intermediate_outputs,
                    p,
                    separator=self.separator,
                    allow_early_return=False,
                    return_remainder=False,
                    is_leaf=is_shape_leaf,
                ),
                out_input_deps,
            )
            return out_data, new_state

        return nonseq_callable

    def get_state(self) -> ModelState:
        r"""
        Get the state of all modules in the model as a PyTree.

        Override of the base method in order to ignore modules that are not in
        the execution order.

        Returns
        -------
            A PyTree of the states of all modules in the model with the same
            structure as the modules PyTree. Modules that are not in the
            execution order will have state None.
        """
        if not self.is_ready():
            raise RuntimeError(
                f"{self.name} is not ready. Call compile() first."
            )

        # if the model is ready, then all the modules in the execution order
        # are ready, and otherwise we can ignore them

        def get_state_or_none(module: BaseModule) -> State:
            if module.is_ready():
                return module.get_state()
            else:
                return None

        return jax.tree.map(
            get_state_or_none,
            self.modules,
        )

    def get_hyperparameters(self) -> HyperParams:
        return {
            "execution_order": self.execution_order,
            "connections": self.connections,
            "separator": self.separator,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            **super().get_hyperparameters(),
        }

    def set_hyperparameters(self, hyperparams: HyperParams, /) -> None:
        self.execution_order = hyperparams.get(
            "execution_order", self.execution_order
        )
        self.connections = hyperparams.get("connections", self.connections)
        self.separator = hyperparams.get("separator", self.separator)
        self.input_shape = hyperparams.get("input_shape", self.input_shape)
        self.output_shape = hyperparams.get("output_shape", self.output_shape)

        super().set_hyperparameters(hyperparams)

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
        # TODO: handle connections: anything that goes to "output" in self
        # should now go to the new module/model and anything that comes from
        # that module/model should now go to output
        raise NotImplementedError(
            "'+' not yet implemented for NonSequentialModel."
        )
        new_model = NonSequentialModel(
            self.modules,
            self.connections,
            rng=self.get_rng(),
            separator=self.separator,
        )
        new_model.append_module(other)
        return new_model
