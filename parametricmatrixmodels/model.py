from __future__ import annotations

import random
import sys
import uuid
import warnings
from abc import abstractmethod
from pathlib import Path
from typing import IO

import jax
import jax.numpy as np
import numpy as onp
from beartype import beartype
from jaxtyping import jaxtyped
from packaging.version import parse

import parametricmatrixmodels as pmm

from . import tree_util
from .model_util import (
    ModelCallable,
    ModelModules,
    ModelParams,
    ModelState,
    ModelStruct,
    autobatch,
)
from .modules import BaseModule
from .training import OptimizerState, make_loss_fn, train
from .tree_util import is_shape_leaf, safecast, strfmt_pytree
from .typing import (
    Any,
    Array,
    Callable,
    Data,
    DataShape,
    Dict,
    HyperParams,
    Inexact,
    List,
    PyTree,
    Serializable,
    Tuple,
)


class Model(BaseModule):
    r"""

    Abstract base class for all models. Do not instantiate this class directly.

    A ``Model`` is a PyTree of modules that can be trained and
    evaluated. Inputs are passed through each module to produce
    outputs. ``Model`` s are also ``BaseModule`` s, so they can be
    nested inside other models.

    For confidence intervals or uncertainty quantification, wrap a trained
    model with ``ConformalModel``.

    See Also
    --------
    jax.tree
        PyTree utilities and concepts in JAX.
    SequentialModel
        A simple sequential model that chains modules together.
    NonsequentialModel
        A model that allows for non-sequential connections between modules.
    ConformalModel
        Wrap a trained model to produce confidence intervals.
    """

    def __repr__(self) -> str:
        return (
            f"{self.name}(\n"
            + strfmt_pytree(
                self.modules,
                indent=0,
                indentation=1,
                base_indent_str="  ",
            )
            + "\n)"
        )

    # override the trainable property
    @property
    def trainable(self) -> bool:
        return any(
            module.trainable for module in jax.tree.leaves(self.modules)
        )

    @trainable.setter
    def trainable(self, value: bool) -> None:
        for module in jax.tree.leaves(self.modules):
            module.trainable = value

    def __init__(
        self,
        modules: ModelModules | BaseModule | None = None,
        /,
        *,
        rng: Any | int | None = None,
    ) -> None:
        """
        Initialize the model with a PyTree of modules and a random key.

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
        ModelModules : Type alias for a PyTree of modules in a model.
        jax.random.key : JAX function to create a random key.
        """
        self.modules = modules if modules is not None else []
        if isinstance(modules, BaseModule):
            self.modules = [modules]
        if rng is None:
            self.rng = jax.random.key(random.randint(0, 2**32 - 1))
        elif isinstance(rng, int):
            self.rng = jax.random.key(rng)
        else:
            self.rng = rng
        self.reset()

    def get_num_trainable_floats(self) -> int | None:
        num_trainable_floats = [
            module.get_num_trainable_floats()
            for module in jax.tree.leaves(self.modules)
        ]
        if None in num_trainable_floats:
            return None
        else:
            return sum(num_trainable_floats)

    def is_ready(self) -> bool:
        return (
            len(jax.tree.leaves(self.modules)) > 0
            and all(
                module.is_ready() for module in jax.tree.leaves(self.modules)
            )
            and self.input_shape is not None
            and self.output_shape is not None
        )

    def _get_all_unready_modules(self) -> List[BaseModule]:
        return [
            module
            for module in jax.tree.leaves(self.modules)
            if not module.is_ready()
        ]

    def _get_unready_reason(self) -> str:
        if len(jax.tree.leaves(self.modules)) == 0:
            return "No modules in the model."
        unready_modules = self._get_all_unready_modules()
        if len(unready_modules) > 0:
            return f"Unready modules: {unready_modules}."
        if self.input_shape is None:
            return (
                "Model input shape is not set. Call compile() with an input"
                " shape to set it."
            )
        if self.output_shape is None:
            return (
                "Model output shape is not set. Call compile() with an input"
                " shape to set it."
            )
        return "Unknown reason."

    def reset(self) -> None:
        self.input_shape: DataShape | None = None
        self.output_shape: DataShape | None = None
        self.callable: ModelCallable | None = None
        self._optimizer_state = None
        self.grad_callable_params = None
        self.grad_callable_params_options = None
        self.grad_callable_inputs = None
        self.grad_callable_inputs_options = None

    @abstractmethod
    def compile(
        self,
        rng: Any | int | None,
        input_shape: DataShape,
        /,
        *,
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

        raise NotImplementedError(
            f"{self.name}.compile() not implemented. Must be implemented by "
            "subclasses."
        )

    @abstractmethod
    def get_output_shape(self, input_shape: DataShape, /) -> DataShape:
        """
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
        raise NotImplementedError(
            f"{self.name}.get_output_shape() not implemented. Must be"
            " implemented by subclasses."
        )

    def get_modules(self) -> ModelModules:
        """
        Get the modules of the model.

        Returns
        -------
            modules
                PyTree of modules in the model. The structure of the PyTree
                will match that of the modules in the model.

        See Also
        --------
        ModelModules : Type alias for a PyTree of modules in a model.
        """
        return self.modules

    def get_params(self) -> ModelParams:
        """
        Get the parameters of the model.

        Returns
        -------
            params
                PyTree of PyTrees of numpy arrays representing the parameters
                of each module in the model. The structure of the PyTree will
                be a composite structure where the upper level structure
                matches that of the modules in the model, and the lower level
                structure matches that of the parameters of each module.

        See Also
        --------
        ModelParams : Type alias for a PyTree of parameters in a model.
        get_modules : Get the modules of the model, in the same structure as
            the parameters returned by this method.
        set_params : Set the parameters of the model from a corresponding
            PyTree of PyTrees of numpy arrays.
        """
        return jax.tree.map(lambda m: m.get_params(), self.modules)

    def set_params(self, params: ModelParams, /) -> None:
        """
        Set the parameters of the model from a PyTree of PyTrees of numpy
        arrays.

        Parameters
        ----------
            params
                PyTree of PyTrees of numpy arrays representing the parameters
                of each module in the model. The structure of the PyTree must
                match that of the modules in the model, and the lower level
                structure must match that of the parameters of each module.

        See Also
        --------
        ModelParams : Type alias for a PyTree of parameters in a model.
        get_modules : Get the modules of the model, in the same structure as
            the parameters accepted by this method.
        get_params : Get the parameters of the model, in the same structure
            as the parameters accepted by this method.
        """

        # this will fail if the structure of self.modules isn't a prefix of the
        # structure of params
        try:
            jax.tree.map(lambda m, p: m.set_params(p), self.modules, params)
        except ValueError as e:
            raise ValueError(
                "Structure of params does not match structure of modules. "
                "The structure of the modules must be a prefix of the "
                "structure of the params."
            ) from e

    def get_state(self) -> ModelState:
        r"""
        Get the state of the model. The state is a PyTree of PyTrees of numpy
        arrays representing the state of each module in the model. The
        structure of the PyTree will be a composite structure where the upper
        level structure matches that of the modules in the model, and the lower
        level structure matches that of the state of each module.

        Returns
        -------
            state
                PyTree of PyTrees of numpy arrays representing the state of
                each module in the model. The structure of the PyTree will be
                a composite structure where the upper level structure matches
                that of the modules in the model, and the lower level structure
                matches that of the state of each module.

        See Also
        --------
        ModelState : Type alias for a PyTree of states in a model.
        get_modules : Get the modules of the model, in the same structure as
            the state returned by this method.
        set_state : Set the state of the model from a corresponding PyTree
            of PyTrees of numpy arrays.
        """

        return jax.tree.map(lambda m: m.get_state(), self.modules)

    def set_state(self, state: ModelState, /) -> None:
        r"""
        Set the state of the model from a PyTree of PyTrees of numpy arrays.

        Parameters
        ----------
            state
                PyTree of PyTrees of numpy arrays representing the state of
                each module in the model. The structure of the PyTree must
                match that of the modules in the model, and the lower level
                structure must match that of the state of each module.

        See Also
        --------
        ModelState : Type alias for a PyTree of states in a model.
        get_modules : Get the modules of the model, in the same structure as
            the state accepted by this method.
        get_state : Get the state of the model, in the same structure as
            the state accepted by this method.
        """

        if state is None:
            return

        # this will fail if the structure of self.modules isn't a prefix of the
        # structure of state
        try:
            jax.tree.map(lambda m, s: m.set_state(s), self.modules, state)
        except Exception as e:
            raise ValueError(
                "Unable to set model state, see inner exception for details."
            ) from e

    def get_rng(self) -> Any:
        return self.rng

    def set_rng(self, rng: Any, /) -> None:
        """
        Set the random key for the model.

        Parameters
        ----------
            rng : Any
                Random key to set for the model. JAX PRNGKey, integer seed, or
                `None`. If None, a new random key will be generated using JAX's
                ``random.key``. If an integer is provided, it will be used as
                the seed to create the key.
        """
        if isinstance(rng, int):
            self.rng = jax.random.key(rng)
        elif rng is None:
            self.rng = jax.random.key(random.randint(0, 2**32 - 1))
        else:
            self.rng = rng

    @abstractmethod
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

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the subclass.

        See Also
        --------
        __call__ : Calls the model with the current parameters and
            given input, state, and rng.
        ModelCallable : Typing for the callable returned by this method.
        Params : Typing for the model parameters.
        Data : Typing for the input and output data.
        State : Typing for the model state.
        """
        raise NotImplementedError(
            f"{self.name}._get_callable() not implemented. Must be"
            " implemented by subclasses."
        )

    def _verify_input(self, X: Data) -> None:
        if not self.is_ready():
            unready_reason = self._get_unready_reason()
            raise RuntimeError(
                f"{self.name} is not ready. Call compile() first."
                f"\nReason: {unready_reason}"
            )
        # assert that the model was compiled for this input shape
        if not tree_util.all_equal(
            self.input_shape,
            tree_util.get_shapes(X, axis=slice(1, None)),
        ):
            raise ValueError(
                f"Input shape {tree_util.get_shapes(X, axis=slice(1, None))} "
                f"does not match compiled input shape {self.input_shape}. "
                "Please re-compile the model with the correct input shape."
            )

    def __call__(
        self,
        X: Data,
        /,
        *,
        dtype: jax.typing.DTypeLike = np.float64,
        rng: Any | int | None = None,
        return_state: bool = False,
        update_state: bool = False,
        max_batch_size: int | None = None,
    ) -> Tuple[Data, ModelState] | Data:
        """
        Call the model with the input data.

        Parameters
        ----------
            X
                Input array of shape (batch_size, <input feature axes>).
                For example, (batch_size, input_features) for a 1D input or
                (batch_size, input_height, input_width, input_channels) for a
                3D input.
            dtype
                Data type of the output array. Default is jax.numpy.float64.
                It is strongly recommended to perform training in single
                precision (float32 and complex64) and inference with double
                precision inputs (float64, the default here) with single
                precision weights. Default is float64.
            rng
                JAX random key for stochastic modules. Default is None.
                If None, the saved rng key will be used if it exists, which
                would be the final rng key from the last training run. If an
                integer is provided, it will be used as the seed to create a
                new JAX random key. Default is the saved rng key if it exists,
                otherwise a new random key will be generated.
            return_state
                If True, the model will return the state of the model after
                evaluation. Default is ``False``.
            update_state
                If True, the model will update the state of the model after
                evaluation. Default is ``False``.
            max_batch_size
                If provided, the input will be split into batches of at most
                this size and processed sequentially to avoid OOM errors.
                Default is ``None``, which means the input will be processed in
                a single batch.

        Returns
        -------
            Data
                Output data as a PyTree of JAX arrays, the structure and shape
                of which is determined by the model's specific modules.
            ModelState
                If ``return_state`` is ``True``, the state of the model after
                evaluation as a PyTree of PyTrees of JAX arrays, the structure
                of which matches that of the model's modules.
        """
        if not self.is_ready():
            unready_reason = self._get_unready_reason()
            raise RuntimeError(
                f"{self.name} is not ready. Call compile() first."
                f"\nReason: {unready_reason}"
            )

        # assert that the model was compiled for this input shape
        self._verify_input(X)

        if self.callable is None:
            self.callable = jax.jit(self._get_callable(), static_argnums=(2,))

        # safecast input to requested dtype
        X_ = safecast(X, dtype)

        if rng is None:
            rng = self.get_rng()
        elif isinstance(rng, int):
            rng = jax.random.key(rng)

        autobatched_callable = autobatch(self.callable, max_batch_size)

        out, new_state = autobatched_callable(
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

    def grad_input(
        self,
        X: PyTree[Inexact[Array, "num_samples ..."], " DataStruct"],
        /,
        *,
        dtype: jax.type.DTypeLike = np.float64,
        rng: Any | int | None = None,
        return_state: bool = False,
        update_state: bool = False,
        fwd: bool | None = None,
        max_batch_size: int | None = None,
    ) -> (
        Tuple[
            PyTree[Inexact[Array, "num_samples ..."], "... DataStruct"],
            ModelState,
        ]
        | PyTree[Inexact[Array, "num_samples ..."], "... DataStruct"]
    ):
        r"""
        Doc TODO

        Parameters
        ----------
        fwd
            If True, use ``jax.jacfwd``, otherwise use ``jax.jacrev``. Default
            is ``None``, which decides based on the input and output shapes.
        max_batch_size
            If provided, the input will be split into batches of at most
            this size and processed sequentially to avoid OOM errors.
            Default is ``None``, which means the input will be processed in
            a single batch. If ``max_batch_size`` is set to ``1``, the gradient
            will be computed one sample at a time without batching. This case
            is particularly important for ``grad_input`` since the Jacobian
            contains gradients across different batch samples and thus scales
            with the square of the batch size.

        Returns
        -------
            PyTree
                Gradient of the model output with respect to the input data,
                as a PyTree of JAX arrays, the structure of which matches that
                of the output structure of the model composed above the input
                data structure. Each leaf array will have shape
                (num_samples, output_dim1, output_dim2, ..., input_dim1,
                input_dim2, ...), where the output dimensions correspond to
                the shape of the model output for that leaf, and the input
                dimensions correspond to the shape of the input data for that
                leaf.
            ModelState
                If ``return_state`` is ``True``, the state of the model after
                evaluation as a PyTree of PyTrees of JAX arrays, the structure
                of which matches that of the model's modules.
        """

        if not self.is_ready():
            unready_reason = self._get_unready_reason()
            raise RuntimeError(
                f"{self.name} is not ready. Call compile() first."
                f"\nReason: {unready_reason}"
            )
        self._verify_input(X)
        if self.callable is None:
            self.callable = jax.jit(self._get_callable(), static_argnums=(2,))

        def get_num_elems(count: int, shape: Tuple[int, ...]) -> int:
            return count + onp.prod(shape)

        if self.grad_callable_inputs is None and fwd is None:
            num_input_elems = jax.tree.reduce(
                get_num_elems, self.input_shape, initializer=0
            )
            num_output_elems = jax.tree.reduce(
                get_num_elems, self.output_shape, initializer=0
            )

            # if fwd is None, decide based on input and output sizes
            # fwd is more efficient when the number of input elements is less
            # than the number of output elements in general
            fwd = (
                fwd
                if fwd is not None
                else (num_input_elems < num_output_elems)
            )

        batched = (max_batch_size is None) or (max_batch_size > 1)

        # prepare the grad callable if not already done
        if (self.grad_callable_inputs is None) or (
            fwd is not None
            and (self.grad_callable_inputs_options != (batched, fwd))
        ):
            self.grad_callable_inputs_options = (batched, fwd)
            if not batched:
                # make non-batched version of the callable
                def remove_batch_dim(x: np.ndarray) -> np.ndarray:
                    return x[0, ...]

                @jaxtyped(typechecker=beartype)
                def callable_single(
                    params: ModelParams,
                    x: Data,
                    training: bool,
                    state: ModelState,
                    rng: Any,
                ) -> Data:
                    y, new_state = self.callable(
                        params, x[None, ...], training, state, rng
                    )
                    return jax.tree.map(remove_batch_dim, y), new_state

                if fwd:
                    grad_single = jax.jacfwd(
                        callable_single, argnums=1, has_aux=True
                    )
                else:
                    grad_single = jax.jacrev(
                        callable_single, argnums=1, has_aux=True
                    )
                self.grad_callable_inputs = jax.jit(
                    jax.vmap(grad_single, in_axes=(None, 0, None, None, None)),
                    static_argnums=(2,),
                )
            else:
                if fwd:
                    grad_callable_inputs_ = jax.jit(
                        jax.jacfwd(self.callable, argnums=1, has_aux=True),
                        static_argnums=(2,),
                    )
                else:
                    grad_callable_inputs_ = jax.jit(
                        jax.jacrev(self.callable, argnums=1, has_aux=True),
                        static_argnums=(2,),
                    )

                output_ndims = jax.tree.map(
                    lambda shape: len(shape),
                    self.output_shape,
                    is_leaf=is_shape_leaf,
                )
                input_ndims = jax.tree.map(
                    lambda shape: len(shape),
                    self.input_shape,
                    is_leaf=is_shape_leaf,
                )

                # take the diagonal (batch-wise jacobian)
                @jaxtyped(typechecker=beartype)
                def inner_map_fn(
                    input_ndim: int,
                    Y_out: Inexact[Array, "batch ..."],
                    output_ndim: int,
                ) -> Inexact[Array, "batch ..."]:

                    # Y_out is an array of shape (batch, <output dims>, batch,
                    # <input dims>)

                    batch_size = Y_out.shape[0]
                    diag_indices = (
                        (np.arange(batch_size),)
                        + (slice(None),) * output_ndim
                        + (np.arange(batch_size),)
                        + (slice(None),) * input_ndim
                    )
                    return Y_out[diag_indices]

                # map over output structure
                @jaxtyped(typechecker=beartype)
                def outer_map_fn(
                    output_ndim: int,
                    Y_out: PyTree[Inexact[Array, "batch ..."]],
                    in_ndims: PyTree[int],
                ) -> PyTree[Inexact[Array, "..."]]:
                    return jax.tree.map(
                        lambda in_ndim, y: inner_map_fn(
                            in_ndim, y, output_ndim
                        ),
                        in_ndims,
                        Y_out,
                    )

                @jaxtyped(typechecker=beartype)
                def grad_callable_inputs(
                    params: ModelParams,
                    X: Data,
                    training: bool,
                    state: ModelState,
                    rng: Any,
                ) -> Tuple[Data, ModelState]:
                    Y, new_states = grad_callable_inputs_(
                        params, X, training, state, rng
                    )
                    # the structure of Y is a composite PyTree where
                    # the outer structure matches that of the model output,
                    # and the inner structure matches that of the input X
                    # so we have to map over both structures to take the
                    # diagonal
                    # each leaf of Y is and array of shape
                    # (batch_size, output_dim1, output_dim2, ...,
                    # batch_size, input_dim1, input_dim2, ...)
                    # we want to take the diagonal along the two batch axes

                    Y_diag = jax.tree.map(
                        lambda out_ndim, y: outer_map_fn(
                            out_ndim, y, input_ndims
                        ),
                        output_ndims,
                        Y,
                    )
                    return Y_diag, new_states

                self.grad_callable_inputs = grad_callable_inputs

        # safecast input to requested dtype
        X_ = safecast(X, dtype)

        if rng is None:
            rng = self.get_rng()
        elif isinstance(rng, int):
            rng = jax.random.key(rng)

        autobatched_grad_callable = autobatch(
            self.grad_callable_inputs, max_batch_size
        )
        grad_input_result, new_state = autobatched_grad_callable(
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
            return grad_input_result, new_state
        else:
            return grad_input_result

    def grad_params(
        self,
        X: PyTree[Inexact[Array, "num_samples ..."], " DataStruct"],
        /,
        *,
        dtype: jax.typing.DTypeLike = np.float64,
        rng: Any | int | None = None,
        return_state: bool = False,
        update_state: bool = False,
        fwd: bool | None = None,
        max_batch_size: int | None = None,
    ) -> (
        Tuple[
            PyTree[Inexact[Array, "num_samples ..."] | Tuple[()] | None],
            ModelState,
        ]
        | PyTree[Inexact[Array, "num_samples ..."] | Tuple[()] | None]
    ):
        r"""
        Doc TODO

        Parameters
        ----------
        fwd
            If True, use ``jax.jacfwd``, otherwise use ``jax.jacrev``. Default
            is ``None``, which decides based on the input and output shapes.
        max_batch_size
            If provided, the input will be split into batches of at most
            this size and processed sequentially to avoid OOM errors.
            Default is ``None``, which means the input will be processed in
            a single batch. Only applies if ``batched=True``.
        Returns
        -------
            PyTree
                Gradient of the model output with respect to the model
                parameters, as a PyTree of PyTrees of JAX arrays, the upper
                level structure of which matches that of the model's modules,
                and the lower level structure of which matches that of the
                parameters of each module. Each leaf array will have shape
                (num_samples, output_dim1, output_dim2, ..., param_dim1,
                param_dim2, ...), where the output dimensions correspond to
                the shape of the model output for that leaf, and the param
                dimensions correspond to the shape of the parameter for that
                leaf.
            ModelState
                If ``return_state`` is ``True``, the state of the model after
                evaluation as a PyTree of PyTrees of JAX arrays, the structure
                of which matches that of the model's modules.
        """

        if not self.is_ready():
            unready_reason = self._get_unready_reason()
            raise RuntimeError(
                f"{self.name} is not ready. Call compile() first."
                f"\nReason: {unready_reason}"
            )
        self._verify_input(X)
        if self.callable is None:
            self.callable = jax.jit(self._get_callable(), static_argnums=(2,))

        def get_num_elems(count: int, shape: Tuple[int, ...]) -> int:
            return count + onp.prod(shape)

        if self.grad_callable_params is None and fwd is None:
            num_input_elems = jax.tree.reduce(
                get_num_elems, self.input_shape, initializer=0
            )
            num_output_elems = jax.tree.reduce(
                get_num_elems, self.output_shape, initializer=0
            )

            # if fwd is None, decide based on input and output sizes
            # fwd is more efficient when the number of input elements is less
            # than the number of output elements in general
            fwd = (
                fwd
                if fwd is not None
                else (num_input_elems < num_output_elems)
            )

        if self.grad_callable_params is None or (
            fwd is not None and (self.grad_callable_params_options != fwd)
        ):
            self.grad_callable_params_options = fwd
            if fwd:
                self.grad_callable_params = jax.jit(
                    jax.jacfwd(self.callable, argnums=0, has_aux=True),
                    static_argnums=(2,),
                )
            else:
                self.grad_callable_params = jax.jit(
                    jax.jacrev(self.callable, argnums=0, has_aux=True),
                    static_argnums=(2,),
                )

        # safecast input to requested dtype
        X_ = safecast(X, dtype)

        if rng is None:
            rng = self.get_rng()
        elif isinstance(rng, int):
            rng = jax.random.key(rng)

        autobatched_grad_callable = autobatch(
            self.grad_callable_params, max_batch_size
        )
        grad_params_result, new_state = autobatched_grad_callable(
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
            return grad_params_result, new_state
        else:
            return grad_params_result

    def set_precision(self, prec: jax.typing.DTypeLike | str | int, /) -> None:
        """
        Set the precision of the model parameters and states.

        Parameters
        ----------
            prec
                Precision to set for the model parameters and states.
                Valid options are:
                [for 32-bit precision (all options are equivalent)]
                - np.float32, np.complex64, "float32", "complex64"
                - "single", "f32", "c64", 32
                [for 64-bit precision (all options are equivalent)]
                - np.float64, np.complex128, "float64", "complex128"
                - "double", "f64", "c128", 64
        """
        if not self.is_ready():
            unready_reason = self._get_unready_reason()
            raise RuntimeError(
                f"{self.name} is not ready. Call compile() first."
                f"\nReason: {unready_reason}"
            )

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

        jax.tree.map(lambda m: m.set_precision(prec), self.modules)

    # alias for set_precision method that returns self
    def astype(self, dtype: jax.typing.DTypeLike | int, /) -> "Model":
        """
        Convenience wrapper to set_precision using the dtype argument, returns
        self.
        """
        self.set_precision(dtype)
        return self

    def train(
        self,
        X: PyTree[
            Inexact[Array, "num_samples feature0 ?*features"], " InStruct"
        ],  # in features
        /,
        Y: (
            PyTree[
                Inexact[Array, "num_samples target0 ?*targets"], " OutStruct"
            ]
            | None
        ) = None,  # targets
        *,
        Y_unc: (
            PyTree[
                Inexact[Array, "num_samples target0 ?*targets"], " OutStruct"
            ]
            | None
        ) = None,  # uncertainty in the targets, if applicable
        X_val: (
            PyTree[
                Inexact[Array, "num_val_samples feature0 ?*features"],
                " InStruct",
            ]
            | None
        ) = None,  # validation in features
        Y_val: (
            PyTree[
                Inexact[Array, "num_val_samples target0 ?*targets"],
                " OutStruct",
            ]
            | None
        ) = None,  # validation targets
        Y_val_unc: (
            PyTree[
                Inexact[Array, "num_val_samples target0 ?*targets"],
                " OutStruct",
            ]
            | None
        ) = None,  # uncertainty in the validation targets, if applicable
        loss_fn: str | Callable = "mse",
        lr: float | Callable[[int], float] = 1e-3,
        batch_size: int = 32,
        epochs: int = 100,
        target_loss: float = -np.inf,
        early_stopping_patience: int = 100,
        early_stopping_min_delta: float = -np.inf,
        # advanced options
        resume: bool = False,
        initialization_seed: Any | int | None = None,
        callback: Callable | None = None,
        unroll: int | None = None,
        verbose: bool = True,
        batch_rng: Any | int | None = None,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        clip: float = 1e3,
    ) -> None:

        # check if the model is ready
        if not self.is_ready():
            if initialization_seed is None:
                initialization_seed = jax.random.key(
                    random.randint(0, 2**32 - 1)
                )
            elif isinstance(initialization_seed, int):
                initialization_seed = jax.random.key(initialization_seed)

            self.compile(
                initialization_seed, jax.tree.map(lambda x: x.shape[1:], X)
            )

        self._verify_input(X)

        # input validation happens in the training.train function

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

        # check if any of the model parameters are complex
        params = self.get_params()
        any_complex = jax.tree_util.tree_reduce(
            lambda acc, x: acc or np.iscomplexobj(x), params, initializer=False
        )

        # get the trainable modules and map those bools to the param structure
        # i.e. if trainable modules is [False, True] and the first module has
        # parameters p1 and the second module has parameters [p2, p3], then
        # we want trainable_params to be [False, [True, True]] so that the
        # training function knows to update p2 and p3 but not p1
        trainable_modules = self.get_trainable_modules()

        trainable_flags = jax.tree.broadcast(trainable_modules, params)

        # train the model
        (
            final_params,
            final_model_states,
            final_model_rng,
            final_epoch,
            final_optimizer_states,
        ) = train(
            resume=resume,
            adam_state=self._optimizer_state,
            init_params=self.get_params(),
            trainable_flags=trainable_flags,
            init_state=self.get_state(),
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
            epochs=epochs,
            target_loss=target_loss,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            callback=callback,
            unroll=unroll,
            verbose=verbose,
            batch_rng=batch_rng,
            b1=b1,
            b2=b2,
            eps=eps,
            clip=clip,
            real=not any_complex,
        )

        # set the final parameters
        self.set_params(final_params)
        # set the final state
        self.set_state(final_model_states)
        # set the final rng
        self.set_rng(final_model_rng)
        # set the final optimizer state
        self._optimizer_state = final_optimizer_states

    def get_trainable_modules(self) -> PyTree[bool, ModelStruct]:
        """
        Return a PyTree with the same structure as the model's modules, where
        each leaf node is a boolean indicating whether the corresponding module
        is trainable (i.e. mod.trainable == True and mod.get_params() is not
        None/())
        """

        def is_trainable(mod: BaseModule) -> bool:
            p = mod.get_params()
            has_params = (p is not None) and len(jax.tree.leaves(p)) > 0
            return mod.trainable and has_params

        return jax.tree.map(is_trainable, self.modules)

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
                    "modules are not stored in a "
                    "dictionary."
                )
            elif isinstance(key, (int, slice)) and not isinstance(
                self.modules,
                (List, Tuple),
            ):
                raise KeyError(
                    f"Cannot access module '{key}' by index since "
                    "modules are not stored in a "
                    "list or tuple."
                )
            return self.modules[key]

        else:
            raise TypeError(f"Invalid key type {type(key)} for module access.")

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
                    "modules are not stored in a "
                    "dictionary."
                )
            self.modules[key] = module
        elif isinstance(key, (int, slice)):
            if not isinstance(self.modules, (List, Tuple)):
                raise KeyError(
                    f"Cannot set module '{key}' by index since "
                    "modules are not stored in a "
                    "list or tuple."
                )
            self.modules[key] = module
        else:
            raise TypeError(f"Invalid key type {type(key)} for module access.")

    def insert_module(
        self,
        index: int,
        module: BaseModule,
        /,
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
                "Cannot insert module to since "
                "modules are not stored in a list, tuple, or "
                "dictionary."
            )

    def append_module(
        self,
        module: BaseModule,
        /,
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
            len(jax.tree.leaves(self.modules)),
            module,
            key=key,
        )

    def prepend_module(
        self,
        module: BaseModule,
        /,
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
            0,
            module,
            key=key,
        )

    def pop_module_by_index(
        self,
        index: int,
        /,
    ) -> BaseModule:
        r"""
        Remove and return a module at a specific index in the model.
        Parameters
        ----------
            index
                Index of the module to remove.
        Returns
        -------
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
                "Cannot pop module from since "
                "modules are not stored in a list, tuple, or "
                "dictionary."
            )

    def pop_module_by_key(
        self,
        key: jax.tree_util.KeyPath | str | int,
        /,
    ) -> BaseModule:
        r"""
        Remove and return a module by key or index in the model.
        Parameters
        ----------
            key
                Key or index of the module to remove.
        Returns
        -------
            The removed module.
        """
        self.reset()
        if isinstance(key, str):
            if not isinstance(self.modules, Dict):
                raise KeyError(
                    f"Cannot pop module '{key}' by name since "
                    "modules are not stored in a "
                    "dictionary."
                )
            return self.modules.pop(key)
        elif isinstance(key, int):
            if not isinstance(self.modules, (List, Tuple)):
                raise KeyError(
                    f"Cannot pop module '{key}' by index since "
                    "modules are not stored in a "
                    "list or tuple."
                )
            return self.pop_module_by_index(key)
        else:
            raise TypeError(f"Invalid key type {type(key)} for module access.")

    def __add__(self, other: BaseModule) -> Model:
        r"""
        Overload the + operator to append a module or model to the current
        model.

        Parameters
        ----------
            other
                Module or model to append.

        Returns
        -------
            New model with the other module or model appended.
        """
        new_model = self.__class__(self.modules)
        new_model.append_module(other)
        return new_model

    # aliases
    append = append_module
    prepend = prepend_module
    insert = insert_module
    add_module = append_module
    add = append_module
    pop = pop_module_by_key

    def get_hyperparameters(self) -> HyperParams:
        """
        Get the hyperparameters of the model as a dictionary.

        Returns
        -------
            Dict[str, Any]
                Dictionary containing the hyperparameters of the model.
        """
        return {
            "modules": self.modules,
            "rng": self.rng,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "_optimizer_state": self._optimizer_state,
        }

    def set_hyperparameters(self, hyperparams: HyperParams, /) -> None:
        """
        Set the hyperparameters of the model from a dictionary.

        Parameters
        ----------
            hyperparams : Dict[str, Any]
                Dictionary containing the hyperparameters of the model.
        """
        # reset input_shape and output_shape to force re-compilation
        modules_ = hyperparams.get("modules", None)
        if modules_ is not None:
            self.modules = modules_
        rng_ = hyperparams.get("rng", None)
        if rng_ is not None:
            self.set_rng(rng_)
        self.input_shape = hyperparams["input_shape"]
        self.output_shape = hyperparams["output_shape"]
        optimizer_state_ = hyperparams.get("_optimizer_state", None)
        if optimizer_state_ is not None:
            # ensure the structure of the optimizer state matches the model
            # parameters struct
            param_struct = jax.tree.structure(self.get_params())
            adam_struct = jax.tree.structure(
                optimizer_state_,
                is_leaf=lambda x: isinstance(x, OptimizerState),
            )
            if param_struct != adam_struct:
                raise ValueError(
                    "Adam state structure does not match model parameters "
                    "structure."
                )
            self._optimizer_state = optimizer_state_

    @jaxtyped(typechecker=beartype)
    def serialize(self) -> Dict[str, Serializable]:
        """
        Serialize the model to a dictionary. This is done by serializing the
        model's parameters/metadata and then serializing each module.

        Returns
        -------
            Dict[str, Serializable]
        """

        module_fulltypenames = jax.tree.map(
            lambda m: str(type(m)), self.modules
        )
        module_typenames = jax.tree.map(
            lambda m: m.__class__.__name__, self.modules
        )
        module_modules = jax.tree.map(lambda m: m.__module__, self.modules)
        module_names = jax.tree.map(lambda m: m.name, self.modules)

        serialized_modules = jax.tree.map(
            lambda m: m.serialize(), self.modules
        )

        model_structure = jax.tree.structure(self.modules)

        # serialize rng key
        key_data = jax.random.key_data(self.get_rng())

        # serialize optimizer_state if applicable
        optimizer_struct = jax.tree.structure(self.get_params())
        serial_optimizer_state = (
            [
                a_s.serialize()
                for a_s in jax.tree.leaves(
                    self._optimizer_state,
                    is_leaf=lambda x: isinstance(x, OptimizerState),
                )
            ]
            if self._optimizer_state
            else None
        )

        return {
            "module_typenames": module_typenames,
            "module_modules": module_modules,
            "module_fulltypenames": module_fulltypenames,
            "module_names": module_names,
            "serialized_modules": serialized_modules,
            "model_structure": model_structure,
            "key_data": key_data,
            "optimizer_state": serial_optimizer_state,
            "optimizer_struct": optimizer_struct,
            "package_version": pmm.__version__,
            "model_version": self.__version__,
            "self_serialized": super().serialize(),
        }

    def deserialize(
        self,
        data: Dict[str, Serializable],
        /,
        *,
        strict_package_version: bool = False,
    ) -> None:
        """
        Deserialize the model from a dictionary. This is done by deserializing
        the model's parameters/metadata and then deserializing each module.

        Parameters
        ----------
            data
                Dictionary containing the serialized model data.

            strict_package_version
                If True, raises an error if the package version used to
                serialize the model does not match the current package version.
                Default is False.
        """
        self.reset()

        # read the version of the package this model was serialized with
        current_version = parse(pmm.__version__)
        package_version = parse(str(data["package_version"]))

        if current_version != package_version:
            if strict_package_version:
                raise ValueError(
                    "Model was saved with package version "
                    f"{package_version}, but current package version is "
                    f"{current_version}."
                )

        # read the model version, and handle any backward compatibility here
        model_version = parse(str(data["model_version"]))
        current_model_version = parse(str(self.__version__))
        if model_version != current_model_version:
            # nothing to do yet
            pass

        # initialize the modules
        self.modules = jax.tree.map(
            lambda mod_name, mod_module: getattr(
                sys.modules[mod_module], mod_name
            )(),
            data["module_typenames"],
            data["module_modules"],
        )

        # deserialize the modules
        jax.tree.map(
            lambda m, sm: m.deserialize(
                sm, strict_package_version=strict_package_version
            ),
            self.modules,
            data["serialized_modules"],
        )

        # check that the structure matches
        if jax.tree.structure(self.modules) != data["model_structure"]:
            raise ValueError(
                "Deserialized model structure does not match the expected "
                "structure."
            )

        # deserialize the rng key
        key = jax.random.wrap_key_data(data["key_data"])
        self.set_rng(key)

        # deserialize optimizer_state if applicable
        if (
            data["optimizer_state"] is not None
            and data["optimizer_struct"] is not None
        ):
            optimizer_state = [
                OptimizerState.deserialize(a_s_data)
                for a_s_data in data["optimizer_state"]
            ]
            self._optimizer_state = jax.tree.unflatten(
                data["optimizer_struct"], optimizer_state
            )

        # remove rng from data["self_serialized"] since it's already
        # deserialized above
        data["self_serialized"].pop("rng", None)
        # remove _optimizer_state from data["self_serialized"] since it's
        # already deserialized above
        data["self_serialized"].pop("_optimizer_state", None)

        # let default deserialization handle the rest of the attributes
        super().deserialize(
            data["self_serialized"],
            strict_package_version=strict_package_version,
        )

    @jaxtyped(typechecker=beartype)
    def save(self, file: str | IO | Path, /) -> None:
        """
        Save the model to a file.

        Parameters
        ----------
            file
                File to save the model to.
        """

        # if everything serializes correctly, we can save the model with just
        # savez
        data: Dict[str, Serializable] = self.serialize()

        if isinstance(file, str):
            file = file if file.endswith(".npz") else file + ".npz"
        np.savez(file, **data)

    @jaxtyped(typechecker=beartype)
    def save_compressed(self, file: str | IO | Path, /) -> None:
        """
        Save the model to a compressed file.

        Parameters
        ----------
            file
                File to save the model to.
        """
        # if everything serializes correctly, we can save the model with just
        # savez_compressed
        data: Dict[str, Serializable] = self.serialize()

        if isinstance(file, str):
            file = file if file.endswith(".npz") else file + ".npz"

        # jax.numpy doesn't have savez_compressed, so we use numpy
        onp.savez_compressed(file, **data)

    def load(
        self, file: str | IO | Path, /, *, strict_package_version=False
    ) -> None:
        """
        Load the model from a file. Supports both compressed and uncompressed

        Parameters
        ----------
            file
                File to load the model from.

            strict_package_version
                If True, raises an error if the package version used to
                serialize the model does not match the current package version.
                Default is False.
        """
        if isinstance(file, str):
            file = file if file.endswith(".npz") else file + ".npz"
        # jax numpy load supports both compressed and uncompressed npz files
        data = np.load(file, allow_pickle=True)

        # first convert data to a dictionary
        data = {key: data[key] for key in data.files}

        # then convert certain items back to lists or bare values
        if isinstance(data["module_typenames"], (onp.ndarray, np.ndarray)):
            data["module_typenames"] = data["module_typenames"].tolist()
        if isinstance(data["module_modules"], (onp.ndarray, np.ndarray)):
            data["module_modules"] = data["module_modules"].tolist()
        if isinstance(data["serialized_modules"], (onp.ndarray, np.ndarray)):
            data["serialized_modules"] = data["serialized_modules"].tolist()
        if isinstance(data["model_structure"], (onp.ndarray, np.ndarray)):
            data["model_structure"] = data["model_structure"].item()
        if isinstance(data["self_serialized"], (onp.ndarray, np.ndarray)):
            data["self_serialized"] = data["self_serialized"].item()
        if isinstance(data["optimizer_state"], (onp.ndarray, np.ndarray)):
            data["optimizer_state"] = data["optimizer_state"].tolist()
        if isinstance(data["optimizer_struct"], (onp.ndarray, np.ndarray)):
            data["optimizer_struct"] = data["optimizer_struct"].item()

        # deserialize the model
        self.deserialize(data, strict_package_version=strict_package_version)

    @classmethod
    def from_file(
        cls, file: str | IO | Path, /, *, strict_package_version: bool = False
    ) -> "Model":
        """
        Load a model from a file and return an instance of the Model class.

        Parameters
        ----------
            file : str
                File to load the model from.

            strict_package_version
                If True, raises an error if the package version used to
                serialize the model does not match the current package version.
                Default is False.

        Returns
        -------
            Model
                An instance of the Model class with the loaded parameters.
        """
        model = cls()
        model.load(file, strict_package_version=strict_package_version)
        return model
