from __future__ import annotations

import random
import sys
import warnings

import jax
import jax.numpy as np
import numpy as onp
from beartype import beartype
from jaxtyping import jaxtyped
from packaging.version import parse

import parametricmatrixmodels as pmm

from .model_util import (
    ModelCallable,
    ModelModules,
    ModelParams,
    ModelState,
    autobatch,
    safecast,
    strfmt_pytree,
)
from .modules import BaseModule
from .training import make_loss_fn, train
from .typing import (
    Any,
    Callable,
    Data,
    DataShape,
    Dict,
    Tuple,
)


class Model(BaseModule):
    r"""

    Abstract base class for all models. Do not instantiate this class directly.

    A ``Model`` is a PyTree of modules that can be trained and
    evaluated. Inputs are passed through each module to produce
    outputs. ``Model``s are also ``BaseModule``s, so they can be
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
            f"{self.name()}(\n"
            + strfmt_pytree(
                self.modules, indent=0, indentation=1, base_indent_str="  "
            )
            + "\n)"
        )

    def __init__(
        self,
        modules: ModelModules | BaseModule | None = None,
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

    def reset(self) -> None:
        self.input_shape: DataShape | None = None
        self.output_shape: DataShape | None = None
        self.callable: ModelCallable | None = None
        self.grad_callable_params = None
        self.grad_callable_params_options = None
        self.grad_callable_inputs = None
        self.grad_callable_inputs_options = None

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

        raise NotImplementedError(
            f"{self.name()}.compile() not implemented. Must be implemented by "
            "subclasses."
        )

    def get_output_shape(self, input_shape: DataShape) -> DataShape:
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
            f"{self.name()}.get_output_shape() not implemented. Must be"
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

    def set_params(self, params: ModelParams) -> None:
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

        if not self.is_ready():
            raise RuntimeError(
                f"{self.name()} is not ready. Call compile() first."
            )

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

        if not self.is_ready():
            raise RuntimeError(
                f"{self.name()} is not ready. Call compile() first."
            )

        return jax.tree.map(lambda m: m.get_state(), self.modules)

    def set_state(self, state: ModelState) -> None:
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
        if not self.is_ready():
            raise RuntimeError(
                f"{self.name()} is not ready. Call compile() first."
            )

        # this will fail if the structure of self.modules isn't a prefix of the
        # structure of state
        try:
            jax.tree.map(lambda m, s: m.set_state(s), self.modules, state)
        except ValueError as e:
            raise ValueError(
                "Structure of state does not match structure of modules. "
                "The structure of the modules must be a prefix of the "
                "structure of the state."
            ) from e

    def get_rng(self) -> Any:
        return self.rng

    def set_rng(self, rng: Any) -> None:
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
            f"{self.name()}._get_callable() not implemented. Must be"
            " implemented by subclasses."
        )

    def __call__(
        self,
        X: Data,
        dtype: Any = np.float64,
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
            raise RuntimeError(
                f"{self.name()} is not ready. Call compile() first."
            )

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
        X: Data,
        dtype: Any = np.float64,
        rng: Any | int | None = None,
        return_state: bool = False,
        update_state: bool = False,
        fwd: bool | None = None,
        max_batch_size: int | None = None,
    ) -> Tuple[Data, ModelState] | Data:
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
        """

        if not self.is_ready():
            raise RuntimeError(
                f"{self.name()} is not ready. Call compile() first."
            )
        if self.callable is None:
            self.callable = jax.jit(self._get_callable(), static_argnums=(2,))

        def get_num_elems(count: int, arr: np.ndarray) -> int:
            return count + arr.size

        num_input_elems = jax.tree.reduce(
            get_num_elems, self.input_shape, initializer=0
        )
        num_output_elems = jax.tree.reduce(
            get_num_elems, self.output_shape, initializer=0
        )

        # if fwd is None, decide based on input and output sizes
        # fwd is more efficient when the number of input elements is less
        # than the number of output elements in general
        fwd = fwd if fwd is not None else (num_input_elems < num_output_elems)

        batched = (max_batch_size is None) or (max_batch_size > 1)

        # prepare the grad callable if not already done
        if (self.grad_callable_inputs is None) or (
            self.grad_callable_inputs_options != (batched, fwd)
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

                # take the diagonal (batch-wise jacobian)
                @jaxtyped(typechecker=beartype)
                def take_diag(
                    arr: np.ndarray,
                    input_shape: Tuple[int, ...],
                    output_shape: Tuple[int, ...],
                ) -> np.ndarray:
                    batch_size = arr.shape[0]
                    input_ndim = len(input_shape)
                    output_ndim = len(output_shape)
                    diag_indices = (
                        (np.arange(batch_size),)
                        + (slice(None),) * output_ndim
                        + (np.arange(batch_size),)
                        + (slice(None),) * input_ndim
                    )
                    return arr[diag_indices]

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
                    # each leaf of Y is and array of shape
                    # (batch_size, output_dim1, output_dim2, ...,
                    # batch_size, input_dim1, input_dim2, ...)
                    # we want to take the diagonal along the two batch axes

                    Y_diag = jax.tree.map(
                        take_diag,
                        Y,
                        self.input_shape,
                        self.output_shape,
                        is_leaf=lambda x: isinstance(x, tuple)
                        and all(isinstance(i, int) for i in x),
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
        X: Data,
        dtype: Any = np.float64,
        rng: Any | int | None = None,
        return_state: bool = False,
        update_state: bool = False,
        fwd: bool | None = None,
        max_batch_size: int | None = None,
    ) -> Tuple[ModelParams, ModelState] | ModelParams:
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
        """

        if not self.is_ready():
            raise RuntimeError(
                f"{self.name()} is not ready. Call compile() first."
            )
        if self.callable is None:
            self.callable = jax.jit(self._get_callable(), static_argnums=(2,))

        def get_num_elems(count: int, arr: np.ndarray) -> int:
            return count + arr.size

        num_input_elems = jax.tree.reduce(
            get_num_elems, self.input_shape, initializer=0
        )
        num_output_elems = jax.tree.reduce(
            get_num_elems, self.output_shape, initializer=0
        )

        # if fwd is None, decide based on input and output sizes
        # fwd is more efficient when the number of input elements is less
        # than the number of output elements in general
        fwd = fwd if fwd is not None else (num_input_elems < num_output_elems)

        if self.grad_callable_params is None or (
            self.grad_callable_params_options != fwd
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

    def set_precision(self, prec: np.dtype | str | int) -> None:
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
            raise RuntimeError(
                f"{self.name()} is not ready. Call compile() first."
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

        for module in self.modules:
            module.set_precision(prec)

    # alias for set_precision method that returns self
    def astype(self, dtype: np.dtype | str) -> "Model":
        """
        Convenience wrapper to set_precision using the dtype argument, returns
        self.
        """
        self.set_precision(dtype)
        return self

    def train(
        self,
        X: np.ndarray,
        Y: np.ndarray | None = None,
        Y_unc: np.ndarray | None = None,
        X_val: np.ndarray | None = None,
        Y_val: np.ndarray | None = None,
        Y_val_unc: np.ndarray | None = None,
        loss_fn: str | Callable = "mse",
        lr: float = 1e-3,
        batch_size: int = 32,
        num_epochs: int = 100,
        convergence_threshold: float = 1e-12,
        early_stopping_patience: int = 10,
        early_stopping_tolerance: float = 1e-6,
        # advanced options
        initialization_seed: int | None = None,
        callback: Callable | None = None,
        unroll: int | None = None,
        verbose: bool = True,
        batch_seed: int | None = None,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        clip: float = 1e3,
    ) -> None:

        # check if the model is ready
        if not self.is_ready():
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

    def serialize(self) -> Dict[str, Any]:
        """
        Serialize the model to a dictionary. This is done by serializing the
        model's parameters/metadata and then serializing each module.

        Returns
        -------
            Dict[str, Any]
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
