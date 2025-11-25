import jax
import jax.numpy as np
from beartype import beartype
from jaxtyping import Array, Inexact, PyTree, jaxtyped

from ..tree_util import is_shape_leaf
from ..typing import (
    Any,
    Data,
    DataShape,
    HyperParams,
    ModuleCallable,
    Params,
    Tuple,
)
from .basemodule import BaseModule


class Einsum(BaseModule):
    """
    Module that implements Einsum operations. Supports both bare arrays
    (``ArrayData``) and PyTrees of arrays (``Data``).
    """

    def __init__(
        self,
        einsum_str: PyTree[str] | str | None = None,
        params: Params | Inexact[Array, "..."] = None,
        shapes: (
            PyTree[Tuple[int, ...], "Params"] | Tuple[int, ...] | None
        ) = None,
        init_magnitude: float = 1e-2,
        real: bool = True,
    ) -> None:
        """
        Initialize an ``Einsum`` module.

        The einsum operation is defined by the provided einsum string, which
        must specify all indices except for the batch index. For PyTree inputs,
        the indices for each operation on each leaf must be specified in a
        PyTree of strings with the same structure as the input data.

        Parameters
        ----------
        einsum_str
            The einsum string defining the operation(s) to be performed.
            Each string must specify all indices except for the batch index.
            The form for each string is
            ``"<input_indices>,<weight_indices>-><output_indices>"``. See
            Examples for details.
            If ``None``, this value must be provided by
            ``set_hyperparameters`` prior to compilation.
        params
            The parameters (weight arrays) for the einsum operation(s). If the
            input is a bare array, this should be a single array. If the input
            is a PyTree, this should be a PyTree of arrays with the same
            structure as the input data. If ``None``, then ``shape``s must be
            provided to initialize the weights randomly during compilation.
        shapes
            The shapes of the parameter arrays. If the input is a bare array,
            this should be a single tuple. If the input is a PyTree, this
            should be a PyTree of tuples with the same structure as the input
            data. Each shape must be compatible with the corresponding
            einsum string.
            If ``params`` is provided, then the shapes are inferred from the
            provided parameter arrays and must match this argument if given.
        init_magnitude
            Magnitude for the random initialization of weights.
            Default is ``1e-2``.
        real
            If ``True``, the weights and biases will be real-valued. If
            ``False``, they will be complex-valued. Default is ``True``.

        Examples
        --------

        .. code-block:: python

            # Simple Matmul for 1D inputs
            # For an input array of shape (num_features,) (excluding batch
            # dimension), this will compute
            # output_j = sum_i input_i * W_ij, or out = in @ W
            # this is automatically applied over the batch dimension in the
            # input.
            einsum_module = Einsum("i,ij->j")

            # Contraction over a single index for multidimensional array input
            # For an input array of shape (H, W, C), this will compute
            # output_hk = sum_w input_hwc * W_wc, i.e. contracting over the W
            # index.
            einsum_module = Einsum("hwc,wk->hk")

            # PyTree input with different einsum operations for each leaf
            # For a PyTree input with three leaves, structured as:
            # PyTree([*, (*, *)]), with each leaf being a 1D array (excluding
            # the batch dimension), this will compute:
            # output[0]_j = sum_i input[0]_i * W0_ij
            # output[1]_j = sum_ijk input[1]_ij * W1_ijk
            # output[2]_j = sum_i input[2]_ii * W2_ij
            einsum_module = Einsum(
                einsum_str=[
                    "i,ij->j",
                    (
                        "ij,ijk->k",
                        "ii,ij->j",
                    ),
                ]
            )

        """

        # deal with shapes
        if params is None and shapes is None:
            raise ValueError(
                "If params is not provided, shapes must be provided to "
                "initialize the parameters during compilation."
            )
        if params is not None:
            # infer shapes from params
            def get_shape(p: np.ndarray) -> Tuple[int, ...]:
                return p.shape

            inferred_shapes = jax.tree.map(get_shape, params)

            if shapes is not None:
                # ensure shapes match inferred shapes
                def validate_shape(
                    inferred: Tuple[int, ...],
                    given: Tuple[int, ...],
                ) -> None:
                    if inferred != given:
                        raise ValueError(
                            f"Provided shape {given} does not match "
                            f"inferred shape {inferred} from params."
                        )

                jax.tree.map(
                    validate_shape,
                    inferred_shapes,
                    shapes,
                    is_leaf=is_shape_leaf,
                )

            shapes = inferred_shapes

        self.shapes = shapes

        # verify that einsum_str is valid (exactly one ',')
        # '->' is optional (can be inferred by numpy) and not checked here
        if einsum_str is not None:

            def validate_einsum_str(s: str) -> str:
                if s.count(",") != 1:
                    raise ValueError(
                        f"Invalid einsum string: {s}. Must contain exactly "
                        "one ','."
                    )
                return s

            self.einsum_str = jax.tree.map(validate_einsum_str, einsum_str)

        else:
            self.einsum_str = einsum_str
        self.init_magnitude = init_magnitude
        self.real = real

        # if params are provided, ensure their dimensions match with einsum_str
        if params is not None and einsum_str is not None:

            if isinstance(einsum_str, str) != isinstance(params, np.ndarray):
                raise ValueError(
                    "If einsum_str is a single string, params must be a "
                    "single array. If einsum_str is a PyTree of strings, "
                    "params must be a PyTree of arrays with the same "
                    "structure."
                )
            elif not isinstance(einsum_str, str) and not isinstance(
                params, np.ndarray
            ):
                einsum_structure = jax.tree.structure(einsum_str)
                param_structure = jax.tree.structure(params)
                if einsum_structure != param_structure:
                    raise ValueError(
                        "If einsum_str is a PyTree, params must be a PyTree "
                        "with the same structure. Got einsum_str structure "
                        f"{einsum_structure} and params structure "
                        f"{param_structure}."
                    )

            def validate_params(s: str, p: np.ndarray) -> np.ndarray:
                # extract weight indices from einsum string
                weight_indices = s.split(",")[1].split("->")[0]
                expected_ndim = len(weight_indices)
                if p.ndim != expected_ndim:
                    raise ValueError(
                        f"Parameter array has {p.ndim} dimensions, but "
                        f"expected {expected_ndim} based on einsum string "
                        f"{s}"
                    )

                if self.real and not np.isrealobj(p):
                    raise ValueError(
                        "Parameter array must be real-valued for a real module"
                    )
                if not self.real and np.isrealobj(p):
                    raise ValueError(
                        "Parameter array must be complex-valued for a "
                        "complex module"
                    )
                return p

            self.params = jax.tree.map(validate_params, einsum_str, params)
        else:
            self.params = params

        self._batch_einsum_str: PyTree[str] | str | None = None

    def name(self) -> str:
        return f"Einsum(einsum_str={self.einsum_str})"

    def is_ready(self) -> bool:
        return self._batch_einsum_str is not None and self.params is not None

    def _get_callable(self) -> ModuleCallable:

        @jaxtyped(typechecker=beartype)
        def einsum_callable(
            params: Params | Inexact[Array, "..."],
            input_data: Data,
        ) -> Data:
            return jax.tree.map(
                lambda s, p, x: np.einsum(s, x, p),
                self._batch_einsum_str,
                params,
                input_data,
            )

        return lambda p, d, t, s, k: (einsum_callable(p, d), s)

    def compile(self, rng: Any, input_shape: DataShape) -> None:
        if self.einsum_str is None:
            raise ValueError(
                "einsum_str must be set before compiling the module"
            )

        # ensure params are initialized
        if self.params is None:

            # assert that the tree structure of input_shape and shapes match
            input_shape_structure = jax.tree.structure(
                input_shape, is_leaf=is_shape_leaf
            )
            shapes_structure = jax.tree.structure(
                self.shapes, is_leaf=is_shape_leaf
            )
            if input_shape_structure != shapes_structure:
                raise ValueError(
                    "The structure of input_shape and shapes must match. "
                    f"Got input_shape structure {input_shape_structure} and "
                    f"shapes structure {shapes_structure}."
                )

            def initialize_param(
                shape: Tuple[int, ...],
                rng_key: Any,
            ) -> np.ndarray:
                if self.real:
                    p = self.init_magnitude * jax.random.normal(
                        rng_key,
                        shape,
                    )
                else:
                    p = self.init_magnitude * (
                        jax.random.normal(rng_key, shape)
                        + 1j * jax.random.normal(rng_key, shape)
                    )
                return p

            rngs = jax.random.split(
                rng,
                len(jax.tree.leaves(self.shapes, is_leaf=is_shape_leaf)),
            )
            rngs = jax.tree.unflatten(
                jax.tree.structure(self.shapes, is_leaf=is_shape_leaf),
                rngs,
            )
            self.params = jax.tree.map(
                initialize_param,
                self.shapes,
                rngs,
                is_leaf=is_shape_leaf,
            )

        # ensure the input_shapes are compatible with the einsum_str
        def validate_input_shape(s: str, shape: tuple[int, ...]) -> None:
            input_indices = s.split(",")[0]
            expected_ndim = len(input_indices)
            if len(shape) != expected_ndim:
                raise ValueError(
                    f"Input array has shape {shape}, but expected "
                    f"{expected_ndim} dimensions based on einsum string {s}"
                )

        jax.tree.map(
            validate_input_shape,
            self.einsum_str,
            input_shape,
            is_leaf=is_shape_leaf,
        )

        # to ensure the batch index ends up as the first index in the output
        # in the case where the output indices are not explicitly specified, we
        # need to choose the first alphabetical character, 'a'
        # thus, if 'a' is already in use, we need to swap it to any other
        # available character
        # the simplest way to do this without changing the einsum operation is
        # to cycle all the characters forward by one, i.e.
        # a->b, b->c, ..., z->A, A->B, ..., Z->a
        # where if 'Z' is used, we must throw an error

        # check for 'Z' in einsum strings
        def check_for_Z(s: str) -> None:
            if "Z" in s:
                raise ValueError(
                    "Einsum strings cannot contain the character 'Z', as it "
                    "is reserved for batch index handling."
                )

        jax.tree.map(check_for_Z, self.einsum_str)
        # create translation table
        orig = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        shifted = "bcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZa"
        translation_table = str.maketrans(orig, shifted)

        # apply translation table to einsum strings
        def shift_einsum_str(s: str) -> str:
            return s.translate(translation_table)

        self.einsum_str = jax.tree.map(shift_einsum_str, self.einsum_str)
        # define batch index character
        batch_char = "a"

        # add the batch index char to the einsum strings
        # it should be the first index in the input and output
        # if '->' is present, then the batch char is added immediately after it
        # otherwise, numpy will place the output indices in alphabetical order
        def add_batch_index(s: str) -> str:
            if "->" in s:
                input_part, output_part = s.split("->")
                new_input_part = batch_char + input_part
                new_output_part = batch_char + output_part
                return new_input_part + "->" + new_output_part
            else:
                return batch_char + s

        self._batch_einsum_str = jax.tree.map(add_batch_index, self.einsum_str)

    def get_output_shape(self, input_shape: DataShape) -> DataShape:
        if not self.is_ready():
            raise ValueError(
                "Module must be compiled before getting output shape."
            )

        dummy_params = jax.tree.map(lambda p: np.zeros_like(p), self.params)
        dummy_input = jax.tree.map(
            lambda shape: np.zeros((1,) + shape),
            input_shape,
            is_leaf=is_shape_leaf,
        )
        output = jax.tree.map(
            lambda s, p, x: np.einsum(s, x, p),
            self._batch_einsum_str,
            dummy_params,
            dummy_input,
        )
        return jax.tree.map(
            lambda o: o.shape,
            output,
        )

    def get_hyperparameters(self) -> HyperParams:
        return {
            "einsum_str": self.einsum_str,
            "init_magnitude": self.init_magnitude,
            "real": self.real,
        }

    def set_hyperparameters(self, hyperparams: HyperParams) -> None:
        super(Einsum, self).set_hyperparameters(hyperparams)

    def get_params(self) -> Params:
        return self.params

    def set_params(self, params: Params) -> None:
        # if params is not None, ensure their pytree structures match
        # and the shapes match
        if self.params is not None:

            if isinstance(self.params, np.ndarray):
                if not isinstance(params, np.ndarray):
                    raise ValueError(
                        "New parameters must be a single array to match "
                        "existing parameters."
                    )
                if self.params.shape != params.shape:
                    raise ValueError(
                        f"New parameter array has shape {params.shape}, but "
                        f"expected {self.params.shape}"
                    )

            else:

                def validate_param_shape(
                    existing_p: np.ndarray, new_p: np.ndarray
                ) -> None:
                    if existing_p.shape != new_p.shape:
                        raise ValueError(
                            f"New parameter array has shape {new_p.shape}, "
                            f"but expected {existing_p.shape}"
                        )

                jax.tree.map(
                    validate_param_shape,
                    self.params,
                    params,
                )

        self.params = params
