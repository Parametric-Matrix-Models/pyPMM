from __future__ import annotations

import jax
import jax.numpy as np
from beartype import beartype
from jaxtyping import Array, Inexact, jaxtyped

from ..tree_util import is_shape_leaf, is_single_leaf
from ..typing import (
    Any,
    Data,
    DataShape,
    HyperParams,
    ModuleCallable,
    Params,
    State,
    Tuple,
)
from .basemodule import BaseModule


class AffineTensor(BaseModule):
    r"""
    Module that builds a parametric tensor that is affine in the
    input features.

    :math:`M(x) = M_0 + x_1 M_1 + ... + x_p M_p`
    where :math:`M_0, M_1, ..., M_p` are (trainable) tensors,
    :math:`x_1, ..., x_p` are the input features

    Only accepts PyTree data that has a single leaf array that is 1D, excluding
    the batch dimension. The PyTree structure is preserved in the output.

    See Also
    --------
    AffineHermitianTensor
        Module that builds a parametric Hermitian tensor that is affine in the
        input features with optional smoothing.
    """

    def __init__(
        self,
        tensor_shape: Tuple[int, ...] | None = None,
        Ms: Inexact[Array, "_ ..."] | None = None,
        init_magnitude: float = 1e-2,
        bias_term: bool = True,
        real: bool = True,
    ) -> None:
        r"""
        Create an ``AffineTensor`` module.

        Parameters
        ----------
            tensor_shape
                Size of the PMM tensors (n1 x n2 x ...)
            Ms
                Optional array of tensors :math:`M_0, M_1, ..., M_p` that
                define the parametric affine tensor. If not provided, the
                tensors will be randomly
                initialized when the module is compiled. Default is ``None``.
            init_magnitude
                Optional initial magnitude of the random tensors, used when
                initializing the module. Default is ``1e-2``.
            bias_term
                If ``True``, include the bias term :math:`M_0` in the equation
                for :math:`M(x)`. Default is ``True``.
            real
                If ``True``, the tensors will be real-valued. If ``False``,
                the tensors will be complex-valued. Default is ``True``.
        """

        # input validation
        if tensor_shape is not None and (
            not isinstance(tensor_shape, tuple)
            or not all(isinstance(s, int) and s > 0 for s in tensor_shape)
        ):
            raise ValueError(
                "tensor_shape must be a tuple of positive integers"
            )
        if Ms is not None:
            if not isinstance(Ms, np.ndarray):
                raise ValueError("Ms must be a numpy array")
            tensor_shape = tensor_shape or Ms.shape[1:]
            if Ms.shape != (Ms.shape[0], *tensor_shape):
                raise ValueError(
                    "Ms must be a ND array of shape (input_size+1,"
                    " *tensor_shape)"
                    f" [{(Ms.shape[0], *tensor_shape)}], got {Ms.shape}"
                )
        self.tensor_shape = tensor_shape
        self.bias_term = bias_term
        self.Ms = Ms  # tensors M0, M1, ..., Mp
        self.init_magnitude = init_magnitude
        self.real = real

    @property
    def name(self) -> str:
        return (
            f"AffineTensor({self.tensor_shape}"
            f"{'' if self.bias_term else ', no bias'}"
            f"{', real' if self.real else ', complex'})"
        )

    def is_ready(self) -> bool:
        return self.Ms is not None

    def _get_callable(self) -> ModuleCallable:

        if self.bias_term:

            @jaxtyped(typechecker=beartype)
            def make_affine_tensor(
                Ms: Inexact[Array, "p ..."],
                features: Inexact[Array, "b p-1"],
            ) -> Inexact[Array, "b ..."]:

                # convert to common dtype, this should be traced out
                dtype = np.result_type(Ms.dtype, features.dtype)
                Ms = Ms.astype(dtype)
                features = features.astype(dtype)

                M = Ms[0][None] + np.einsum("ni,i...->n...", features, Ms[1:])

                return M

        else:

            @jaxtyped(typechecker=beartype)
            def make_affine_tensor(
                Ms: Inexact[Array, "p ..."],
                features: Inexact[Array, "b p"],
            ) -> Inexact[Array, "b ..."]:
                # convert to common dtype, this should be traced out
                dtype = np.result_type(Ms.dtype, features.dtype)
                Ms = Ms.astype(dtype)
                features = features.astype(dtype)

                M = np.einsum("ni,i...->n...", features, Ms)

                return M

        @jaxtyped(typechecker=beartype)
        def affine_tensor(
            params: Params,
            data: Data,
            training: bool,
            state: State,
            rng: Any,
        ) -> Tuple[Data, State]:

            # tree map over the data to preserve the PyTree structure

            # compile will have validated that there is only one leaf that is
            # a 1D array, excluding the batch dimension
            M = jax.tree.map(
                lambda x: make_affine_tensor(params, x),
                data,
            )

            return (M, state)

        return affine_tensor

    def compile(self, rng: Any, input_shape: DataShape) -> None:

        valid, leaf = is_single_leaf(
            input_shape, ndim=1, is_leaf=is_shape_leaf
        )

        # input shape must be a PyTree with a single leaf that is 1D
        if not valid:
            raise ValueError(
                "Input shape must be a PyTree with a single leaf consisting of"
                " a 1D array, got: {input_shape}"
            )

        # number of tensors is number of features + 1 (bias) if bias is used
        p = leaf[0] + 1 if self.bias_term else leaf[0]

        # if the module is already ready, just verify the input shape
        if self.is_ready():
            if self.Ms.shape[0] != p:
                raise ValueError(
                    f"Input shape {leaf} does not match the expected "
                    f"number of features {self.Ms.shape[0] - 1} "
                )
            return

        rng_Mreal, rng_Mimag = jax.random.split(rng, 2)

        self.Ms = self.init_magnitude * (
            jax.random.normal(
                rng_Mreal,
                (p, *self.tensor_shape),
                dtype=np.float32,
            )
        )

        if not self.real:
            self.Ms = self.Ms.astype(np.complex64) + self.init_magnitude * (
                +1j
                * jax.random.normal(
                    rng_Mimag,
                    (p, *self.tensor_shape),
                    dtype=np.complex64,
                )
            )

    def get_output_shape(self, input_shape: DataShape) -> DataShape:
        # return (n,m) with the same PyTree structure as input_shape, so long
        # as the input shape is valid
        valid, _ = is_single_leaf(input_shape, ndim=1, is_leaf=is_shape_leaf)
        if not valid:
            raise ValueError(
                "Input shape must be a PyTree with a single leaf consisting of"
                " a 1D array, got: {input_shape}"
            )
        return jax.tree.map(
            lambda x: self.tensor_shape,
            input_shape,
            is_leaf=is_shape_leaf,
        )

    def get_hyperparameters(self) -> HyperParams:
        return {
            "tensor_shape": self.tensor_shape,
            "init_magnitude": self.init_magnitude,
            "bias_term": self.bias_term,
            "real": self.real,
        }

    def set_hyperparameters(self, hyperparams: HyperParams) -> None:
        if self.Ms is not None:
            raise ValueError(
                "Cannot set hyperparameters after the module has parameters"
            )

        super().set_hyperparameters(hyperparams)

    def get_params(self) -> Params:
        return self.Ms

    def set_params(self, params: Params) -> None:
        if not isinstance(params, np.ndarray):
            raise ValueError("Params must be a numpy array")

        Ms = params

        expected_shape = (
            Ms.shape[0] if self.Ms is None else self.Ms.shape[0],
            *self.tensor_shape,
        )

        if Ms.shape != expected_shape:
            raise ValueError(
                "Ms must be a 3D array of shape (input_size"
                f"{'+1' if self.bias_term else ''}, *tensor_shape) "
                f"[{expected_shape}], got {Ms.shape}"
            )
        self.Ms = Ms
