import jax
import jax.numpy as np
from beartype import beartype
from jaxtyping import jaxtyped

from parametricmatrixmodels.typing import (
    Any,
    ArrayData,
    Data,
    DataShape,
    HyperParams,
    ModuleCallable,
    Params,
    State,
    Tuple,
)

from .basemodule import BaseModule


class PReLU(BaseModule):
    r"""
    Element-wise Parametric Rectified Linear Unit (PReLU) activation function.

    .. math::

        \text{PReLU}(x) =
        \begin{cases}
        x, & \text{ if } x \ge 0 \\
        ax, & \text{ otherwise }
        \end{cases}

    where :math:`a` is a learnable parameter that controls the slope of the
    negative part of the function. :math:`a` can be either a single parameter
    shared across all input features, or a separate parameter for each input
    feature. Operates both on PyTrees and bare arrays.

    See Also
    --------
    torch.nn.PReLU
        PyTorch implementation of PReLU activation function.
    LeakyReLU
        Non-parametric ReLU activation function with a fixed negative slope.
    """

    __version__: str = "0.0.0"

    def __init__(
        self,
        single_parameter: bool = True,
        init_magnitude: float = 0.25,
        real: bool = True,
    ) -> None:
        """
        Create a new ``PReLU`` module.

        Parameters
        ----------
        single_parameter
            If ``True``, use a single learnable parameter for all input
            features. If ``False``, use a separate learnable parameter for each
            input feature. Default is ``True``.
        init_magnitude
            Initial magnitude of the learnable parameter(s).
            Default is ``0.25``.
        real
            If ``True``, the learnable parameter(s) will be real-valued.
            If ``False``, the learnable parameter(s) will be complex-valued.
            Default is ``True``.
        """

        self.single_parameter = single_parameter
        self.init_magnitude = init_magnitude
        self.real = real

        self.a: Params | float | complex | None = (
            None  # learnable parameter(s), will be set in compilation
        )
        self.input_shape: DataShape | None = (
            None  # input shape, will be set in compilation
        )

    @property
    def name(self) -> str:
        return f"PReLU(real={self.real}, single={self.single_parameter})"

    def is_ready(self) -> bool:
        return (self.a is not None) and (self.input_shape is not None)

    def _get_callable(self) -> ModuleCallable:
        @jaxtyped(typechecker=beartype)
        def prelu_array(
            arr: ArrayData,
            a: np.ndarray | float | complex,
        ) -> ArrayData:
            return jax.nn.leaky_relu(
                arr,
                negative_slope=a,
            )

        if self.single_parameter:

            def module_callable(
                params: Params,
                input_data: Data,
                training: bool,
                state: State,
                rng: Any,
            ) -> Tuple[Data, State]:
                a = params[0].squeeze()
                return (
                    jax.tree.map(lambda arr: prelu_array(arr, a), input_data),
                    state,
                )

        else:

            def module_callable(
                params: Params,
                input_data: Data,
                training: bool,
                state: State,
                rng: Any,
            ) -> Tuple[Data, State]:
                return (
                    jax.tree.map(
                        prelu_array,
                        input_data,
                        params,
                    ),
                    state,
                )

        return module_callable

    def compile(self, rng: Any, input_shape: DataShape) -> None:

        # if the module is already ready, check the input shape
        if self.is_ready():
            if self.input_shape != input_shape:
                raise ValueError(
                    "PReLU module has already been compiled with a different "
                    "input shape."
                )
            return

        self.input_shape = input_shape

        if self.single_parameter:
            a_shape = (1,)
        else:
            a_shape = input_shape

        def make_a(
            key: Any,
            shape: Tuple[int, ...],
        ) -> np.ndarray:

            if self.real:
                return self.init_magnitude * jax.random.normal(
                    key, shape, dtype=np.float32
                )
            else:
                rkey, ikey = jax.random.split(key)
                return self.init_magnitude * (
                    jax.random.normal(rkey, shape, dtype=np.complex64)
                    + 1j * jax.random.normal(ikey, shape, dtype=np.complex64)
                )

        if self.single_parameter:
            self.a = (make_a(rng, a_shape),)
        else:
            keys = jax.random.split(
                rng,
                len(
                    jax.tree.leaves(
                        input_shape,
                        is_leaf=lambda x: isinstance(x, tuple)
                        and all(isinstance(i, int) for i in x),
                    )
                ),
            )
            keys = jax.tree.unflatten(
                jax.tree.structure(
                    input_shape,
                    is_leaf=lambda x: isinstance(x, tuple)
                    and all(isinstance(i, int) for i in x),
                ),
                keys,
            )
            self.a = jax.tree.map(make_a, keys, a_shape)

    def get_output_shape(self, input_shape: DataShape) -> DataShape:
        return input_shape

    def get_hyperparameters(self) -> HyperParams:
        return {
            "single_parameter": self.single_parameter,
            "init_magnitude": self.init_magnitude,
            "input_shape": self.input_shape,
            "real": self.real,
        }

    def set_hyperparameters(self, hyperparams: HyperParams) -> None:
        if self.a is not None:
            raise ValueError(
                "Cannot set hyperparameters after the module has parameters"
            )

        super(PReLU, self).set_hyperparameters(hyperparams)

    def get_params(self) -> Params:
        return self.a

    def set_params(self, params: Params) -> None:
        # ensure the params match the expected shape
        if self.is_ready():
            if self.single_parameter:
                expected_shape = (1,)
                if len(params) != 1 or params[0].shape != expected_shape:
                    raise ValueError(
                        f"Expected single parameter of shape {expected_shape},"
                        f" got {params}"
                    )
            else:
                expected_shape = self.input_shape
                param_structure = jax.tree.structure(
                    params,
                )
                expected_structure = jax.tree.structure(
                    expected_shape,
                    is_leaf=lambda x: isinstance(x, tuple)
                    and all(isinstance(i, int) for i in x),
                )
                if param_structure != expected_structure:
                    raise ValueError(
                        "Expected parameters with structure"
                        f" {expected_structure}, got {param_structure}"
                    )

                def check_shape(
                    param: np.ndarray, shape: Tuple[int, ...]
                ) -> None:
                    if param.shape != shape:
                        raise ValueError(
                            f"Expected parameter of shape {shape}, got"
                            f" {param.shape}"
                        )

                jax.tree.map(check_shape, params, expected_shape)

        self.a = params
