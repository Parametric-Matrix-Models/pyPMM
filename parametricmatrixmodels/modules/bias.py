import jax
import jax.numpy as np
from beartype import beartype
from jaxtyping import jaxtyped

from parametricmatrixmodels.typing import (
    Any,
    Data,
    DataShape,
    HyperParams,
    ModuleCallable,
    State,
    Tuple,
    TupleParams,
)

from .basemodule import BaseModule


@jaxtyped(typechecker=beartype)
class Bias(BaseModule):
    r"""
    A simple bias module that adds a (trainable by default) bias array
    (default) or scalar to the input. Can be real (default) or complex-valued.

    If the input is a ``PyTree`` of arrays, the same bias will be added to
    each leaf array and therefore the bias shape must match the shape of each
    leaf array (or be a scalar).
    """

    def __init__(
        self,
        bias: np.ndarray | float | complex | None = None,
        init_magnitude: float = 1e-2,
        real: bool = True,
        scalar: bool = False,
        trainable: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        bias
            Bias array or scalar. If None, it will be initialized randomly
        init_magnitude
            Magnitude for the random initialization of the bias.
            Default is ``1e-2``.
        real
            If ``True``, the biases will be real-valued. If
            ``False``, they will be complex-valued. Default is ``True``.
        scalar
            If ``True`` the bias will be a scalar shared across all input
            features. If ``False``, the bias will be a array with one entry
            per input feature. Default is ``False``.
        trainable
            If ``True``, the bias will be trainable. Default is ``True``.
        """
        self.bias = bias
        self.init_magnitude = init_magnitude
        self.real = real
        self.scalar = scalar
        self.trainable = trainable

        if self.bias is not None:
            # input validation
            if self.scalar and not np.isscalar(self.bias):
                raise ValueError(
                    "If scalar is True, bias must be a scalar or None"
                )
            if not self.scalar and not isinstance(self.bias, np.ndarray):
                raise ValueError(
                    "If scalar is False, bias must be a numpy array or None"
                )
            if self.real and not np.isrealobj(self.bias):
                raise ValueError("Bias must be real-valued for a real module")
            if not self.real and np.isrealobj(self.bias):
                raise ValueError(
                    "Bias must be complex-valued for a complex module"
                )

            if self.scalar:
                self.bias = np.array(self.bias).reshape((1,))

    def name(self) -> str:
        return f"Bias(real={self.real})"

    def is_ready(self) -> bool:
        return self.bias is not None

    def _get_callable(self) -> ModuleCallable:

        @jaxtyped(typechecker=beartype)
        def callable(
            params: TupleParams,
            data: Data,
            training: bool,
            state: State,
            rng: Any,
        ) -> Tuple[Data, State]:
            # tree map over data to add bias
            bias = params[0]

            def add_bias(x: np.ndarray) -> np.ndarray:
                return x + bias

            output = jax.tree.map(add_bias, data)
            return output, state

        return callable

    def compile(self, rng: Any, input_shape: DataShape) -> None:
        # if the module is already ready, just verify the input shape
        if (
            self.is_ready()
            and self.bias.shape != (1,)
            and self.bias.shape != ()
        ):
            # check if input shape is a single tuple (no PyTree)
            if (
                isinstance(input_shape, tuple)
                and self.bias.shape != input_shape
            ):
                raise ValueError(
                    f"Bias shape {self.bias.shape} does not match input "
                    f"shape {input_shape}"
                )
            # else if the input shape is a PyTree of tuples, all shapes must
            # match
        elif any(
            [
                shape != self.bias.shape
                for shape in jax.tree.leaves(input_shape)
            ]
        ):
            raise ValueError(
                f"Bias shape {self.bias.shape} does not match all input "
                f"shapes {jax.tree.leaves(input_shape)}"
            )

        shape = (1,) if self.scalar else input_shape

        # otherwise, initialize the bias
        subkey_real, subkey_imag = jax.random.split(rng, 2)

        if self.bias is None:
            if self.real:
                self.bias = (
                    jax.random.normal(subkey_real, shape) * self.init_magnitude
                )
            else:
                real_part = (
                    jax.random.normal(subkey_real, shape) * self.init_magnitude
                )
                imag_part = (
                    jax.random.normal(subkey_imag, shape) * self.init_magnitude
                )
                self.bias = real_part + 1j * imag_part

    def get_output_shape(self, input_shape: DataShape) -> DataShape:
        return input_shape

    def get_hyperparameters(self) -> HyperParams:
        return {
            "init_magnitude": self.init_magnitude,
            "real": self.real,
            "scalar": self.scalar,
            "trainable": self.trainable,
        }

    def set_hyperparameters(self, hyperparams: HyperParams) -> None:
        super(Bias, self).set_hyperparameters(hyperparams)

    def get_params(self) -> TupleParams:
        return (self.bias,)

    def set_params(self, params: TupleParams) -> None:
        if not isinstance(params, tuple) or not all(
            isinstance(p, np.ndarray) for p in params
        ):
            raise ValueError("params must be a tuple of numpy arrays")
        if len(params) != 1:
            raise ValueError(f"Expected 1 parameter array, got {len(params)}")
        if self.real and not np.isrealobj(params[0]):
            raise ValueError(
                "Parameter array 0 must be real-valued for a real module"
            )
        if not self.real and np.isrealobj(params[0]):
            raise ValueError(
                "Parameter array 0 must be complex-valued for a complex module"
            )
        if self.scalar and params[0].shape != (1,):
            raise ValueError(
                "Parameter array 0 must be a scalar array with shape (1,),"
                f" got {params[0].shape}"
            )

        self.bias = params[0]
