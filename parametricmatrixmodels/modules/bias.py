from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as np

from .basemodule import BaseModule


class Bias(BaseModule):
    r"""
    A simple bias module that adds a (trainable by default) bias array
    (default) or scalar to the input. Can be real (default) or complex-valued.
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

    def _get_callable(self) -> Callable:
        return lambda params, input_NF, training, state, rng: (
            input_NF + params[0],
            state,  # state is not used in this module, return it unchanged
        )

    def compile(self, rng: Any, input_shape: tuple[int, ...]) -> None:
        # if the module is already ready, just verify the input shape
        if self.is_ready():
            if self.bias.shape != (1,) and self.bias.shape != input_shape:
                raise ValueError(
                    f"Bias shape {self.bias.shape} does not match input "
                    f"shape {input_shape}"
                )
            return

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

    def get_output_shape(
        self, input_shape: tuple[int, ...]
    ) -> tuple[int, ...]:
        return input_shape

    def get_hyperparameters(self) -> dict[str, Any]:
        return {
            "init_magnitude": self.init_magnitude,
            "real": self.real,
            "scalar": self.scalar,
            "trainable": self.trainable,
        }

    def set_hyperparameters(self, hyperparams: dict[str, Any]) -> None:
        super(Bias, self).set_hyperparameters(hyperparams)

    def get_params(self) -> tuple[np.ndarray, ...]:
        return (self.bias,)

    def set_params(self, params: tuple[np.ndarray, ...]) -> None:
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
