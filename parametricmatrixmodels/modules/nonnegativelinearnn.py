from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as np

from .basemodule import BaseModule


class NonnegativeLinearNN(BaseModule):
    """
    Module that implements a single linear NN layer with non-negative weights
    and biases.
    """

    def __init__(
        self,
        k: int = None,
        W: np.ndarray = None,
        b: np.ndarray = None,
        init_magnitude: float = 1e-2,
        real: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        k
            Number of output features.
        W
            Weight matrix of shape (num_features, k). If None, it will be
            initialized randomly.
        b
            Bias vector of shape (k,). If None, it will be initialized randomly
        init_magnitude
            Magnitude for the random initialization of weights and biases.
            Default is ``1e-2``.
        real
            If ``True``, the weights and biases will be real-valued. If
            ``False``, they will be complex-valued. Default is ``True``.
        """

        self.k = k
        self.W = W
        self.b = b
        self.init_magnitude = init_magnitude
        self.real = real

        # make sure either neither W nor b are provided, or both are provided
        if (W is None) != (b is None):
            raise ValueError(
                "Either both W and b must be provided, or neither."
            )
        if W is not None:
            self.p = W.shape[0]  # number of input features
        else:
            self.p = None

        # ensure that real is True
        if not real:
            raise NotImplementedError(
                "Complex-valued weights and biases are not supported in this "
                "module."
            )

        # ensure that W and b are real if provided
        if real and W is not None and not np.isrealobj(W):
            raise ValueError("W must be real-valued for real weights")
        if real and b is not None and not np.isrealobj(b):
            raise ValueError("b must be real-valued for real biases")

    def name(self) -> str:
        return f"NonnegativeLinearNN(k={self.k}, real={self.real})"

    def is_ready(self) -> bool:
        return (
            self.k is not None
            and self.p is not None
            and self.W is not None
            and self.b is not None
        )

    def _get_callable(self) -> Callable:
        # nonnegativity is ensured by taking the square of the weights and
        # biases
        return lambda params, input_NF, training, state, rng: (
            input_NF @ (params[0] ** 2) + (params[1] ** 2)[None, :],
            state,  # state is not used in this module, return it unchanged
        )

    def compile(self, rng: Any, input_shape: tuple[int, ...]) -> None:
        # input shape must be 1D
        if len(input_shape) != 1:
            raise ValueError(
                f"Input shape must be 1D, got {len(input_shape)}D shape: "
                f"{input_shape}"
            )

        # if the module is already ready, just verify the input shape
        if self.is_ready():
            if self.p != input_shape[0]:
                raise ValueError(
                    f"Input shape {input_shape} does not match the expected "
                    f"number of features {self.p}"
                )
            return

        # otherwise, initialize the matrices
        self.p = input_shape[0]  # number of input features

        subkey_real_W, subkey_imag_W, subkey_real_b, subkey_imag_b = (
            jax.random.split(rng, 4)
        )

        if self.W is None:
            if self.real:
                self.W = (
                    jax.random.normal(subkey_real_W, (self.p, self.k))
                    * self.init_magnitude
                )
            else:
                real_part = (
                    jax.random.normal(subkey_real_W, (self.p, self.k))
                    * self.init_magnitude
                )
                imag_part = (
                    jax.random.normal(subkey_imag_W, (self.p, self.k))
                    * self.init_magnitude
                )
                self.W = real_part + 1j * imag_part
        if self.b is None:
            if self.real:
                self.b = (
                    jax.random.normal(subkey_real_b, (self.k,))
                    * self.init_magnitude
                )
            else:
                real_part = (
                    jax.random.normal(subkey_real_b, (self.k,))
                    * self.init_magnitude
                )
                imag_part = (
                    jax.random.normal(subkey_imag_b, (self.k,))
                    * self.init_magnitude
                )
                self.b = real_part + 1j * imag_part

    def get_output_shape(
        self, input_shape: tuple[int, ...]
    ) -> tuple[int, ...]:
        return (self.k,)

    def get_hyperparameters(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "p": self.p,
            "init_magnitude": self.init_magnitude,
            "real": self.real,
        }

    def set_hyperparameters(self, hyperparams: dict[str, Any]) -> None:
        if self.W is not None or self.b is not None:
            raise ValueError(
                "Cannot set hyperparameters after the module has parameters"
            )

        super(NonnegativeLinearNN, self).set_hyperparameters(hyperparams)

    def get_params(self) -> tuple[np.ndarray, ...]:
        return (self.W, self.b)

    def set_params(self, params: tuple[np.ndarray, ...]) -> None:
        if not isinstance(params, tuple) or not all(
            isinstance(p, np.ndarray) for p in params
        ):
            raise ValueError("params must be a tuple of numpy arrays")
        if len(params) != 2:
            raise ValueError(f"Expected 2 parameter array, got {len(params)}")
        if params[0].shape != (self.p, self.k):
            raise ValueError(
                f"Parameter array 0 must be of shape ({self.p}, {self.k}, "
                f"{self.n}), got {params[0].shape}"
            )
        if params[1].shape != (self.k,):
            raise ValueError(
                f"Parameter array 1 must be of shape ({self.k},), "
                f"got {params[1].shape}"
            )
        if self.real and not np.isrealobj(params[0]):
            raise ValueError(
                "Parameter array 0 must be real-valued for a real module"
            )
        if not self.real and np.isrealobj(params[0]):
            raise ValueError(
                "Parameter array 0 must be complex-valued for a complex module"
            )
        if self.real and not np.isrealobj(params[1]):
            raise ValueError(
                "Parameter array 1 must be real-valued for a real module"
            )
        if not self.real and np.isrealobj(params[1]):
            raise ValueError(
                "Parameter array 1 must be complex-valued for a complex module"
            )

        self.W = params[0]
        self.b = params[1]
