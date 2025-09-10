from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as np

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
    feature.

    See Also
    --------
    torch.nn.PReLU
        PyTorch implementation of PReLU activation function.
    LeakyReLU
        Non-parametric ReLU activation function with a fixed negative slope.
    """

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

        self.a = None  # learnable parameter(s), will be set in compilation
        self.input_shape = None  # input shape, will be set in compilation

    def name(self) -> str:
        return f"PReLU(real={self.real})"

    def is_ready(self) -> bool:
        return (self.a is not None) and (self.input_shape is not None)

    def _get_callable(self) -> Callable:
        return lambda params, input_NF, training, state, rng: (
            jax.nn.leaky_relu(
                input_NF,
                negative_slope=params[0],
            ),
            state,  # state is not used in this module, return it unchanged
        )

    def compile(self, rng: Any, input_shape: tuple[int, ...]) -> None:
        # if the module is already ready, just verify the input shape
        if self.is_ready():
            if input_shape != self.input_shape:
                raise ValueError(
                    f"Input shape mismatch: expected {self.input_shape}, "
                    f"got {input_shape}"
                )
            return

        self.input_shape = input_shape

        if self.single_parameter:
            a_shape = (1,)
        else:
            a_shape = input_shape

        rng_areal, rng_aimag = jax.random.split(rng)

        if self.real:
            self.a = self.init_magnitude * jax.random.normal(
                rng_areal, a_shape
            )
        else:
            self.a = self.init_magnitude * (
                jax.random.normal(rng_areal, a_shape)
                + 1j * jax.random.normal(rng_aimag, a_shape)
            )

    def get_output_shape(
        self, input_shape: tuple[int, ...]
    ) -> tuple[int, ...]:
        return input_shape

    def get_hyperparameters(self) -> dict[str, Any]:
        return {
            "single_parameter": self.single_parameter,
            "init_magnitude": self.init_magnitude,
        }

    def set_hyperparameters(self, hyperparams: dict[str, Any]) -> None:
        if self.a is not None:
            raise ValueError(
                "Cannot set hyperparameters after the module has parameters"
            )

        super(PReLU, self).set_hyperparameters(hyperparams)

    def get_params(self) -> tuple[np.ndarray, ...]:
        return (self.a,)

    def set_params(self, params: tuple[np.ndarray, ...]) -> None:
        if not isinstance(params, tuple) or not all(
            isinstance(p, np.ndarray) for p in params
        ):
            raise ValueError("params must be a tuple of numpy arrays")
        if len(params) != 1:
            raise ValueError(f"Expected 1 parameter array, got {len(params)}")

        self.a = params[0]

        if np.iscomplexobj(self.a) and self.real:
            raise ValueError(
                "Parameter 'a' must be real-valued, but got complex-valued"
                " array"
            )

        if self.input_shape is not None:
            expected_shape = (
                (1,) if self.single_parameter else self.input_shape
            )
            if self.a.shape != expected_shape:
                raise ValueError(
                    f"Parameter 'a' shape mismatch: expected {expected_shape},"
                    f" got {self.a.shape}"
                )
        elif self.single_parameter and self.a.shape != (1,):
            raise ValueError(
                "Parameter 'a' shape mismatch: expected (1,), got"
                f" {self.a.shape}"
            )
