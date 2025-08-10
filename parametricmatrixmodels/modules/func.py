from __future__ import annotations

import base64
from inspect import signature
from typing import Any, Callable

import dill
import jax
import jax.numpy as np

from .basemodule import BaseModule


class Func(BaseModule):
    """
    Module that implements a general element-wise function, optionally with
    trainable parameters and state.
    """

    def __init__(
        self,
        f: (
            Callable[[np.ndarray], np.ndarray]
            | Callable[
                [tuple[np.ndarray, ...], np.ndarray],
                np.ndarray,
            ]
            | Callable[
                [tuple[np.ndarray, ...], np.ndarray, tuple[np.ndarray, ...]],
                tuple[np.ndarray, tuple[np.ndarray, ...]],
            ]
            | Callable[
                [
                    tuple[np.ndarray, ...],
                    np.ndarray,
                    tuple[np.ndarray, ...],
                    Any,
                ],
                tuple[np.ndarray, tuple[np.ndarray, ...]],
            ]
        ) = None,
        fname: str = None,
        params: tuple[np.ndarray, ...] = None,
        state: tuple[np.ndarray, ...] = (),
    ) -> None:
        """
        Parameters
        ----------

        f
            A function that performs the modules operation. It can take only
            the input features and return only the output features (if there
            are no trainable parameters), or the tuple of trainable parameters
            and the input features and return the output features, or the tuple
            of trainable parameters, the input features and module state and
            return the output features and new state, or the trainable
            parameters, input features, module state, and a JAX rng
            key and return the output features and new state.
            This function will be applied element-wise to the input data. The
            output shape need not match the input shape, but the function
            should return a constant shape for all inputs of a given shape.

            Summary of allowed signatures:

            1. ``f(input_NF: Array) -> Array``
                used in the case of no trainable parameters, no state, and no
                rng.

            2. ``f(params: tuple[Array, ...], input_NF: Array) -> Array``
                used in the case of trainable parameters, no state, and no rng.

            3. ``f(params: tuple[Array, ...], input_NF: Array, state: \
tuple[Array, ...]) -> tuple[Array, tuple[Array, ...]]``
                used in the case of trainable parameters, state, but no rng.

            4. ``f(params: tuple[Array, ...], input_NF: Array, state: \
tuple[Array, ...], rng: Any) -> \
tuple[Array, tuple[Array, ...]]``
                used in the case of trainable parameters, state, and rng.

        fname
            Name of the function. If not provided, the function's Pythonic name
            will be used.

        params
            Initial trainable parameters of the module. This can be used to
            store any tuple of numpy arrays that the function might need to
            maintain state across calls. If not provided, an empty tuple will
            be used. If the function requires trainable parameters, their
            initial values must be provided here.

        state
            Initial state of the module. This can be used to store any tuple of
            numpy arrays that the function might need to maintain state across
            calls. If not provided, an empty tuple will be used.
        """

        self._handle_inputs(
            f=f,
            fname=fname,
            params=params,
            state=state,
            input_shape=None,  # will be set during compile
            output_shape=None,  # will be set during compile
        )

    def _handle_inputs(
        self,
        f: (
            Callable[[np.ndarray], np.ndarray]
            | Callable[
                [tuple[np.ndarray, ...], np.ndarray],
                np.ndarray,
            ]
            | Callable[
                [tuple[np.ndarray, ...], np.ndarray, tuple[np.ndarray, ...]],
                tuple[np.ndarray, tuple[np.ndarray, ...]],
            ]
            | Callable[
                [
                    tuple[np.ndarray, ...],
                    np.ndarray,
                    tuple[np.ndarray, ...],
                    Any,
                ],
                tuple[np.ndarray, tuple[np.ndarray, ...]],
            ]
        ),
        fname: str = None,
        params: tuple[np.ndarray, ...] = None,
        state: tuple[np.ndarray, ...] = (),
        input_shape: tuple[int, ...] = None,
        output_shape: tuple[int, ...] = None,
    ) -> None:
        """
        Handle the hyperparameters for the Func module.

        This includes input validation and setting up underlying
        hyperparameters from the provided inputs
        """

        if f is not None and not callable(f):
            raise ValueError("f must be a callable function")
        if f is not None:
            sig = signature(f)
            self._orig_signature = sig
            self._orig_f = f
            if len(sig.parameters) < 1 or len(sig.parameters) > 3:
                raise ValueError(
                    "Function f "
                    f"({fname if fname is not None else f.__name__}) "
                    "must take either one, two, or three arguments: "
                    "input features, state, and a JAX rng key."
                )

            requires_params = True
            if len(sig.parameters) == 1:
                # no trainable parameters, no state, no rng
                self.f = lambda p, input_NF, state, rng: (f(input_NF), state)
                self.f.__name__ = f.__name__
                requires_params = False
            elif len(sig.parameters) == 2:
                # trainable parameters, no state, no rng
                self.f = lambda p, input_NF, state, rng: (
                    f(p, input_NF),
                    state,
                )
                self.f.__name__ = f.__name__
            elif len(sig.parameters) == 3:
                # trainable parameters, state, no rng
                self.f = lambda p, input_NF, state, rng: f(p, input_NF, state)
                self.f.__name__ = f.__name__
            elif len(sig.parameters) == 4:
                # trainable parameters, state, and rng
                self.f = lambda p, input_NF, state, rng: f(
                    p, input_NF, state, rng
                )
                self.f.__name__ = f.__name__
            else:
                raise ValueError(
                    "Function f "
                    f"({fname if fname is not None else f.__name__}) "
                    "must take either one, two, three, or four "
                    "arguments: input features, state, and a JAX rng key."
                )

        # validation for the output signature will be done during compile

        # ensure params is a tuple of numpy arrays
        if params is not None:
            if not isinstance(params, tuple) or not all(
                isinstance(p, np.ndarray) for p in params
            ):
                raise ValueError("params must be a tuple of numpy arrays")
        elif f is not None and requires_params:
            # Func module cannot randomly initialize trainable parameters of
            # general functions, so the ininital params must be provided
            raise ValueError(
                f"Function f ({fname if fname is not None else f.__name__}) "
                "requires trainable parameters, but no "
                "initial parameters were provided."
            )
        else:
            params = ()

        # ensure state is a tuple of numpy arrays
        if state is not None:
            if not isinstance(state, tuple) or not all(
                isinstance(s, np.ndarray) for s in state
            ):
                raise ValueError("state must be a tuple of numpy arrays")

        self.fname = fname if fname is not None else f.__name__ if f else None
        self.state = state if state is not None else ()
        self.params = params if params is not None else ()
        self.input_shape = input_shape
        self.output_shape = output_shape

    def name(self) -> str:
        fname_ = (
            self.fname
            if self.fname
            else self.f.__name__ if self.f else "uninitialized func"
        )

        return f"Func({fname_})"

    def is_ready(self) -> bool:
        return (
            self.f is not None
            and self.input_shape is not None
            and self.output_shape is not None
        )

    def get_num_trainable_floats(self) -> int | None:
        if not self.is_ready():
            return None

        # count the total number of floats in the parameters
        # multiplying by 2 if the parameters are complex
        param_count = sum(
            np.prod(np.array(p.shape)) * (2 if np.iscomplexobj(p) else 1)
            for p in self.get_params()
        )
        # can be 0
        return param_count

    def _get_callable(self) -> Callable:
        return lambda params, input_NF, training, state, rng: self.f(
            params, input_NF, state, rng
        )

    def compile(self, rng: Any, input_shape: tuple[int, ...]) -> None:
        self.input_shape = input_shape
        self.output_shape = self.get_output_shape(input_shape)

    def get_output_shape(
        self, input_shape: tuple[int, ...]
    ) -> tuple[int, ...]:
        if not self.is_ready():
            # compute the output shape if the module is not ready
            # validate output signature and get output shape
            # easiest way to do this is to call the function with dummy data
            dummy_input = np.ones(input_shape, dtype=np.float32)
            # add batch dimension
            dummy_input = dummy_input[None, :]  # shape (1, num_features)

            # the Func module must already have params initialized
            dummy_output_and_state = self.f(
                self.params, dummy_input, self.state, jax.random.key(0)
            )

            if (
                not isinstance(dummy_output_and_state, tuple)
                or len(dummy_output_and_state) != 2
            ):
                # give more meaningful error message based on what the original
                # function signature was
                if len(self._orig_signature.parameters) == 1:
                    raise ValueError(
                        f"Function f ({self.fname}) must return a single"
                        " output array when its signature has only one"
                        " argument (input features)."
                    )
                elif len(self._orig_signature.parameters) == 2:
                    raise ValueError(
                        f"Function f ({self.fname}) must return a single"
                        " output array when its signature has two arguments"
                        " (trainable parameters and input features)."
                    )
                elif (
                    len(self._orig_signature.parameters) == 3
                    or len(self._orig_signature.parameters) == 4
                ):
                    raise ValueError(
                        f"Function f ({self.fname}) must return a tuple of"
                        " output array and state when its signature has three"
                        " or four arguments (trainable parameters, input"
                        " features, state, [rng key])."
                    )

            dummy_output, dummy_state = dummy_output_and_state

            if not isinstance(dummy_output, np.ndarray):
                raise ValueError(
                    f"Function f ({self.fname}) must return an output array as"
                    " the first output, but got"
                    f" {type(dummy_output).__name__} instead."
                )
            if not isinstance(dummy_state, tuple) or not all(
                isinstance(s, np.ndarray) for s in dummy_state
            ):
                raise ValueError(
                    f"Function f ({self.fname}) must return a state tuple as"
                    " the second output, but got"
                    f" {type(dummy_state).__name__} instead."
                )

            # ensure the output shape is at least 2D (i.e. it has a batch
            # dimension)
            if len(dummy_output.shape) < 2:
                raise ValueError(
                    f"Function f ({self.fname}) must return an output array"
                    " with at least 2 dimensions (batch dimension and"
                    f" features), but got {dummy_output.shape} instead."
                )

            return dummy_output.shape[1:]

        if input_shape != self.input_shape:
            raise ValueError(
                f"Input shape {input_shape} does not match "
                f"the expected input shape {self.input_shape}. "
                "Call compile() with the correct input shape first."
            )
        return self.output_shape

    def get_hyperparameters(self) -> dict[str, Any]:
        return {
            "f": self.f,
            "fname": self.fname,
            "params": self.params,
            "state": self.state,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
        }

    def set_hyperparameters(self, hyperparams: dict[str, Any]) -> None:
        if not isinstance(hyperparams, dict):
            raise ValueError("hyperparams must be a dictionary")

        # ensure all required keys are present
        if "f" not in hyperparams:
            raise ValueError("hyperparams must contain the key 'f'")

        self._handle_inputs(
            f=hyperparams["f"],
            fname=hyperparams.get("fname"),
            params=hyperparams.get("params"),
            state=hyperparams.get("state"),
            input_shape=hyperparams.get("input_shape"),
            output_shape=hyperparams.get("output_shape"),
        )

    def get_params(self) -> tuple[np.ndarray, ...]:
        return self.params

    def set_params(self, params: tuple[np.ndarray, ...]) -> None:
        if not isinstance(params, tuple) or not all(
            isinstance(p, np.ndarray) for p in params
        ):
            raise ValueError("params must be a tuple of numpy arrays")

        if len(params) != len(self.params):
            raise ValueError(
                f"Expected {len(self.params)} parameters, but got "
                f"{len(params)} instead."
            )

        self.params = params

    def serialize(self) -> dict[str, Any]:
        # serializing "f" isn't straightforward so it has to be handled
        # differently from the default implementation

        # first call the default implementation to get most of the
        # module serialized
        serial = super().serialize()

        # get the original function's python module and name
        # only the original function will be serialized, not the
        # morphed "f" function, self._handle_inputs will
        # take care of that when deserializing

        raw = dill.dumps(self._orig_f)
        encoded = base64.b64encode(raw).decode("utf-8")
        serial["hyperparameters"]["f"] = encoded

        return serial

    def deserialize(self, serial: dict[str, Any]) -> None:
        raw = base64.b64decode(serial["hyperparameters"]["f"].encode("utf-8"))
        self._orig_f = dill.loads(raw)

        # reset `f` in the hyperparameters
        serial["hyperparameters"]["f"] = self._orig_f

        # call the default implementation to set the rest of the serialized
        # object
        super().deserialize(serial)
