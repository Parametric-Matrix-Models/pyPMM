from __future__ import annotations

import sys
from typing import Any, Callable

import jax
import jax.numpy as np

from .basemodule import BaseModule


class MultiModule(BaseModule):
    r"""
    Meta-module that applies multiple modules in sequence.

    See Also
    --------
    Model
        Class that chains multiple modules together into a single model.

    """

    def __init__(
        self,
        *args: BaseModule,
    ):
        """
        Initialize a ``MultiModule``.

        Parameters
        ----------
        *args : BaseModule
            The modules to apply in sequence.

        Raises
        ------
        TypeError
            If any of the provided arguments is not a ``BaseModule``.
        """

        # check if args is empty
        if len(args) == 0:
            self.modules = ()
        else:
            for module in args:
                if not isinstance(module, BaseModule):
                    raise TypeError(
                        "All arguments to MultiModule must be BaseModule "
                        "instances."
                    )
            self.modules = args

        self.reset()

    def __getitem__(self, idx: int) -> BaseModule:
        return self.modules[idx]

    def reset(self) -> None:
        self.input_shape = None
        self.output_shape = None
        self.parameter_counts = None
        self.state_counts = None

    def name(self) -> str:

        num_modules = len(self.modules)
        mod_idx_width = len(str(num_modules - 1)) if num_modules > 0 else 1

        namestr = "MultiModule("
        input_shape = self.input_shape if self.input_shape else None
        for i, module in enumerate(self.modules):
            input_shape = (
                module.get_output_shape(input_shape) if input_shape else None
            )
            comment = module.name().startswith("#")
            namestr += f"\n  ({i:>{mod_idx_width}}): {module}" + (
                f" -> {input_shape}" if input_shape and not comment else ""
            )
        namestr += "\n)"
        return namestr

    def is_ready(self) -> bool:
        return all(module.is_ready() for module in self.modules)

    def get_num_trainable_floats(self) -> int | None:
        module_nums = [
            module.get_num_trainable_floats() for module in self.modules
        ]
        if any(num is None for num in module_nums):
            return None
        return sum(module_nums)

    def _get_callable(self) -> Callable:
        if not self.is_ready():
            raise ValueError(
                "MultiModule is not ready. "
                "Call compile() with the input shape and rng."
            )

        module_callables = [module._get_callable() for module in self.modules]

        def _callable(
            params: tuple[np.ndarray, ...],
            input_NF: np.ndarray,
            training: bool = False,
            state: tuple[np.ndarray, ...] = (),
            rng: Any = None,
        ) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
            param_index = 0
            state_index = 0
            # split rng key into a key for each module
            rngs = jax.random.split(rng, len(self.modules))
            for idx, module in enumerate(self.modules):
                param_count = self.parameter_counts[idx]
                state_count = self.state_counts[idx]
                module_params = tuple(
                    params[param_index : param_index + param_count]
                )
                module_state = tuple(
                    state[state_index : state_index + state_count]
                )

                input_NF, new_module_states = module_callables[idx](
                    module_params,
                    input_NF,
                    training,
                    module_state,
                    rngs[idx],
                )
                # update the states
                state = (
                    state[:state_index]
                    + new_module_states
                    + state[state_index + state_count :]
                )
                # update indices
                param_index += param_count
                state_index += state_count

            return input_NF, state

        return _callable

    def compile(self, rng: Any, input_shape: tuple[int, ...]) -> None:
        self.input_shape = input_shape

        for i, module in enumerate(self.modules):
            rng, modrng = jax.random.split(rng)
            module.compile(modrng, input_shape)
            input_shape = module.get_output_shape(input_shape)
        self.output_shape = input_shape

        # get parameter and state counts
        self.parameter_counts = [
            len(module.get_params()) for module in self.modules
        ]
        self.state_counts = [
            len(module.get_state()) for module in self.modules
        ]

    def get_output_shape(
        self, input_shape: tuple[int, ...]
    ) -> tuple[int, ...]:
        for module in self.modules:
            input_shape = module.get_output_shape(input_shape)
        return input_shape

    def get_hyperparameters(self) -> dict[str, Any]:
        return {
            "modules": self.modules,
            "parameter_counts": self.parameter_counts,
            "state_counts": self.state_counts,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
        }

    def set_hyperparameters(self, hyperparams: dict[str, Any]) -> None:
        super(MultiModule, self).set_hyperparameters(hyperparams)

    def get_params(self) -> tuple[np.ndarray, ...]:
        if not self.is_ready():
            raise ValueError(
                "MultiModule is not ready. "
                "Call compile() with the input shape and rng."
            )
        return tuple(
            param for module in self.modules for param in module.get_params()
        )

    def set_params(self, params: tuple[np.ndarray, ...]) -> None:
        if len(params) != sum(self.parameter_counts):
            raise ValueError(
                f"Expected {sum(self.parameter_counts)} parameters, "
                f"but got {len(params)}."
            )

        # set the parameters for each module
        param_index = 0
        for module in self.modules:
            count = len(module.get_params())
            module.set_params(params[param_index : param_index + count])
            param_index += count

    def get_state(self) -> tuple[np.ndarray, ...]:
        if not self.is_ready():
            raise ValueError(
                "MultiModule is not ready. "
                "Call compile() with the input shape and rng."
            )
        return tuple(
            state for module in self.modules for state in module.get_state()
        )

    def set_state(self, state: tuple[np.ndarray, ...]) -> None:
        if not self.is_ready():
            raise ValueError(
                "MultiModule is not ready. "
                "Call compile() with the input shape and rng."
            )
        if len(state) != sum(self.state_counts):
            raise ValueError(
                f"Expected {sum(self.state_counts)} state variables, "
                f"but got {len(state)}."
            )

        # set the state for each module
        state_index = 0
        for module in self.modules:
            count = len(module.get_state())
            module.set_state(state[state_index : state_index + count])
            state_index += count

    def serialize(self) -> dict[str, Any]:
        module_fulltypenames = [str(type(module)) for module in self.modules]
        module_typenames = [
            module.__class__.__name__ for module in self.modules
        ]
        module_modules = [module.__module__ for module in self.modules]
        module_names = [module.name() for module in self.modules]

        serialized_modules = [module.serialize() for module in self.modules]

        return {
            "module_typenames": module_typenames,
            "module_modules": module_modules,
            "module_fulltypenames": module_fulltypenames,
            "module_names": module_names,
            "serialized_modules": serialized_modules,
        }

    def deserialize(self, data: dict[str, Any]) -> None:
        self.reset()

        module_typenames = data["module_typenames"]
        module_modules = data["module_modules"]

        # initialize modules
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
