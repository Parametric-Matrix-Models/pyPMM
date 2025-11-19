import jax.numpy as np

from parametricmatrixmodels.typing import (
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


class Constant(BaseModule):
    r"""
    Module that always returns a constant value.
    """

    def __init__(
        self,
        constant: Data | np.ndarray | float | complex | None = None,
    ) -> None:
        """
        Parameters
        ----------
        constant: Data | np.ndarray | float | complex | None
            The constant value to return. If None, the module is uninitialized
            and must be set later via `set_hyperparameters`.
        """
        # check if constant is a scalar
        if constant is not None and np.isscalar(constant):
            constant = np.array(constant)

        self.constant = constant

    def name(self) -> str:
        return f"Constant({type(self.constant)})"

    def is_ready(self) -> bool:
        return self.constant is not None

    def _get_callable(self) -> ModuleCallable:

        def callable(
            params: Params,
            data: Data,
            training: bool,
            state: State,
            rng: Any,
        ) -> Tuple[Data, State]:
            return self.constant, state

        return callable

    def compile(self, rng: Any, input_shape: DataShape) -> None:
        return

    def get_output_shape(self, input_shape: DataShape) -> DataShape:
        return input_shape

    def get_hyperparameters(self) -> HyperParams:
        return {
            "constant": self.constant,
        }

    def set_hyperparameters(self, hyperparams: HyperParams) -> None:
        super(Constant, self).set_hyperparameters(hyperparams)

    def get_params(self) -> Params:
        return ()

    def set_params(self, params: Params) -> None:
        return
