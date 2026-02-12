from abc import abstractmethod
from typing import final

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
    Params,
    State,
    Tuple,
)

from ..tree_util import get_shapes, is_shape_leaf
from .basemodule import BaseModule


class FuncBase(BaseModule):
    """
    Base class for simple non-trainable function modules. Not to be
    instantiated directly.
    """

    __version__: str = "0.0.0"

    def __init__(self):
        """
        Initialize the function module.
        """
        pass

    @abstractmethod
    def get_hyperparameters(self) -> HyperParams:
        """
        Get the hyperparameters of the function module.

        Returns
        -------
            An empty dictionary, as function modules do not have
            hyperparameters.
        """
        raise NotImplementedError(
            "Subclasses must implement `get_hyperparameters`."
        )

    @abstractmethod
    def f(self, data: Data) -> Data:
        """
        Apply the function to the input data

        Parameters
        ----------
        data
            Input Data (PyTree of arrays).

        Returns
        -------
            Output Data (PyTree of arrays).
        """
        raise NotImplementedError("Subclasses must implement `f`.")

    @property
    def name(self) -> str:
        return f"FuncBase({self.__class__.__name__})"

    @final
    def is_ready(self) -> bool:
        """
        Funcs are always ready to be used.

        Returns
        -------
            Always returns True.
        """
        return True

    @final
    def get_num_trainable_floats(self) -> int | None:
        """
        Funcs do not have trainable parameters.

        Returns
        -------
            Always returns 0.
        """
        return 0

    @final
    def _get_callable(self) -> ModuleCallable:
        """
        Get the callable for the function module.

        Returns
        -------
            A callable that applies the function to the input data in the form
            the PMM library expects.
        """

        @jaxtyped(typechecker=beartype)
        def func_callable(
            params: Params,
            data: Data,
            training: bool,
            state: State,
            rng: Any,
        ) -> Tuple[Data, State]:
            return self.f(data), state

        return func_callable

    @final
    def compile(self, rng: Any, input_shape: DataShape) -> None:
        """
        Compile the function module. No action is needed for function modules.

        Parameters
        ----------
        rng
            Random number generator state.
        input_shape
            Shape of the input arrays.
        """
        # make sure that f can handle the input shape
        self.get_output_shape(input_shape)

    @final
    def get_output_shape(self, input_shape: DataShape) -> DataShape:
        """
        Get the output shape of the function given an input shape.

        Parameters
        ----------
        input_shape
            Shape of the input arrays.

        Returns
        -------
            Output shapes after applying the function.
        """

        # only way to do this automatically is to run a dummy input

        # add batch dimension to all shapes
        input_w_batch_shape = jax.tree.map(
            lambda s: (1,) + s, input_shape, is_leaf=is_shape_leaf
        )
        dummy_input = jax.tree.map(
            lambda s: np.zeros(s, dtype=np.float32),
            input_w_batch_shape,
            is_leaf=is_shape_leaf,
        )
        try:
            dummy_output = self.f(dummy_input)
        except Exception as e:
            raise RuntimeError(
                "Failed to compute output shape in `get_output_shape`. "
                "Make sure the function `f` can handle F32 inputs with shape "
                f"{get_shapes(dummy_input)}, which includes a leading size-1 "
                "batch dimension."
            ) from e

        output_shape = get_shapes(dummy_output, axis=slice(1, None))

        return output_shape

    @final
    def get_params(self) -> Params:
        """
        Get the parameters of the function module, of which there are none.

        Returns
        -------
            An empty tuple, as function modules do not have parameters.
        """
        return ()

    @final
    def set_params(self, params: Params) -> None:
        return
