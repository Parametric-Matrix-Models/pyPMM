from __future__ import annotations

import sys
from typing import Any, Callable

import jax.numpy as np

from ._helpers import subsets_to_string
from .basemodule import BaseModule


class SubsetModule(BaseModule):
    """
    Meta-module that applies a given Module to a subset of the input data and
    optionally passes the remaining input data through unchanged.

    Diagrammatic example:

       [x0] -> ...................................... -> [x0]
       [x1] -> ...................................... -> [x1]
       [x2] -> [ SubsetModule([2:], APPEND, module) ] -> [module(x2, x3)]
       [x3] -> [                                    ]

    The output shape need not match the input shape, as the module may change
    the shape of the data, as in the example above, where the module takes two
    input features (x2, x3) and produces one output feature (module(x2, x3)).

    The output of the module that is applied can either be prepended or
    appended to the unchanged input data, as specified by the `prepend`
    parameter in the constructor.

    Additionally, the unchanged input data can be passed through unchanged
    alongside the output of the module, or it can be ignored, depending on the
    `passthrough` parameter in the constructor. A diagrammatic example of a
    passthrough SubsetModule is as follows:

    [x0] -> .............................................. -> [x0]
    [x1] -> .............................................. -> [x1]
    [x2] -> [ SubsetModule([2:], APPEND, PASSTHROUGH, m) ] -> [x2]
    [x3] -> [                                            ] -> [x3]
            [                                            ] -> [m(x2, x3)]

    """

    def __init__(
        self,
        subset: tuple[slice, ...] | tuple[np.ndarray, ...] = None,
        module: BaseModule = None,
        prepend: bool = True,
        axis: int = 0,
        passthrough: bool = False,
    ):
        """
        Parameters
        ----------
        subset
            A tuple of slices or index arrays indicating which parts of the
            input data to apply the module to. The number of slices must match
            the shape of the input data, not including the batch dimension.
            For example, for
            input data of shape (num_samples, num_features), a subset of all
            except the first two features would be specified as
            (slice(2, None),).
        module
            The module to apply to the specified subset of the input data.
        prepend
            If True, the output of the module will be prepended to the
            unchanged input data. If False, the output will be appended.
            Defaults to True.
        axis
            The axis along which to concatenate the output of the module and
            the unchanged input data. Defaults to 0 (i.e., along the first
            feature dimension). This does not include the batch dimension.
        passthrough
            If True, the unchanged input data will be passed through alongside
            the output of the module. If False, the unchanged input data will
            be dropped. Defaults to False.
        """
        if subset is not None and not isinstance(subset, tuple):
            raise TypeError(
                "Subset must be a tuple of slices or index arrays, "
                f"not {type(subset)}"
            )

        self.subset = subset
        self.module = module
        self.prepend = prepend
        self.axis = axis
        self.passthrough = passthrough
        self.input_shape = None  # will be set in compile
        self.output_shape = None  # will be set in compile

    def name(self) -> str:
        subname = self.module.name()
        subset_str = subsets_to_string(self.subset)
        pend_str = "PREPEND" if self.prepend else "APPEND"
        pass_str = "PASSTHROUGH" if self.passthrough else "CONSUME"
        return (
            f"SubsetModule({subset_str}, {pass_str}, {pend_str}, "
            f"axis={self.axis}, {subname})"
        )

    def is_ready(self) -> bool:
        return (
            self.module.is_ready()
            and self.input_shape is not None
            and self.output_shape is not None
        )

    def get_num_trainable_floats(self) -> int | None:
        return self.module.get_num_trainable_floats()

    def _get_inverse_subset_mask(
        self, input_shape: tuple[int, ...]
    ) -> np.ndarray:
        # make a mask for the inverse of the subset
        # if passthrough, then the mask is just the entire input
        mask = np.ones(input_shape, dtype=bool)
        if not self.passthrough:
            mask = mask.at[self.subset].set(False)

        return mask

    def _get_callable(self) -> Callable:
        if not self.is_ready():
            raise ValueError(
                "SubsetModule is not ready. "
                "Call compile() with the input shape and rng."
            )

        # first add a full slice to the subset to include the batch dimension
        full_subset = (slice(None),) + self.subset

        mask = self._get_inverse_subset_mask(self.input_shape)

        subcallable = self.module._get_callable()

        def _callable(
            params: tuple[np.ndarray, ...],
            input_NF: np.ndarray,
            training: bool = False,
            state: tuple[np.ndarray, ...] = (),
            rng: Any = None,
        ) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:

            # apply the module to the subset of the input data
            subset_input = input_NF[full_subset]
            module_output, new_state = subcallable(
                params, subset_input, training, state, rng
            )

            # prepend or append the module output
            if self.prepend:
                output_NF = np.concatenate(
                    (module_output, input_NF[:, mask]),
                    axis=self.axis + 1,
                )
            else:
                output_NF = np.concatenate(
                    (input_NF[:, mask], module_output),
                    axis=self.axis + 1,
                )

            return output_NF, new_state

        return _callable

    def compile(self, rng: Any, input_shape: tuple[int, ...]) -> None:
        self.input_shape = input_shape

        # get subinput shape to pass to the module's compile method
        subset_input_zeros = np.zeros(input_shape, dtype=np.float32)[
            self.subset
        ]
        self.module.compile(rng, subset_input_zeros.shape)
        self.output_shape = self.get_output_shape(input_shape)

    def get_output_shape(
        self, input_shape: tuple[int, ...]
    ) -> tuple[int, ...]:
        # easiest way to get the output shape is to perform a dummy forward
        # pass with zeros and the module's `get_output_shape` method

        # don't need to include the batch dimension in any of the shapes here

        mask = self._get_inverse_subset_mask(input_shape)
        input_zeros = np.zeros(input_shape, dtype=np.float32)
        subset_input = input_zeros[self.subset]
        module_output_shape = self.module.get_output_shape(subset_input.shape)

        module_output_zeros = np.zeros(module_output_shape, dtype=np.float32)

        if self.prepend:
            output_zeros = np.concatenate(
                (module_output_zeros, input_zeros[mask]), axis=self.axis
            )
        else:
            output_zeros = np.concatenate(
                (input_zeros[mask], module_output_zeros), axis=self.axis
            )

        return output_zeros.shape

    def get_hyperparameters(self) -> dict[str, Any]:
        return {
            "subset": self.subset,
            "prepend": self.prepend,
            "axis": self.axis,
            "passthrough": self.passthrough,
            "module": self.module,
        }

    def set_hyperparameters(self, hyperparams: dict[str, Any]) -> None:
        if self.input_shape is not None or self.output_shape is not None:
            raise ValueError(
                "Cannot set hyperparameters after compile. "
                "Please set hyperparameters before calling compile()."
            )

        super(SubsetModule, self).set_hyperparameters(hyperparams)

    def get_params(self) -> tuple[np.ndarray, ...]:
        return self.module.get_params()

    def set_params(self, params: tuple[np.ndarray, ...]) -> None:
        self.module.set_params(params)

    def get_state(self) -> tuple[np.ndarray, ...]:
        return self.module.get_state()

    def set_state(self, state: tuple[np.ndarray, ...]) -> None:
        self.module.set_state(state)

    def serialize(self) -> dict[str, Any]:
        # call the base class serialize method to get most of the data
        # serialized
        serial = super(SubsetModule, self).serialize()

        # then replace the few fields that weren't actually serialized:
        # module and subset

        module_typename = self.module.__class__.__name__
        module_module = self.module.__module__
        module_serial = self.module.serialize()

        # serialize the subsets by converting any slices to their start, stop,
        # and step values
        serial_subsets = (
            [
                (
                    (slc.start, slc.stop, slc.step)
                    if isinstance(slc, slice)
                    else slc
                )
                for slc in self.subset
            ]
            if self.subset
            else None
        )

        serial["hyperparameters"]["subset"] = serial_subsets
        serial["hyperparameters"]["module"] = {
            "type": module_typename,
            "module": module_module,
            "data": module_serial,
        }

        return serial

    def deserialize(self, data: dict[str, Any]) -> None:
        # replace non-standard serialized fields with the partially
        # deserialized values before calling the base class's deserialize

        subset_slices = data["hyperparameters"]["subset"]

        # deserialize the subsets by converting any tuples back to slices
        if subset_slices is not None:
            subset = tuple(
                slice(*slc) if isinstance(slc, tuple) else slc
                for slc in subset_slices
            )
        else:
            subset = None

        module_module = data["hyperparameters"]["module"]["module"]
        module_typename = data["hyperparameters"]["module"]["type"]
        module_data = data["hyperparameters"]["module"]["data"]

        module = getattr(sys.modules[module_module], module_typename)()
        module.deserialize(module_data)

        data["hyperparameters"]["subset"] = subset
        data["hyperparameters"]["module"] = module

        super(SubsetModule, self).deserialize(data)
