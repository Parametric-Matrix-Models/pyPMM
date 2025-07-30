import sys
from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax.numpy as np

from ._helpers import subsets_to_string
from .basemodule import BaseModule


class SubsetModule(BaseModule):
    """
    Meta-module that applies a given Module to a subset of the input data and
    passes the remaining input data through unchanged.

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
        subset: Union[Tuple[slice, ...], Tuple[np.ndarray, ...]] = None,
        module: BaseModule = None,
        prepend: bool = True,
        axis: int = 0,
        passthrough: bool = False,
    ):
        """
        Initialize the SubsetModule with a subset of the input data and a
        module to apply to that subset.

        Parameters
        ----------
        subset : Union[Tuple[slice, ...], Tuple[np.ndarray, ...]], optional
            A tuple of slices or index arrays indicating which parts of the
            input data to apply the module to. The number of slices must match
            the shape of the input data, not including the batch dimension.
            For example, for
            input data of shape (num_samples, num_features), a subset of all
            except the first two features would be specified as
            (slice(2, None),).
        module : BaseModule
            The module to apply to the specified subset of the input data.
        prepend : bool, optional
            If True, the output of the module will be prepended to the
            unchanged input data. If False, the output will be appended.
            Defaults to True.
        axis : int, optional
            The axis along which to concatenate the output of the module and
            the unchanged input data. Defaults to 0 (i.e., along the first
            feature dimension). This does not include the batch dimension.
        passthrough : bool, optional
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
        """
        Returns the name of the module
        """
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

    def get_num_trainable_floats(self) -> Optional[int]:
        return self.module.get_num_trainable_floats()

    def _get_inverse_subset_mask(
        self, input_shape: Tuple[int, ...]
    ) -> np.ndarray:
        # make a mask for the inverse of the subset
        # if passthrough, then the mask is just the entire input
        mask = np.ones(input_shape, dtype=bool)
        if not self.passthrough:
            mask = mask.at[self.subset].set(False)

        return mask

    def _get_callable(self) -> Callable:
        """
        This method must return a jax-jittable and jax-gradable callable in the
        form of
        ```
        (
            params: Tuple[np.ndarray, ...],
            input_NF: np.ndarray[num_samples, num_features],
            training: bool,
            state: Tuple[np.ndarray, ...],
            rng: key<fry>
        ) -> (
                output_NF: np.ndarray[num_samples, num_output_features],
                new_state: Tuple[np.ndarray, ...]
            )
        ```
        That is, all hyperparameters are traced out and the callable depends
        explicitly only on a Tuple of parameter numpy arrays, the input array,
        the training flag, a state Tuple of numpy arrays, and a JAX rng key.

        The training flag will be traced out, so it doesn't need to be jittable
        """

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
            params: Tuple[np.ndarray, ...],
            input_NF: np.ndarray,
            training: bool = False,
            state: Tuple[np.ndarray, ...] = (),
            rng: Any = None,
        ) -> Tuple[np.ndarray, Tuple[np.ndarray, ...]]:

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

    def compile(self, rng: Any, input_shape: Tuple[int, ...]) -> None:
        """
        Compile the module to be used with the given input shape.

        This method should initialize the module's parameters and state based
        on the input shape and random key.

        This is needed since Models are built before the input data is given,
        so before training or inference can be done, the module needs to be
        compiled and each Module passes its output shape to the next Module's
        compile method.

        The rng key is used to initialize random parameters, if needed.

        Parameters
        ----------
        rng : Any
            JAX random key.
        input_shape : Tuple[int, ...]
            Shape of the input data, e.g. (num_features,).
        """
        self.input_shape = input_shape

        # get subinput shape to pass to the module's compile method
        subset_input_zeros = np.zeros(input_shape, dtype=np.float32)[
            self.subset
        ]
        self.module.compile(rng, subset_input_zeros.shape)
        self.output_shape = self.get_output_shape(input_shape)

    def get_output_shape(
        self, input_shape: Tuple[int, ...]
    ) -> Tuple[int, ...]:
        """
        Get the output shape of the module given the input shape.

        Parameters
        ----------
        input_shape : Tuple[int, ...]
            Shape of the input data, e.g. (num_features,).

        Returns
        Tuple[int, ...]
            Shape of the output data, e.g. (num_output_features,).
        """

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

    def get_hyperparameters(self) -> Dict[str, Any]:
        """
        Get the hyperparameters of the module.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the hyperparameters of the module.
        """
        return {
            "subset": self.subset,
            "prepend": self.prepend,
            "axis": self.axis,
            "passthrough": self.passthrough,
            "module": self.module,
        }

    def set_hyperparameters(self, hyperparams: Dict[str, Any]) -> None:
        """
        Set the hyperparameters of the module using the default implementation,
        so just do input validation and set the hyperparameters.

        Parameters
        ----------
        hyperparams : Dict[str, Any]
            Dictionary containing the hyperparameters to set.
        """

        if self.input_shape is not None or self.output_shape is not None:
            raise ValueError(
                "Cannot set hyperparameters after compile. "
                "Please set hyperparameters before calling compile()."
            )

        super(SubsetModule, self).set_hyperparameters(hyperparams)

    def get_params(self) -> Tuple[np.ndarray, ...]:
        """
        Get the current trainable parameters of the module. If the module has
        no trainable parameters, this method should return an empty tuple.

        Returns
        -------
        Tuple[np.ndarray, ...]
            Tuple of numpy arrays representing the module's parameters.
        """
        return self.module.get_params()

    def set_params(self, params: Tuple[np.ndarray, ...]) -> None:
        """
        Set the parameters of the module.

        Parameters
        ----------
        params : Tuple[np.ndarray, ...]
            Tuple of numpy arrays representing the new parameters.
        """
        self.module.set_params(params)

    def get_state(self) -> Tuple[np.ndarray, ...]:
        """
        Get the current state of the module.

        States are used to store "memory" or other information that is not
        passed between modules, is not trainable, but may be updated during
        either training or inference. e.g. batch normalization state.

        The state is optional, in which case this method should return the
        empty tuple.

        Returns
        -------
        Tuple[np.ndarray, ...]
            Tuple of numpy arrays representing the module's state.
        """
        return self.module.get_state()

    def set_state(self, state: Tuple[np.ndarray, ...]) -> None:
        """
        Set the state of the module.

        This method is optional.

        Parameters
        ----------
        state : Tuple[np.ndarray, ...]
            Tuple of numpy arrays representing the new state.
        """
        self.module.set_state(state)

    def serialize(self) -> Dict[str, Any]:
        """
        Serialize the module to a dictionary.

        This method should return a dictionary representation of the module,
        including its parameters and state.

        The default implementation serializes the module's name, trainable
        parameters, and state via a simple dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the serialized module data.
        """

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

    def deserialize(self, data: Dict[str, Any]) -> None:
        """
        Deserialize the module from a dictionary.

        This method should restore the module's parameters and state from the
        given dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary containing the serialized module data.
        """

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
