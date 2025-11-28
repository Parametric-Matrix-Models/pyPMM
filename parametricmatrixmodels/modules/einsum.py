import jax
import jax.numpy as np
from beartype import beartype
from jaxtyping import Array, Inexact, PyTree, jaxtyped

from ..tree_util import is_shape_leaf
from ..typing import (
    Any,
    Data,
    DataShape,
    Dict,
    HyperParams,
    ModuleCallable,
    OrderedSet,
    Params,
    State,
    Tuple,
)
from .basemodule import BaseModule


class Einsum(BaseModule):
    """
    Module that implements Einsum operations between array leaves in a PyTree
    input. Optionally with leading trainable arrays. The einsum operation is
    defined by the provided einsum string, which must specify all indices
    except for the batch index. The batch index should not be included in the
    provided einsum string, as it will be automatically added as the leading
    index later.

    Examples
    --------
    Taking in a PyTree with two leaves, each being a 1D array (excluding the
    batch dimension), the dot product can be expressed as:

        >>> einsum_module = Einsum("i,i->")

    Taking in a PyTree with three leaves, each of which being a matrix
    (excluding the batch dimension), and computing the matrix multiplication
    of a trainable matrix with the elementwise product of these three input
    matrices can be expressed as:

        >>> einsum_module = Einsum("ij,jk,jk,jk->ik")

    Since the einsum string in this example has four input arrays (separated by
    commas), when the module is compiled with only three input leaves, it will
    infer that the leading array is trainable, and initialize it randomly
    during compilation. This array can be directly specified by calling
    ``set_params`` or by providing it in the ``params`` argument during
    initialization.

    Alternatively, to perform the same operation but with the leading array
    fixed (not trainable), the module must be initialized with this array
    specified in the ``params`` argument and ``trainable`` set to ``False``:

        >>> W = np.random.normal(size=(input_dim, output_dim))
        >>> einsum_module = Einsum(
        ...     "ij,jk,jk,jk->ik",
        ...     params = W,
        ...     trainable = False)

    In this case, the fixed arrays can be specified later by calling
    ``set_hyperparameters`` with a dictionary containing the key ``params``.

    Any additional trainable or fixed arrays will always be treated as leading
    arrays in the einsum operation.

    If no additional fixed or trainable arrays are to be used, the einsum
    string can alternatively be provided as a two-element Tuple
    consisting of a PyTree of strings with the same structure as the input data
    and string representing the output of the einsum, which can be omitted if
    the output string is to be inferred. For example, for a PyTree input with
    structure ``PyTree([*, (*, *)])``, with each leaf being a 1D array, to
    specify the operation of the three-way outer product between the three
    leaves in the order ``PyTree([2, (0, 1)])``, the einsum string can be
    provided in any of the following equivalent ways:

        >>> Einsum('c,a,b->abc') # full string
        >>> Einsum('c,a,b') # output inferred 'abc'
        >>> Einsum(['k', ('i', 'j')]) # output inferred 'ijk'
        >>> Einsum((['k', ('i', 'j')], 'ijk')) # full tuple
        >>> Einsum((['a', ('b', 'c')], 'bca')) # full tuple

    If additional fixed or trainable arrays are to be used, the einsum string
    can be provided as a three-element tuple where the first element is an
    einsum str for the additional arrays, the second element is a PyTree of
    strings with the same structure as the input
    data, and the third element is the output string, which can be omitted if
    to be inferred. For example, to perform the same three-way outer product as
    above but with a leading trainable array, the einsum string can be provided
    in any of the following equivalent ways:

        >>> Einsum('ab,c,a,b->c') # full string
        >>> Einsum('ab,c,a,b') # output inferred 'c'
        >>> Einsum(('ij', ['k', ('i', 'j')])) # output inferred 'k'
        >>> Einsum(('ij', ['k', ('i', 'j')], 'k')) # full tuple

    For multiple leading arrays the following are equivalent:

        >>> Einsum('ab,cd,c,a,b->d') # full string
        >>> Einsum('ab,cd,c,a,b') # output inferred 'd'
        >>> Einsum(('ab,cd', ['c', ('a', 'b')])) # output inferred 'd'
        >>> Einsum(('ab,cd', ['c', ('a', 'b')], 'd')) # full tuple

    """

    def __init__(
        self,
        einsum_str: (
            Tuple[str, PyTree[str], str]
            | Tuple[str, PyTree[str]]
            | Tuple[PyTree[str], str]
            | PyTree[str]
            | str
            | None
        ) = None,
        params: (
            PyTree[Inexact[Array, "..."]] | Inexact[Array, "..."] | None
        ) = None,
        dim_map: Dict[str, int] | None = None,
        trainable: bool = False,
        init_magnitude: float = 1e-2,
        real: bool = True,
    ) -> None:
        """
        Initialize an ``Einsum`` module.

        Parameters
        ----------

            einsum_str
                The einsum string defining the operation. The batch index
                should not be included in the provided einsum string, as it
                will be automatically added as the leading index later. Can be
                provided as a single string, a PyTree of input_strings, a Tuple
                of (PyTree[input_strings], output_string), or a tuple of
                (leading_arrays_einsum_str, PyTree[input_strings],
                output_string). The input_strings in a PyTree must have
                the same structure as the input data. output_string can be
                omitted to have it inferred. If ``None``, it
                must be set before compilation via ``set_hyperparameters`` with
                the ``einsum_str`` key. Default is ``None``.
            params
                Optional additional leading arrays for the einsum operation. If
                trainable is ``True``, these will be treated as the initial
                values for trainable arrays. If ``False``, they will be treated
                as fixed arrays. If ``None`` and trainable is ``True``, the
                leading arrays will be initialized randomly during compilation.
                Default is ``None``. Can be provided later via
                ``set_hyperparameters`` with the ``params`` key if
                ``trainable`` is ``False``, or via ``set_params`` if
                ``trainable`` is ``True``.
            dim_map
                Dictionary mapping einsum indices (characters) to integer
                sizes for the array dimensions. Only entries for indices that
                cannot be inferred from the input data shapes or parameter
                shapes need to be provided. Default is ``None``.
            trainable
                Whether the provided ``params`` are trainable or fixed. If
                ``True``, the arrays in ``params`` will be treated as initial
                values for trainable arrays. If ``False``, they will be treated
                as fixed arrays. Default is ``False``.
            init_magnitude
                Magnitude for the random initialization of weights.
                Default is ``1e-2``.
            real
                Ignored when there are no trainable arrays. If ``True``, the
                weights and biases will be real-valued. If ``False``, they will
                be complex-valued. Default is ``True``.
        """

        self.einsum_str = einsum_str
        self.params = params
        self.dim_map = dim_map
        self.trainable = trainable
        self.init_magnitude = init_magnitude
        self.real = real
        self.input_shape: DataShape | None = None

    def _get_dimension_map(
        self,
        concrete_einsum_str: str,
        input_shape: DataShape,
    ) -> Dict[str, int]:
        r"""
        Fill in the dimension map by inferring sizes from the input shapes
        and parameter shapes based on the provided concrete einsum string and
        ``self.params`` if applicable.

        Parameters
        ----------
            concrete_einsum_str
                The concrete einsum string with all indices specified,
                including the output indices and batch index.
            input_shape
                The shape of the input data, used to infer dimension sizes.
                Should not include the batch dimension.
        Returns
        -------
            A complete dimension map with sizes for all indices in the
            einsum string.
        """

        dim_map = (
            dict({k: v for k, v in self.dim_map.items()})
            if self.dim_map
            else {}
        )

        # get the batch dimension character from the concrete einsum string and
        # remove it
        input_str, output_str = concrete_einsum_str.split("->")
        batch_char = output_str[0]
        input_strs = input_str.replace(batch_char, "").split(",")

        input_shapes_list = jax.tree.leaves(input_shape, is_leaf=is_shape_leaf)

        # split input_strs into leading and input based on number of input
        # arrays
        num_input_arrays = len(input_shapes_list)
        leading_strs = input_strs[:-num_input_arrays]
        input_strs = input_strs[-num_input_arrays:]

        # infer from input arrays
        for s, shape in zip(input_strs, input_shapes_list):
            if len(s) != len(shape):
                raise ValueError(
                    f"Einsum input string '{s}' has length {len(s)}, but "
                    f"corresponding input array has shape {shape}."
                )
            for char, size in zip(s, shape):
                if char not in dim_map:
                    dim_map[char] = size
                else:
                    if dim_map[char] != size:
                        raise ValueError(
                            f"Dimension size mismatch for index '{char}': "
                            f"got size {size} from input shape {shape}, "
                            "but previously recorded size in `dim_map` is "
                            f"{dim_map[char]}."
                        )
        # infer from parameter arrays if they exist
        if self.params is not None:
            param_shapes_list = (
                [self.params.shape]
                if isinstance(self.params, np.ndarray)
                else jax.tree.leaves(
                    self.params, is_leaf=lambda x: isinstance(x, np.ndarray)
                )
            )
            if len(leading_strs) != len(param_shapes_list):
                raise ValueError(
                    f"Number of leading einsum strings ({len(leading_strs)}) "
                    "does not match number of parameter arrays "
                    f"({len(param_shapes_list)})."
                )

            for s, shape in zip(leading_strs, param_shapes_list):
                if len(s) != len(shape):
                    raise ValueError(
                        f"Einsum leading string '{s}' has length {len(s)}, "
                        "but corresponding parameter array has shape "
                        f"{shape}."
                    )
                for char, size in zip(s, shape):
                    if char not in dim_map:
                        dim_map[char] = size
                    else:
                        if dim_map[char] != size:
                            raise ValueError(
                                "Dimension size mismatch for index "
                                f"'{char}': got size {size} from parameter "
                                f"shape {shape}, but previously recorded "
                                f"size in `dim_map` is {dim_map[char]}."
                            )

        # now, all indices in the einsum string should be in dim_map, except
        # for the batch index, including leading arrays and output indices
        all_indices = OrderedSet(
            concrete_einsum_str.replace(",", "")
            .replace("->", "")
            .replace(batch_char, "")
        )
        missing_indices = all_indices - OrderedSet(dim_map.keys())
        if missing_indices:
            raise ValueError(
                f"Could not infer sizes for indices {missing_indices}. "
                "Please provide their sizes in the `dim_map` argument."
            )

        return dim_map

    def _get_concrete_einsum_str(self, input_shape: DataShape) -> str:
        r"""
        Get the concrete einsum string by parsing the provided einsum string,
        adding batch indices and inferring any missing output indices. To
        account for batch dimensions. If the einsum string is a PyTree, it must
        have the same structure as the input data.

        Parameters
        ----------
            input_shape
                The shape of the input data, used to infer any missing output
                indices as well as validate existing indices. Should not
                include the batch dimension.

        Examples
        --------
            >>> m = Einsum("ij,jk->ik")
            >>> m._get_concrete_einsum_str(((2, 3), (3, 4)))
            'aij,ajk->aik' # leading index 'a' added for batch dimension

            >>> m = Einsum((('ij', 'jk'), 'ik'))
            >>> m._get_concrete_einsum_str(((2, 3), (3, 4)))
            'aij,ajk->aik'

            >>> m = Einsum("ab,bc->ac")
            >>> m._get_concrete_einsum_str(((5, 2), (2, 4)))
            'dab,dbc->dac' # leading index 'd' added for batch dimension

            >>> m = Einsum((['ij', 'jk'], 'ik'))
            >>> m._get_concrete_einsum_str(((2, 3), (3, 4)))
            ValueError: The structure of the einsum_str PyTree must match that
            of the input data.
            # (since the input data is a Tuple of two arrays, not a List)

            >>> m = Einsum(('ij,jk', {'x': 'ik', 'y': 'ab'}, 'ab'))
            >>> m._get_concrete_einsum_str({'x': (2, 3), 'y': (3, 4)})
            'ij,jk,cik,cab->cab' # leading arrays don't have batch index
            # all arrays from PyTrees are inserted in the same order as the
            # list from jax.tree.leaves(...)
        """

        # standardize einsum_str to the single string case
        if self.einsum_str is None:
            raise ValueError("einsum_str must be set before concretization.")

        was_pytree = False

        # if the einsum_str structure matches the input_shape structure, it's
        # the PyTree case
        input_struct = jax.tree.structure(input_shape, is_leaf=is_shape_leaf)
        einsum_str_struct = jax.tree.structure(self.einsum_str)
        if einsum_str_struct == input_struct:
            leading_str = None
            input_strs = self.einsum_str
            output_str = None
            was_pytree = True

        elif isinstance(self.einsum_str, tuple):
            if len(self.einsum_str) == 3:
                leading_str, input_strs, output_str = self.einsum_str
            elif len(self.einsum_str) == 2:
                first, second = self.einsum_str

                # two cases: (leading_str, input_strs) or
                # (input_strs, output_str)

                # can distinguish based on which is a bare string
                # if both are bare strings, then it's ambiguous (input is a
                # degenerate PyTree of a single array and the user should use
                # the single string case instead)
                first_is_str = isinstance(first, str)
                second_is_str = isinstance(second, str)
                if first_is_str and not second_is_str:
                    leading_str = first
                    input_strs = second
                    output_str = None
                elif not first_is_str and second_is_str:
                    leading_str = None
                    input_strs = first
                    output_str = second
                else:
                    # this case should be caught by the structure check above,
                    # but just in case
                    raise ValueError(
                        "If einsum_str is a tuple of length 2, one element "
                        "must be a bare string and the other must be a "
                        "non-degenerate PyTree of strings matching the "
                        "structure of input_shape. If input_shape is a "
                        "degenerate PyTree representing bare array input, use "
                        "the single string einsum_str format instead. E.g. "
                        f"use Einsum('{first},{second}') or "
                        f"Einsum('{first}->{second}') depending on your "
                        "intention instead of "
                        f"Einsum(('{first}', '{second}'))."
                    )

            else:
                raise ValueError(
                    "If einsum_str is a tuple not matching the structure of "
                    "input_shape, it must have length 2 or 3."
                )
        elif isinstance(self.einsum_str, str):
            leading_str = None
            input_strs = self.einsum_str
            output_str = None
        else:  # PyTree case
            # verify the structure matches input_shape
            einsum_struct = jax.tree.structure(self.einsum_str)
            input_struct = jax.tree.structure(
                input_shape, is_leaf=is_shape_leaf
            )
            if einsum_struct != input_struct:
                raise ValueError(
                    "The structure of the einsum_str PyTree must match that "
                    f"of the input data. Got {einsum_struct} but "
                    f"expected {input_struct}."
                )

            leading_str = None
            input_strs = self.einsum_str
            output_str = None

        # if output_str is None, see if it is included in input_strs, which is
        # only possible if input_strs is a bare string containing '->'
        if (
            output_str is None
            and isinstance(input_strs, str)
            and "->" in input_strs
        ):
            input_strs, output_str = input_strs.split("->")

        # if leading_str is None, see if it is included in input_strs, which is
        # only possible if input_strs is a bare string
        if leading_str is None and isinstance(input_strs, str):
            input_str_list = input_strs.split(",")
            # if the number of input strings is greater than the number of
            # input arrays, then the leading strings are included here
            num_input_arrays = len(
                jax.tree.leaves(input_shape, is_leaf=is_shape_leaf)
            )
            num_input_strings = len(input_str_list)
            if num_input_strings > num_input_arrays:
                leading_str = ",".join(
                    input_str_list[: num_input_strings - num_input_arrays]
                )
                input_strs = ",".join(
                    input_str_list[num_input_strings - num_input_arrays :]
                )

        # now leading_str, input_strs, and output_str are properly separated

        # validate the number of arrays in the input_strs matches input_shape
        num_input_arrays = len(
            jax.tree.leaves(input_shape, is_leaf=is_shape_leaf)
        )
        if isinstance(input_strs, str):
            num_input_strings = len(input_strs.split(","))
        else:
            was_pytree = True
            # no input strings can have non-alphabetic characters, we check
            # that here
            if not jax.tree.all(
                jax.tree.map(
                    lambda s: isinstance(s, str) and s.isalpha(),
                    input_strs,
                )
            ):
                raise ValueError(
                    "All input strings in the einsum_str PyTree must be "
                    "bare strings containing only alphabetic characters."
                )

            num_input_strings = len(
                jax.tree.leaves(
                    input_strs, is_leaf=lambda x: isinstance(x, str)
                )
            )

        # if leading_str is None then params must be None or empty
        # if it's not None, params may still be None (to be initialized later)
        if leading_str is None and not (
            self.params is None or self.params == ()
        ):
            raise ValueError(
                "If einsum_str does not specify leading arrays, then "
                "params must be None or empty."
            )

        # if output_str is not None and contains either ',' or '->', raise
        # error
        if output_str is not None:
            if ("," in output_str) or ("->" in output_str):
                raise ValueError(
                    "output_str cannot contain ',' or '->'. Got "
                    f"'{output_str}'."
                )

        # verify that all strings contain only valid characters (a-z, A-Z, ',')
        valid_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,"

        def validate_chars(s: str) -> None:
            invalid_chars = set(s) - set(valid_chars)
            if invalid_chars:
                raise ValueError(
                    f"Einsum string '{s}' contains invalid characters: "
                    f"{invalid_chars}. Only a-z, A-Z, and ',' are allowed. "
                )

        if leading_str is not None:
            validate_chars(leading_str)
        if output_str is not None:
            validate_chars(output_str)
        if isinstance(input_strs, str):
            validate_chars(input_strs)
        else:
            jax.tree.map(validate_chars, input_strs)

        # if input_strs is a bare string, split on commas
        if isinstance(input_strs, str):
            input_strs = input_strs.split(",")

        # now, leading_str and output_str are either strings or None
        # and input_strs is a PyTree of strings

        # put the input strings in the order of the input_shape
        # so long as the original input was a PyTree
        if was_pytree:
            input_strs = jax.tree.map(
                lambda _, s: s,
                input_shape,
                input_strs,
                is_leaf=is_shape_leaf,
            )

        # reduce over input_strs to single string
        def reduce_input_strs(carry: str, s: str) -> str:
            if carry == "":
                return s
            else:
                return carry + "," + s

        input_str = jax.tree.reduce(
            reduce_input_strs,
            input_strs,
            initializer="",
        )

        # infer output_str if None
        # it should be all the indices that appear only once in
        # leading_str + input_str, and be in alphabetical order, same as einsum
        # would do
        if output_str is None:
            all_input = ""
            if leading_str is not None:
                all_input += leading_str
            all_input += input_str

            index_counts = {}
            for char in all_input:
                if char != ",":
                    if char in index_counts:
                        index_counts[char] += 1
                    else:
                        index_counts[char] = 1

            output_indices = [
                char
                for char in sorted(index_counts.keys())
                if index_counts[char] == 1
            ]
            output_str = "".join(output_indices)

        # find all used indices to find a batch index that doesn't conflict
        # use OrderedSet to have a deterministic order
        used_indices = OrderedSet()
        if leading_str is not None:
            used_indices.update(OrderedSet(leading_str.replace(",", "")))
        used_indices.update(OrderedSet(input_str.replace(",", "")))
        if output_str is not None:
            used_indices.update(OrderedSet(output_str))

        available_indices = (
            OrderedSet(valid_chars.replace(",", "")) - used_indices
        )
        if not available_indices:
            raise ValueError(
                "No available indices to use for batch dimension. "
                "Einsum strings are using all possible indices."
            )
        batch_index = available_indices[0]

        # before adding batch index, validate that the input
        # shapes match the input_strs
        input_str_list = input_str.split(",")

        def validate_shape(s: str, shape: Tuple[int, ...]) -> None:
            if len(s) != len(shape):
                raise ValueError(
                    f"Einsum input string '{s}' has length {len(s)}, but "
                    f"corresponding input array has shape {shape}."
                )

        for s, shape in zip(
            input_str_list,
            jax.tree.leaves(input_shape, is_leaf=is_shape_leaf),
        ):
            validate_shape(s, shape)

        # add batch index to input_str and output_str only
        # leading_str does not get a batch index
        input_str = ",".join([batch_index + s for s in input_str.split(",")])
        output_str = batch_index + output_str

        # construct full einsum string
        full_einsum_str = input_str + "->" + output_str
        if leading_str is not None:
            full_einsum_str = leading_str + "," + full_einsum_str
        return full_einsum_str

    @property
    def name(self) -> str:
        concrete_einsum_str = (
            self._get_concrete_einsum_str(self.input_shape)
            if self.input_shape is not None
            else self.einsum_str
        )
        return f"Einsum({concrete_einsum_str})"

    def is_ready(self) -> bool:
        return self.input_shape is not None

    def _get_callable(self) -> ModuleCallable:

        # set up the callable

        concrete_einsum_str = self._get_concrete_einsum_str(self.input_shape)

        @jaxtyped(typechecker=beartype)
        def einsum_callable(
            params: Params, data: Data, training: bool, state: State, rng: Any
        ) -> Tuple[Data, State]:
            # prepare the list of arrays to einsum over
            # most of this will be traced out by jax
            arrays = []

            # if trainable, params will be the leading arrays, if there are
            # any, otherwise it will be an empty tuple
            if self.trainable:
                if isinstance(params, np.ndarray):
                    arrays.append(params)
                else:
                    arrays.extend(
                        jax.tree.leaves(
                            params,
                            is_leaf=lambda x: isinstance(x, np.ndarray),
                        )
                    )
            elif not self.trainable and self.params is not None:
                # if not trainable, params are fixed leading arrays, if any,
                # and they are stored in self.params
                if isinstance(self.params, np.ndarray):
                    arrays.append(self.params)
                else:
                    arrays.extend(
                        jax.tree.leaves(
                            self.params,
                            is_leaf=lambda x: isinstance(x, np.ndarray),
                        )
                    )
            # add the input data arrays
            arrays.extend(
                jax.tree.leaves(
                    data,
                    is_leaf=lambda x: isinstance(x, np.ndarray),
                )
            )
            output = np.einsum(concrete_einsum_str, *arrays)

            return output, state

        return einsum_callable

    def compile(self, rng: Any, input_shape: DataShape) -> None:
        if self.einsum_str is None:
            raise ValueError(
                "einsum_str must be set before compiling the module"
            )
        if self.is_ready() and self.input_shape != input_shape:
            raise ValueError(
                "Module has already been compiled with a different input "
                "shape."
            )

        self.input_shape = input_shape

        concrete_einsum_str = self._get_concrete_einsum_str(input_shape)
        dim_map = self._get_dimension_map(
            concrete_einsum_str,
            input_shape,
        )

        # figure out how many leading arrays there are
        input_str, output_str = concrete_einsum_str.split("->")
        input_strs = input_str.split(",")
        num_input_arrays = len(
            jax.tree.leaves(input_shape, is_leaf=is_shape_leaf)
        )
        leading_strs = input_strs[:-num_input_arrays]

        # initialize params if needed
        if len(leading_strs) > 0 and self.params is None:
            if not self.trainable:
                raise ValueError(
                    "params must be provided for fixed (non-trainable) "
                    "leading arrays."
                )
            param_arrays = []
            if self.real:
                keys = jax.random.split(rng, len(leading_strs))
            else:
                rkey, ikey = jax.random.split(rng)
                rkeys = jax.random.split(rkey, len(leading_strs))
                ikeys = jax.random.split(ikey, len(leading_strs))
            for i, s in enumerate(leading_strs):
                shape = tuple(dim_map[char] for char in s)
                if self.real:
                    param_array = self.init_magnitude * jax.random.normal(
                        keys[i], shape
                    )
                else:
                    real_part = self.init_magnitude * jax.random.normal(
                        rkeys[i], shape
                    )
                    imag_part = self.init_magnitude * jax.random.normal(
                        ikeys[i], shape
                    )
                    param_array = real_part + 1j * imag_part

                param_arrays.append(param_array)

            # if there's only one leading array, store it as a single array
            if len(param_arrays) == 1:
                self.params = param_arrays[0]
            else:
                self.params = tuple(param_arrays)

        if self.params is not None:
            # make sure params shape matches leading_strs
            expected_shapes = [
                tuple(dim_map[char] for char in s) for s in leading_strs
            ]
            param_shapes = (
                [self.params.shape]
                if isinstance(self.params, np.ndarray)
                else [
                    p.shape
                    for p in jax.tree.leaves(
                        self.params,
                        is_leaf=lambda x: isinstance(x, np.ndarray),
                    )
                ]
            )
            if len(expected_shapes) != len(param_shapes):
                raise ValueError(
                    "Number of leading arrays in einsum_str "
                    f"'{concrete_einsum_str}' "
                    f"({len(leading_strs)}) does not match number of "
                    f"parameter arrays ({len(param_shapes)})."
                )
            for i, (expected_shape, param_shape) in enumerate(
                zip(expected_shapes, param_shapes)
            ):
                if expected_shape != param_shape:
                    raise ValueError(
                        f"Parameter array {i} has shape {param_shape}, but "
                        f"expected shape {expected_shape} based on einsum_str."
                    )

    def get_output_shape(self, input_shape: DataShape) -> DataShape:
        concrete_einsum_str = self._get_concrete_einsum_str(input_shape)
        dim_map = self._get_dimension_map(
            concrete_einsum_str,
            input_shape,
        )
        _, output_str = concrete_einsum_str.split("->")
        # skip the batch dimension
        output_shape = tuple(dim_map[char] for char in output_str[1:])
        return output_shape

    def get_hyperparameters(self) -> HyperParams:

        # include params in hyperparameters only if they are fixed
        return {
            "einsum_str": self.einsum_str,
            "dim_map": self.dim_map,
            **({"params": self.params} if not self.trainable else {}),
            "trainable": self.trainable,
            "init_magnitude": self.init_magnitude,
            "real": self.real,
        }

    def set_hyperparameters(self, hyperparams: HyperParams) -> None:
        # setting the hyperparameters should require recompilation
        self.input_shape = None
        # only allow setting params if they are fixed
        if "einsum_str" in hyperparams:
            self.einsum_str = hyperparams["einsum_str"]
        if "dim_map" in hyperparams:
            self.dim_map = hyperparams["dim_map"]
        if "trainable" in hyperparams:
            self.trainable = hyperparams["trainable"]
        if "params" in hyperparams and not self.trainable:
            self.params = hyperparams["params"]
        if "init_magnitude" in hyperparams:
            self.init_magnitude = hyperparams["init_magnitude"]
        if "real" in hyperparams:
            self.real = hyperparams["real"]

    def get_params(self) -> Params:
        # return params only if they are trainable
        if not self.trainable:
            return ()
        return self.params

    def set_params(self, params: Params) -> None:
        # only allow setting params if they are trainable
        if self.trainable:
            self.params = params
