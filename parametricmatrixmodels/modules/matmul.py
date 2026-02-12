from __future__ import annotations

import jax
from jaxtyping import Array, Inexact

from ..tree_util import (
    extend_structure_from_strpaths,
    getitem_by_strpath,
    is_shape_leaf,
)
from ..typing import (
    Any,
    DataShape,
    Dict,
    HyperParams,
    List,
    Tuple,
)
from .einsum import Einsum


class MatMul(Einsum):
    r"""
    Module for optionally trainable matrix multiplication.
    Just a special case of Einsum. Computes the matrix multiplication of all
    leaves in an input PyTree with an optional trainable or fixed matrix.

    I.e. for a PyTree input with matrix leaves ``(A, B, C)``, and an optional
    trainable or fixed parameter matrix ``M``, the output will be the single
    array ``M @ A @ B @ C``

    The order of the multiplication can be changed by providing the paths of
    each matrix in the parameter ``path_order``. Paths are period-separated
    strings of keys/indices to reach each matrix in the input PyTree. A double
    period ``..`` indicates the trainable/fixed parameter matrix if applicable.

    The final input array may be a vector instead of a matrix.

    All operations are applied over the batch dimension.

    Examples
    --------
    To create a module that takes a single vector input (ignoring batch dim)
    and multiplies it by a trainable weight matrix:

    >>> m = MatMul(output_shape=2, trainable=True)
    >>> m(np.ones((batch_dim, 4))) # vec of size 4 in, vec of size 2 out

    To create a module that multiplies two input matrices together in an order
    different from the default:

    >>> input_data = {
    ...     'x': np.ones((batch_dim, 3, 4)),
    ...     'y': [np.ones((batch_dim, 5, 3)),],
    ... }
    >>> m = MatMul(path_order=['y.0', 'x'])

    """

    __version__: str = "0.0.0"

    def __init__(
        self,
        params: Inexact[Array, "..."] | None = None,
        output_shape: (
            Tuple[int]
            | Tuple[None, int]
            | Tuple[int, int]
            | Tuple[int, None]
            | int
            | None
        ) = None,
        path_order: List[str] | None = None,
        trainable: bool = False,
        init_magnitude: float = 1e-2,
        real: bool = True,
        separator: str = ".",
    ) -> None:
        r"""
        Initialize the MatMul module.

        Parameters
        ----------
        params
            The parameter matrix to use for multiplication. If None and
            ``trainable`` is ``False`` (the default), then no parameter matrix
            is used. If None and ``trainable`` is ``True``, then a randomly
            initialized trainable matrix is created during compilation.
        output_shape
            The shape of the output matrix/vector (excluding batch dimension).
            If an integer is provided, it is treated as the size of a vector
            output. If None (the default), the output size is inferred during
            compilation based on the input shapes and the parameter matrix
            shape if applicable. Can be a tuple with None in one position to
            indicate that the size in that dimension should be inferred.
        path_order
            A list of period-separated strings indicating the order of the
            PyTree paths to multiply. A double separator ``..`` indicates the
            position of the parameter matrix if applicable. If None (the
            default), the order of the matrices is the parameter matrix (if
            applicable) followed by the input PyTree leaves in the order
            returned by ``jax.tree.leaves``. See ``jax.tree_util.keystr`` for
            more details on path strings.
        trainable
            Whether the `params` is trainable.
        init_magnitude
            The magnitude of the random initialization for the trainable
            matrix if applicable.
        real
            Whether to use real or complex parameters for the trainable matrix
            if applicable.
        separator
            The separator to use for path strings. Default is period ('.').
        """
        self.params = params
        self.output_shape = output_shape
        self.path_order = path_order
        self.trainable = trainable
        self.init_magnitude = init_magnitude
        self.real = real
        self.separator = separator

        # pass relevant info to Einsum init
        super().__init__(
            einsum_str=None,  # will be set right before use
            params=params,
            dim_map=None,  # will be set right before use
            trainable=trainable,
            init_magnitude=init_magnitude,
            real=real,
        )

    @property
    def name(self) -> str:

        type_str = (
            "trainable"
            if self.trainable
            else "fixed" if self.params is not None else "inputs"
        )

        return f"MatMul({type_str})"

    def get_output_shape(self, input_shape: DataShape) -> DataShape:
        self.einsum_str, self.dim_map = self._get_einsum_str_and_dim_map(
            input_shape
        )

        return super().get_output_shape(input_shape)

    def compile(self, rng: Any, input_shape: DataShape) -> None:
        self.einsum_str, self.dim_map = self._get_einsum_str_and_dim_map(
            input_shape
        )
        super().compile(rng, input_shape)

    # both compile and get_output_shape need to set einsum_str and dim_map
    # before calling super().compile or super().get_output_shape
    def _get_einsum_str_and_dim_map(
        self, input_shape: DataShape
    ) -> Tuple[str, Dict[str, int]]:
        # validate that input_shape is either a single matrix/vector or a
        # PyTree of all matrices and up to one vector
        ndims = jax.tree.leaves(
            jax.tree.map(lambda x: len(x), input_shape, is_leaf=is_shape_leaf)
        )
        if not all(ndim in (1, 2) for ndim in ndims):
            raise ValueError(
                "All leaves in input_shape must be matrices or vectors."
            )
        if sum(1 for ndim in ndims if ndim == 1) > 1:
            raise ValueError(
                "At most one leaf in input_shape can be a vector."
            )

        double_sep = f"{self.separator}{self.separator}"

        # get the list of input shapes in the order specified by path_order
        if self.path_order is None:
            shape_list = jax.tree.leaves(
                input_shape,
                is_leaf=is_shape_leaf,
            )
            if self.trainable or self.params is not None:
                shape_list = [None] + shape_list
        else:
            # ensure there are no duplicates in path_order
            if len(set(self.path_order)) != len(self.path_order):
                raise ValueError("Duplicate paths found in path_order.")

            shape_list = [
                (
                    getitem_by_strpath(
                        input_shape,
                        path,
                        is_leaf=is_shape_leaf,
                        separator=self.separator,
                    )
                    if path != double_sep
                    else None
                )
                for path in self.path_order
            ]

            # ensure that all leaves in input_shape are used exactly once
            if len(jax.tree.leaves(input_shape, is_leaf=is_shape_leaf)) != len(
                shape_list
            ) - shape_list.count(None):
                raise ValueError(
                    "path_order must include all leaves in input_shape"
                    " exactly once."
                )

        # figure out the dimension of the parameter matrix if applicable
        if self.trainable and self.params is None:
            # find the position of the parameter matrix in the order
            if None not in shape_list:
                raise ValueError(
                    "path_order must contain a double separator "
                    f"('{double_sep}') to indicate the position "
                    "of the trainable parameter matrix."
                )
            param_idx = shape_list.index(None)

            if param_idx == 0:
                # if its the very first operand, then its second dimension is
                # the first dimension of the first input operand, and its first
                # dimension is the first dimension of self.output_shape
                if self.output_shape is None:
                    raise ValueError(
                        "At least the first dimension of output_shape must be"
                        " specified if the parameter matrix is the first"
                        " operand."
                    )
                param_shape = (
                    (
                        self.output_shape[0]
                        if isinstance(self.output_shape, tuple)
                        else self.output_shape
                    ),
                    shape_list[1][0],
                )
            elif param_idx == len(shape_list) - 1:
                # if its the very last operand, then its first dimension is
                # the second dimension of the last input operand, and its
                # second dimension is either the second dimension of the output
                # shape, or it is a vector if the output shape is an int, in
                # which case this int must match the first dimension of the
                # first input operand
                if self.output_shape is None:
                    # vector output
                    param_shape = (shape_list[~1][1],)
                elif isinstance(self.output_shape, int):
                    if self.output_shape != shape_list[0][0]:
                        raise ValueError(
                            "If output_shape is an int and the parameter"
                            " matrix is the last operand, then the int must"
                            " match the first dimension of the first input"
                            " operand. In this case, the output_shape may be"
                            " None to infer this automatically."
                        )
                    param_shape = (shape_list[~1][1],)
                else:
                    if self.output_shape[0] != shape_list[0][0]:
                        raise ValueError(
                            "If output_shape is a tuple and the parameter"
                            " matrix is the last operand, then the first"
                            " dimension of output_shape must match the first"
                            " dimension of the first input operand. In this"
                            " case, the first dimension of output_shape may be"
                            " None to infer this automatically."
                        )
                    param_shape = (shape_list[~1][1], self.output_shape[1])
            else:
                # in the middle somewhere
                param_shape = (
                    shape_list[param_idx - 1][1],
                    shape_list[param_idx + 1][0],
                )
        elif self.params is not None:
            param_idx = shape_list.index(None)
            param_shape = self.params.shape

        else:
            param_idx = None
            param_shape = None
            # check if None is in shape_list, raise an error if so since we're
            # supposed to have a fixed parameter matrix
            if None in shape_list:
                raise ValueError(
                    f"path_order contains a double separator ('{double_sep}')"
                    " indicating a parameter matrix,"
                    " but no trainable or fixed parameter matrix is provided."
                )

        if param_shape is not None:
            # place the shape into the shape_list
            shape_list[param_idx] = param_shape

        # ensure that any 1-dim shapes are at the very end
        vec_idxs = [i for i, shape in enumerate(shape_list) if len(shape) == 1]
        if vec_idxs and (
            len(vec_idxs) > 1 or vec_idxs[0] != len(shape_list) - 1
        ):
            raise ValueError(
                "At most one vector input is allowed, and it must be the"
                " last operand in the multiplication."
            )

        # check the output shape
        inferred_output_shape = [shape_list[0][0]]
        if len(shape_list[~0]) == 2:
            inferred_output_shape.append(shape_list[~0][1])
        if self.output_shape is not None:
            if isinstance(self.output_shape, int):
                if self.output_shape != inferred_output_shape[0]:
                    raise ValueError(
                        "output_shape does not match the inferred output"
                        f" shape. expected {self.output_shape}, inferred"
                        f" {inferred_output_shape}"
                    )
            else:
                if len(self.output_shape) != len(inferred_output_shape):
                    raise ValueError(
                        "output_shape does not match the inferred output"
                        f" shape. expected {self.output_shape}, inferred"
                        f" {inferred_output_shape}"
                    )
                for i in range(len(self.output_shape)):
                    if (
                        self.output_shape[i] is not None
                        and self.output_shape[i] != inferred_output_shape[i]
                    ):
                        raise ValueError(
                            "output_shape does not match the inferred"
                            f" output shape. expected {self.output_shape},"
                            f" inferred {inferred_output_shape}"
                        )

        # now we have all shapes in the desired order, we can build the einsum
        # str, then permute it back to match the leaves order, and build the
        # dim_map for the only necessary dims (just the param matrix if
        # applicable)
        index_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        einsum_terms = []
        dim_map: Dict[str, int] = {}
        for i, shape in enumerate(shape_list):
            if len(shape) == 2:
                einsum_terms.append(f"{index_chars[i]}{index_chars[i+1]}")
                if i == param_idx:
                    dim_map[index_chars[i]] = shape[0]
                    dim_map[index_chars[i + 1]] = shape[1]
            else:
                einsum_terms.append(f"{index_chars[i]}")
                if i == param_idx:
                    dim_map[index_chars[i]] = shape[0]

        # get the reverse permutation to go from path_order to leaves order
        # the easiest way to do this is to place the new indices in the
        # original tree
        if self.path_order is None:
            perm = list(range(len(einsum_terms)))
        else:
            # remove the param matrix path from path_order
            pths_origs = [
                (p, i)
                for i, p in enumerate(self.path_order)
                if p != double_sep
            ]
            # the index for the param matrix is at param_idx (if applicable)
            pths = [p for p, _ in pths_origs]
            idxs = [i for _, i in pths_origs]
            temp_tree = extend_structure_from_strpaths(
                None, pths, separator=self.separator, fill_values=idxs
            )
            # now flatten the tree to get the permutation
            perm = jax.tree.leaves(temp_tree)

            # if param_idx is not None, we just prepend it to the perm
            if param_idx is not None:
                perm = [param_idx] + perm

        # perm represents the inverse permutation from leaves order to
        # path_order, so we need to invert it
        einsum_terms_perm = [einsum_terms[i] for i in perm]

        # we let Einsum infer the rest of dim_map and the output indices
        einsum_str = ",".join(einsum_terms_perm)

        return einsum_str, dim_map

    def get_hyperparameters(self) -> HyperParams:
        # only put parameter matrix in hyperparameters if not trainable
        matmul_hypers = {
            **({"params": self.params} if not self.trainable else {}),
            "output_shape": self.output_shape,
            "path_order": self.path_order,
            "trainable": self.trainable,
            "init_magnitude": self.init_magnitude,
            "real": self.real,
            "separator": self.separator,
        }
        return {**super().get_hyperparameters(), **matmul_hypers}

    def set_hyperparameters(self, hyperparams: HyperParams) -> None:
        super().set_hyperparameters(hyperparams)
        self.trainable = hyperparams.get("trainable", self.trainable)
        if not self.trainable:
            self.params = hyperparams.get("params", self.params)
        self.output_shape = hyperparams.get("output_shape", self.output_shape)
        self.path_order = hyperparams.get("path_order", self.path_order)
        self.init_magnitude = hyperparams.get(
            "init_magnitude", self.init_magnitude
        )
        self.real = hyperparams.get("real", self.real)
        self.separator = hyperparams.get("separator", self.separator)

        # re-init
        self.__init__(
            params=self.params,
            output_shape=self.output_shape,
            path_order=self.path_order,
            trainable=self.trainable,
            init_magnitude=self.init_magnitude,
            real=self.real,
        )

        super().set_hyperparameters(hyperparams)
