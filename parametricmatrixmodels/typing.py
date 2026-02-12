from __future__ import annotations

from sys import version_info
from typing import Any, TypeAlias

import numpy as onp
from jaxtyping import (
    Array,
    Complex,
    Inexact,
    Num,
    PyTree,
    Real,
    Shaped,
)
from ordered_set import OrderedSet  # noqa: F401

r"""
Module for type aliases used throughout the ParametricMatrixModels package.

Type checking at runtime is performed using ``jaxtyping`` and ``beartype``.
Functions that are JIT-compiled with JAX will only be type-checked at trace
(compile) time, not at runtime, and so do not incur any runtime overhead.
"""

if version_info >= (3, 9):
    List = list
    Dict = dict
    Tuple = tuple
    from beartype.typing import Callable, Type

else:
    from typing import Callable, Dict, List, Tuple, Type  # noqa: F401

r"""
This module also contains future-proofing type aliases to handle the PEP 484 /
PEP 585 disaster. See:
https://beartype.readthedocs.io/en/latest/api_roar/#pep-585-deprecations
"""

#: A PyTree representing model parameters. Each leaf is a numerical JAX array.
#:
#: ``Dict`` Example:
#:
#:     .. code-block:: python
#:
#:         params: Params = {
#:                              "weights": jnp.array([[1.0, 2.0], [3.0, 4.0]]),
#:                              "bias": jnp.array([1.0, 2.0])
#:                          }
#:
#: ``Tuple`` Example:
#:
#:     .. code-block:: python
#:
#:         params: Params = (jnp.array([1.0, 2.0]), jnp.array([0.5]))
#:
#: Arbitrary ``PyTree`` Example:
#:
#:     .. code-block:: python
#:
#:         params: Params = {
#:                              "w": (jnp.array([1.0, 1.0]), jnp.array([0.0])),
#:                              "b": jnp.array([0.5]),
#:                              "a": [jnp.array([[1.0]]), jnp.array([[2.0]])],
#:                          }
Params: TypeAlias = PyTree[Inexact[Array, "..."], "Params"]

#: A special case of ``Params`` where the parameters are represented as a
#: tuple of JAX arrays.
TupleParams: TypeAlias = Tuple[Inexact[Array, "..."], ...]
#: A special case of ``Params`` where the parameters are represented as a
#: list of JAX arrays.
ListParams: TypeAlias = List[Inexact[Array, "..."]]
#: A special case of ``Params`` where the parameters are represented as a
#: dictionary of JAX arrays.
DictParams: TypeAlias = Dict[str, Inexact[Array, "..."]]

#: A dictionary representing hyperparameters for model configuration.
#: The keys are strings representing hyperparameter names, and the values
#: can be of any (ideally serializable) type. If the values are not
#: serializable, then default implementations for saving and loading
#: models and modules to file will need to be overridden in subclasses.
HyperParams: TypeAlias = Dict[str, Any]

#: A special case of ``Data`` where the input data is represented as a single
#: JAX array.
ArrayData: TypeAlias = Inexact[Array, "batch_size ..."]

#: A PyTree or JAX array representing input data. If a PyTree, each leaf is a
#: numerical JAX array with a leading batch dimension. The batch dimension must
#: not change during evaluation of a model or module, but all other dimensions,
#: including their number and the structure of the PyTree, may vary.
#: Alternatively, a single JAX array with a leading batch dimension can be
#: used. A module or model may take as input either a PyTree or a single JAX
#: array, and may return either a PyTree or a single JAX array as output,
#: regardless of the input type.
#:
#: The structure of the PyTree can change throughout evaluation, so it is not
#: specified in the type alias.
Data: TypeAlias = PyTree[Inexact[Array, "batch_size ..."]] | ArrayData

#: A PyTree with guaranteed structure and shape and only numerical arrays
DataFixed: TypeAlias = PyTree[Inexact[Array, "batch_size ?*d"], "DataFixed"]
#: A PyTree with guaranteed structure and shape and only numerical arrays
#: with no batch
BatchlessDataFixed: TypeAlias = PyTree[Inexact[Array, "?*d"], "DataFixed"]


#: A PyTree with guaranteed structure and shape and only Reals
RealDataFixed: TypeAlias = PyTree[
    Real[Array, "batch_size ?*d"], "RealDataFixed"
]
#: A PyTree with guaranteed structure and shape and only Reals with no batch
BatchlessRealDataFixed: TypeAlias = PyTree[Real[Array, "?*d"], "RealDataFixed"]
#: A PyTree with guaranteed structure and shape and only Complex numbers
ComplexDataFixed: TypeAlias = PyTree[
    Complex[Array, "batch_size ?*d"], "ComplexDataFixed"
]
#: A PyTree with guaranteed structure and shape and only Complex numbers with
#: no batch
BatchlessComplexDataFixed: TypeAlias = PyTree[
    Complex[Array, "?*d"], "ComplexDataFixed"
]

#: A special case of ``DataShape`` where the input data shape is represented as
#: a single tuple of integers.
ArrayDataShape: TypeAlias = Tuple[int, ...] | Tuple[None, ...]

#: A PyTree representing the shape of input data. Each leaf is a tuple of
#: integers representing the shape of the corresponding leaf in a ``Data``
#: PyTree, excluding the leading batch dimension. Alternatively, a single tuple
#: of integers can be used to represent the shape of a single JAX array.
DataShape: TypeAlias = (
    PyTree[Tuple[int, ...]] | PyTree[Tuple[None, ...]] | ArrayDataShape
)

#: A PyTree representing the private and persistent state of a module. Each
#: leaf is a numerical JAX array of arbitrary shape. The structure of the
#: PyTree and shape of the arrays must not change during evaluation of a
#: module.
State: TypeAlias = PyTree[Num[Array, "*?d"], "State"]

#: A special case of ``State`` where the state is represented as a tuple of JAX
#: arrays.
TupleState: TypeAlias = Tuple[Num[Array, "*?d"], ...]
#: A special case of ``State`` where the state is represented as a list of JAX
#: arrays.
ListState: TypeAlias = List[Num[Array, "*?d"]]
#: A special case of ``State`` where the state is represented as a dictionary
#: of JAX arrays.
DictState: TypeAlias = Dict[str, Num[Array, "*?d"]]

#: A Callable that represents the forward pass of a module. The Callable must
#: JAX-jittable, pure (i.e., no side effects), and JAX-differentiable. It takes
#: the following arguments:
#: - ``params``: A PyTree of model parameters.
#: - ``data``: A PyTree or JAX array of input data.
#: - ``training``: A boolean flag indicating whether the module is being used
#:   for training or evaluation. Useful for modules that behave differently
#:   during training (e.g., dropout, batch normalization).
#: - ``state``: A PyTree representing the current state of the module.
#: - ``rng``: An optional JAX random key for stochastic operations.
#: The Callable returns a tuple containing:
#: - ``data``: A PyTree or JAX array of output data.
#: - ``state``: A PyTree representing the updated state of the module.
#:
#: If the module does not maintain any state, the ``state`` argument can be
#: passed as an empty tuple ``()`` and the returned state will also be an empty
#: tuple.
ModuleCallable: TypeAlias = Callable[
    [
        Params,
        Data,
        bool,
        State,
        Any,
    ],
    Tuple[Data, State],
]

#: Type that represents a union between all types that are serializable without
#: allow_pickle=True
Serializable: TypeAlias = (
    PyTree[int]
    | PyTree[float]
    | PyTree[bool]
    | PyTree[str]
    | PyTree[complex]
    | PyTree[Shaped[Array, "..."]]
    | PyTree[onp.ndarray, "..."]
)
