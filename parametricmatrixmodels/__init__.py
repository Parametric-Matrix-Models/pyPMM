def _get_version():
    try:
        from importlib.metadata import version
    except ImportError:
        from importlib_metadata import version  # type: ignore[no-redef]
    return version("parametric-matrix-models")


__version__ = _get_version()

from . import (
    eigen_util,
    graph_util,
    modules,
    scalers,
    training,
    tree_util,
    typing,
)
from .conformalizedmodel import ConformalizedModel
from .model import Model
from .nonsequentialmodel import NonSequentialModel
from .sequentialmodel import SequentialModel

from . import model_util  # isort: skip
