def _get_version():
    try:
        from importlib.metadata import version
    except ImportError:
        from importlib_metadata import version  # type: ignore[no-redef]
    return version("parametric-matrix-models")


__version__ = _get_version()

from . import modules, scalers, training, typing
from .conformalizedmodel import ConformalizedModel
from .model import Model
