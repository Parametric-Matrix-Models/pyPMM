__version__ = "0.1.0"
__author__ = "Patrick Cook"

import jax
#jax.config.update("jax_enable_x64", True)

from . import Modules
from .Model import Model
from . import Training
from . import Scalers
