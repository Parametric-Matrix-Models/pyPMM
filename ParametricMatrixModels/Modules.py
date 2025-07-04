"""
Re-exporting modules for easier access so that modules can be imported like:

    PMM.Modules.<module_name>
"""

from . import (
    BaseModule,
    AffineEigenvaluePMM,
    AffineObservablePMM,
    SubsetModule,
)
from .BaseModule import BaseModule
from .AffineEigenvaluePMM import AffineEigenvaluePMM
from .AffineObservablePMM import AffineObservablePMM
from .SubsetModule import SubsetModule

# aliases for shortcuts
AEPMM = AffineEigenvaluePMM
AOPMM = AffineObservablePMM

__all__ = [
    "BaseModule",
    "AffineEigenvaluePMM",
    "AffineObservablePMM",
    "SubsetModule",
    "AEPMM",
    "AOPMM",
]
