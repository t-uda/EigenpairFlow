"""
EigenpairFlow: Continuous and stable tracking of eigendecomposition for parameter-dependent matrices.
"""

from .tracking import eigenpairtrack
from .types import EigenTrackingResults

__all__ = [
    "eigenpairtrack",
    "EigenTrackingResults",
]

__version__ = "0.2.0"
