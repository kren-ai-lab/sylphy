"""
Dimensionality reduction subpackage for sylphy.

Public API:
- Base: `Reductions`, `ReturnType`
- Implementations: `LinearReduction`, `NonLinearReductions`
- Factory helpers: `reduce_dimensionality`, `get_available_methods`,
  `is_linear_method`, `is_nonlinear_method`
"""

from .factory import (
    get_available_methods,
    is_linear_method,
    is_nonlinear_method,
    reduce_dimensionality,
)
from .linear_reductions import LinearReduction
from .non_linear_reductions import NonLinearReductions
from .reduction_methods import Preprocess, Reductions, ReturnType

__all__ = [
    "Reductions",
    "ReturnType",
    "Preprocess",
    "LinearReduction",
    "NonLinearReductions",
    "reduce_dimensionality",
    "get_available_methods",
    "is_linear_method",
    "is_nonlinear_method",
]
