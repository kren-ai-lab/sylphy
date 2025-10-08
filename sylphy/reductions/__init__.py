"""
Dimensionality reduction subpackage for sylphy.

Public API:
- Base: `Reductions`, `ReturnType`
- Implementations: `LinearReduction`, `NonLinearReductions`
- Factory helpers: `reduce_dimensionality`, `get_available_methods`,
  `is_linear_method`, `is_nonlinear_method`
"""

from .reduction_methods import Reductions, ReturnType, Preprocess
from .linear_reductions import LinearReduction
from .non_linear_reductions import NonLinearReductions
from .factory import (
    reduce_dimensionality,
    get_available_methods,
    is_linear_method,
    is_nonlinear_method,
)

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
