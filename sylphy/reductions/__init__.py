"""Dimensionality reduction subpackage for sylphy.

Public API:
- Base: `Reductions`, `ReturnType`
- Implementations: `LinearReduction`, `NonLinearReductions`
- Factory helpers: `reduce_dimensionality`, `get_available_methods`,
  `is_linear_method`, `is_nonlinear_method`
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from sylphy.core.optional_dependencies import wrap_optional_dependency_error

__all__ = [
    "LinearReduction",
    "NonLinearReductions",
    "Preprocess",
    "Reductions",
    "ReturnType",
    "get_available_methods",
    "is_linear_method",
    "is_nonlinear_method",
    "reduce_dimensionality",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "Reductions": (".reduction_methods", "Reductions"),
    "ReturnType": (".reduction_methods", "ReturnType"),
    "Preprocess": (".reduction_methods", "Preprocess"),
    "LinearReduction": (".linear_reductions", "LinearReduction"),
    "NonLinearReductions": (".non_linear_reductions", "NonLinearReductions"),
    "reduce_dimensionality": (".factory", "reduce_dimensionality"),
    "get_available_methods": (".factory", "get_available_methods"),
    "is_linear_method": (".factory", "is_linear_method"),
    "is_nonlinear_method": (".factory", "is_nonlinear_method"),
}

_OPTIONAL_DEPENDENCY_EXPORTS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "NonLinearReductions": ("Non-linear reductions", "reductions", ("umap", "umap-learn", "clustpy")),
}


def __getattr__(name: str) -> object:
    spec = _LAZY_EXPORTS.get(name)
    if spec is None:
        msg = f"module '{__name__}' has no attribute '{name}'"
        raise AttributeError(msg)

    mod_name, attr = spec
    try:
        module = import_module(mod_name, package=__name__)
    except (ImportError, ModuleNotFoundError) as exc:
        optional_spec = _OPTIONAL_DEPENDENCY_EXPORTS.get(name)
        if optional_spec is not None:
            feature, extra, packages = optional_spec
            wrapped = wrap_optional_dependency_error(
                exc,
                feature=feature,
                extra=extra,
                packages=packages,
            )
            if wrapped is not None:
                raise wrapped from exc
        raise

    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:  # pragma: no cover
    from .factory import get_available_methods, is_linear_method, is_nonlinear_method, reduce_dimensionality
    from .linear_reductions import LinearReduction
    from .non_linear_reductions import NonLinearReductions
    from .reduction_methods import Preprocess, Reductions, ReturnType
