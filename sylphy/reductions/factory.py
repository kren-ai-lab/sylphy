"""Dispatch dimensionality-reduction methods through a unified factory."""

from __future__ import annotations

import logging
from importlib import import_module
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl

from sylphy.core.model_registry import normalize_name
from sylphy.core.optional_dependencies import wrap_optional_dependency_error
from sylphy.logging import add_context, get_child_logger

if TYPE_CHECKING:
    from .reduction_methods import Preprocess, ReturnType

DatasetLike = np.ndarray | pl.DataFrame
Kind = Literal["linear", "nonlinear"]
_LINEAR_KIND: Kind = "linear"
_NONLINEAR_KIND: Kind = "nonlinear"

# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------
_LINEAR_METHODS: dict[str, str] = {
    "pca": "apply_pca",
    "incremental_pca": "apply_incremental_pca",
    "sparse_pca": "apply_sparse_pca",
    "minibatch_sparse_pca": "apply_minibatch_sparse_pca",
    "fast_ica": "apply_fast_ica",
    "truncated_svd": "apply_truncated_svd",
    "factor_analysis": "apply_factor_analysis",
    "nmf": "apply_nmf",
    "minibatch_nmf": "apply_minibatch_nmf",
    "lda": "apply_latent_dirichlet_allocation",
    "latent_dirichlet_allocation": "apply_latent_dirichlet_allocation",
}

_NONLINEAR_METHODS: dict[str, str] = {
    "tsne": "apply_tsne",
    "isomap": "apply_isomap",
    "mds": "apply_mds",
    "lle": "apply_lle",
    "spectral": "apply_spectral",
    "umap": "apply_umap",
    "dictionary_learning": "apply_dictionary_learning",
    "minibatch_dictionary_learning": "apply_mini_batch_dictionary_learning",
    "dipext": "apply_dip_ext",
}


def _build_methods() -> dict[str, tuple[Kind, str]]:
    """Build the method registry mapping names to implementation attributes."""
    methods: dict[str, tuple[Kind, str]] = {}
    for key, attr in _LINEAR_METHODS.items():
        methods[key] = (_LINEAR_KIND, attr)
    for key, attr in _NONLINEAR_METHODS.items():
        methods[key] = (_NONLINEAR_KIND, attr)
    return methods


_METHODS: dict[str, tuple[Kind, str]] = _build_methods()

logger = get_child_logger("reductions.factory", component="reductions", facility="factory")


def get_available_methods(kind: Kind | None = None) -> list[str]:
    """List available method names.

    Args:
        kind: Optional family filter (``linear`` or ``nonlinear``).

    Returns:
        Method names accepted by ``reduce_dimensionality``.

    """
    if kind is None:
        return sorted(_METHODS.keys())
    if kind == "linear":
        return sorted(_LINEAR_METHODS.keys())
    if kind == "nonlinear":
        return sorted(_NONLINEAR_METHODS.keys())
    msg = "kind must be one of: None, 'linear', 'nonlinear'."
    raise ValueError(msg)


def is_linear_method(name: str) -> bool:
    """Return True if `name` is a registered linear method."""
    return normalize_name(name) in _LINEAR_METHODS


def is_nonlinear_method(name: str) -> bool:
    """Return True if `name` is a registered non-linear method."""
    return normalize_name(name) in _NONLINEAR_METHODS


def reduce_dimensionality(
    method: str,
    dataset: DatasetLike,
    *,
    return_type: ReturnType = "numpy",
    preprocess: Preprocess = "none",
    random_state: int | None = None,
    debug: bool = True,
    debug_mode: int = logging.INFO,
    logger_name: str = "sylphy.reductions.factory",
    **kwargs: object,
) -> tuple[object | None, np.ndarray | pl.DataFrame | None]:
    """Run a dimensionality reduction by method name via a unified factory.

    For linear methods, returns ``(fitted_model, transformed)``.
    For non-linear methods, returns ``(None, transformed)`` (current behavior).

    Args:
        method: Reduction method name (case-insensitive).
        dataset: Input matrix of shape ``(N, D)``.
        return_type: Output container type.
        preprocess: Optional preprocessing strategy.
        random_state: Seed for supported estimators.
        debug: Whether to enable logging for this call.
        debug_mode: Logging level for the invocation logger.
        logger_name: Logger name used during dispatch.
        **kwargs: Extra estimator parameters forwarded to the selected method.

    Returns:
        Tuple ``(model, transformed)`` where model is ``None`` for non-linear
        methods and ``transformed`` may be ``None`` on failure.

    Raises:
        ValueError: If the method name is not registered.

    """
    # Child logger for this invocation (level control via `debug`)
    log = logging.getLogger(logger_name)
    log.setLevel(debug_mode if debug else logging.NOTSET)
    add_context(log, method=normalize_name(method))

    key = normalize_name(method)
    if key not in _METHODS:
        all_methods = ", ".join(sorted(_METHODS.keys()))
        msg = f"Unknown reduction method '{method}'. Available: {all_methods}"
        raise ValueError(msg)

    kind, attr = _METHODS[key]
    log.info("Dispatching method='%s' (kind=%s) | preprocess=%s | kwargs=%s", key, kind, preprocess, kwargs)

    if kind == "linear":
        module = import_module(".linear_reductions", package=__package__)
        LinearReduction = module.LinearReduction

        runner = LinearReduction(
            dataset=dataset,
            return_type=return_type,
            preprocess=preprocess,
            random_state=random_state,
            debug=debug,
            debug_mode=debug_mode,
        )
        apply_fn = getattr(runner, attr)
        model, transformed = apply_fn(**kwargs)  # (model, transformed)
        return model, transformed

    # Non-linear
    try:
        module = import_module(".non_linear_reductions", package=__package__)
        NonLinearReductions = module.NonLinearReductions
    except (ImportError, ModuleNotFoundError) as exc:
        wrapped = wrap_optional_dependency_error(
            exc,
            feature=f"Reduction method '{method}'",
            extra="reductions",
            packages=("umap", "umap-learn", "clustpy"),
        )
        if wrapped is not None:
            raise wrapped from exc
        raise
    runner = NonLinearReductions(
        dataset=dataset,
        return_type=return_type,
        preprocess=preprocess,
        random_state=random_state,
        debug=debug,
        debug_mode=debug_mode,
    )
    apply_fn = getattr(runner, attr)
    transformed = apply_fn(**kwargs)  # transformed or None
    return None, transformed
