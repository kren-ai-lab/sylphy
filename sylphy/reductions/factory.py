from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
import pandas as pd

from sylphy.logging import add_context, get_logger

from .linear_reductions import LinearReduction
from .non_linear_reductions import NonLinearReductions
from .reduction_methods import Preprocess, ReturnType

DatasetLike = np.ndarray | pd.DataFrame
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
    methods: dict[str, tuple[Kind, str]] = {}
    for key, attr in _LINEAR_METHODS.items():
        methods[key] = (_LINEAR_KIND, attr)
    for key, attr in _NONLINEAR_METHODS.items():
        methods[key] = (_NONLINEAR_KIND, attr)
    return methods


_METHODS: dict[str, tuple[Kind, str]] = _build_methods()

# --- logging: ensure parent, then a child logger for the factory -------------
_ = get_logger("sylphy")
logger = logging.getLogger("sylphy.reductions.factory")
add_context(logger, component="reductions", facility="factory")


def get_available_methods(kind: Kind | None = None) -> list[str]:
    """
    List available method names.

    Parameters
    ----------
    kind : {"linear", "nonlinear"}, optional
        Filter by method family. If None, returns all.

    Returns
    -------
    list of str
        Method names you can pass to `reduce_dimensionality`.
    """
    if kind is None:
        return sorted(_METHODS.keys())
    if kind == "linear":
        return sorted(_LINEAR_METHODS.keys())
    if kind == "nonlinear":
        return sorted(_NONLINEAR_METHODS.keys())
    raise ValueError("kind must be one of: None, 'linear', 'nonlinear'.")


def is_linear_method(name: str) -> bool:
    """Return True if `name` is a registered linear method."""
    return name.lower() in _LINEAR_METHODS


def is_nonlinear_method(name: str) -> bool:
    """Return True if `name` is a registered non-linear method."""
    return name.lower() in _NONLINEAR_METHODS


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
    **kwargs: Any,
) -> tuple[object | None, np.ndarray | pd.DataFrame | None]:
    """
    Run a dimensionality reduction by method name via a unified factory.

    For linear methods, returns ``(fitted_model, transformed)``.
    For non-linear methods, returns ``(None, transformed)`` (current behavior).

    Parameters
    ----------
    method : str
        Reduction method name (case-insensitive). See `get_available_methods()`.
    dataset : np.ndarray or pandas.DataFrame
        Input matrix of shape (N, D).
    return_type : {"numpy", "pandas"}, default "numpy"
        Output container for the transformed data.
    preprocess : {"none","standardize","normalize","robust"}, default "none"
        Optional preprocessing to apply before the reduction.
    random_state : int | None, default None
        Seed for supported estimators; defaults to ToolConfig.seed.
    debug : bool, default True
        Enable/disable logging for this call.
    debug_mode : int, default logging.INFO
        Logging level used by the internal logger.
    logger_name : str, default "sylphy.reductions.factory"
        Name for the logger (child of package logger).
    **kwargs : Any
        Extra parameters forwarded to the underlying estimator constructor
        (e.g., n_components, random_state, perplexity, n_neighbors, ...).

    Returns
    -------
    (model, transformed) : tuple
        model : object or None
            Fitted estimator for linear methods; None for non-linear methods.
        transformed : np.ndarray or pandas.DataFrame or None
            Reduced data; None if the run failed.

    Raises
    ------
    ValueError
        If the method name is not registered.
    """
    # Child logger for this invocation (level control via `debug`)
    log = logging.getLogger(logger_name)
    log.setLevel(debug_mode if debug else logging.NOTSET)
    add_context(log, method=method.lower())

    key = method.lower().strip()
    if key not in _METHODS:
        all_methods = ", ".join(sorted(_METHODS.keys()))
        raise ValueError(f"Unknown reduction method '{method}'. Available: {all_methods}")

    kind, attr = _METHODS[key]
    log.info("Dispatching method='%s' (kind=%s) | preprocess=%s | kwargs=%s", key, kind, preprocess, kwargs)

    if kind == "linear":
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
