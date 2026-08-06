from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from sylphy.reductions import reduce_dimensionality


def _any_log_contains(caplog: pytest.LogCaptureFixture, needle: str) -> bool:
    """Helper: returns True if any captured record contains `needle` (case-insensitive)."""
    needle = needle.lower()
    return any(needle in r.getMessage().lower() for r in caplog.records)


def test_factory_linear_polars(df_small: pl.DataFrame, caplog: pytest.LogCaptureFixture) -> None:
    """Linear dispatch: 'pca' should return a fitted linear model and a polars DataFrame."""
    caplog.set_level("INFO")
    model, Z = reduce_dimensionality(
        "pca",
        df_small,
        return_type="polars",
        n_components=2,
        random_state=0,
        debug=True,
    )
    assert Z is not None
    assert isinstance(Z, pl.DataFrame)
    assert Z.columns == ["p_1", "p_2"]
    assert Z.shape[0] == df_small.shape[0]

    assert hasattr(model, "components_")
    assert hasattr(model, "explained_variance_ratio_")
    assert getattr(model, "n_components_", 2) == 2

    if caplog.records:
        assert _any_log_contains(caplog, "pca") or _any_log_contains(caplog, "linear")


def test_factory_nonlinear_numpy(X_small: np.ndarray, caplog: pytest.LogCaptureFixture) -> None:
    """Non-linear dispatch: 'isomap' returns (None, np.ndarray)."""
    caplog.set_level("INFO")
    model, Z = reduce_dimensionality(
        "isomap",
        X_small,
        return_type="numpy",
        n_components=2,
        n_neighbors=3,
        debug=True,
    )
    assert model is None
    assert Z is not None
    assert isinstance(Z, np.ndarray)
    assert Z.shape == (X_small.shape[0], 2)

    if caplog.records:
        assert _any_log_contains(caplog, "isomap") or _any_log_contains(caplog, "nonlinear")


def test_factory_unknown_raises(X_small: np.ndarray) -> None:
    """Unknown method should raise a ValueError."""
    with pytest.raises(ValueError, match=r"Unknown reduction method"):
        reduce_dimensionality("nope", X_small)
