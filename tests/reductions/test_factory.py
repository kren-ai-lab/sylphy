from __future__ import annotations

import numpy as np
import pytest

from sylphy.reductions import reduce_dimensionality


def _any_log_contains(caplog, needle: str) -> bool:
    """Helper: returns True if any captured record contains `needle` (case-insensitive)."""
    needle = needle.lower()
    return any(needle in r.getMessage().lower() for r in caplog.records)


def test_factory_linear_pandas(df_small, caplog):
    """
    Linear dispatch: 'pca' should return a fitted linear model and a pandas DataFrame with
    the expected number of components (2).
    Logging content is optional and tolerant to wording.
    """
    caplog.set_level("INFO")  # do not rely on a specific logger name
    model, Z = reduce_dimensionality(
        "pca",
        df_small,
        return_type="pandas",
        n_components=2,
        random_state=0,
        debug=True,
    )
    # Output checks
    assert Z is not None
    assert list(Z.columns) == ["p_1", "p_2"]
    assert Z.shape[0] == df_small.shape[0]

    # Model checks: should look like a fitted PCA (common PCA attributes)
    # We avoid hard-importing sklearn here; instead, check characteristic attrs.
    assert hasattr(model, "components_")
    assert hasattr(model, "explained_variance_ratio_")
    assert getattr(model, "n_components_", 2) == 2

    # Optional, non-failing log check (robust to message text and logger name)
    # If no logs captured, we don't fail the test.
    if caplog.records:
        assert _any_log_contains(caplog, "pca") or _any_log_contains(caplog, "linear")


def test_factory_nonlinear_numpy(X_small, caplog):
    """
    Non-linear dispatch: 'isomap' usually returns (None, np.ndarray) in our API.
    """
    caplog.set_level("INFO")
    model, Z = reduce_dimensionality(
        "isomap",
        X_small,
        return_type="numpy",
        n_components=2,
        n_neighbors=3,
        debug=True,
    )
    # Output checks
    assert model is None  # non-linear path returns no sklearn model object
    assert Z is not None and isinstance(Z, np.ndarray) and Z.shape == (X_small.shape[0], 2)

    # Optional, non-failing log check
    if caplog.records:
        assert _any_log_contains(caplog, "isomap") or _any_log_contains(caplog, "nonlinear")


def test_factory_unknown_raises(X_small):
    """Unknown method should raise a ValueError."""
    with pytest.raises(ValueError):
        reduce_dimensionality("nope", X_small)
