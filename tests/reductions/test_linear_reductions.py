from __future__ import annotations

import numpy as np
import polars as pl

from sylphy.reductions import LinearReduction


def test_pca_numpy_return(X_small: np.ndarray) -> None:
    """Verify PCA returns numpy array with correct shape."""
    lr = LinearReduction(X_small, return_type="numpy", debug=True)
    _model, Z = lr.apply_pca(n_components=2, random_state=0)
    assert Z is not None
    assert isinstance(Z, np.ndarray)
    assert Z.shape == (X_small.shape[0], 2)


def test_pca_polars_return(df_small: pl.DataFrame) -> None:
    """Verify PCA returns polars DataFrame with named components."""
    lr = LinearReduction(df_small, return_type="polars", debug=True)
    _model, Z = lr.apply_pca(n_components=3, random_state=0)
    assert Z is not None
    assert isinstance(Z, pl.DataFrame)
    assert Z.columns == ["p_1", "p_2", "p_3"]
    assert Z.shape[0] == df_small.shape[0]


def test_truncated_svd_and_factor_analysis(X_small: np.ndarray) -> None:
    """Verify TruncatedSVD and FactorAnalysis produce expected dimensions."""
    lr = LinearReduction(X_small, return_type="numpy", debug=True)
    _, z_svd = lr.apply_truncated_svd(n_components=2, random_state=0)
    _, z_fa = lr.apply_factor_analysis(n_components=2, random_state=0)
    assert z_svd is not None
    assert z_svd.shape[1] == 2
    assert z_fa is not None
    assert z_fa.shape[1] == 2


def test_nmf_on_non_negative_data(X_nonneg: np.ndarray) -> None:
    """Verify NMF works on non-negative data and returns named components."""
    lr = LinearReduction(X_nonneg, return_type="polars", debug=True)
    _, Z = lr.apply_nmf(n_components=2, random_state=0, init="random", max_iter=200)
    assert Z is not None
    assert isinstance(Z, pl.DataFrame)
    assert Z.columns == ["p_1", "p_2"]
