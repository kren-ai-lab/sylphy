# tests/reductions/test_factory.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from protein_representation.reductions import reduce_dimensionality


def test_factory_linear_pandas(df_small, caplog):
    caplog.set_level("INFO", logger="protein_representation.reductions.factory")
    model, Z = reduce_dimensionality(
        "pca",
        df_small,
        return_type="pandas",
        n_components=2,
        random_state=0,
        debug=True,
    )
    assert Z is not None
    assert list(Z.columns) == ["p_1", "p_2"]
    assert any("Dispatching method='pca' (kind=linear)" in r.message for r in caplog.records)


def test_factory_nonlinear_numpy(X_small, caplog):
    caplog.set_level("INFO", logger="protein_representation.reductions.factory")
    model, Z = reduce_dimensionality(
        "isomap",
        X_small,
        return_type="numpy",
        n_components=2,
        n_neighbors=3,
        debug=True,
    )
    assert model is None
    assert Z is not None and isinstance(Z, np.ndarray) and Z.shape[1] == 2
    assert any("Dispatching method='isomap' (kind=nonlinear)" in r.message for r in caplog.records)


def test_factory_unknown_raises(X_small):
    with pytest.raises(ValueError):
        reduce_dimensionality("nope", X_small)
