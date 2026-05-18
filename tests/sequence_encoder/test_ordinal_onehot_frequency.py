from __future__ import annotations

import numpy as np
import polars as pl
import polars.selectors as cs
import pytest

from sylphy.sequence_encoder import FrequencyEncoder, OneHotEncoder, OrdinalEncoder


def test_ordinal_shapes_and_padding(toy_df: pl.DataFrame) -> None:
    """Verify ordinal encoding produces correct shape with zero padding."""
    enc = OrdinalEncoder(dataset=toy_df, max_length=6)
    enc.run_process()
    X = enc.coded_dataset

    assert X.columns[:6] == [f"p_{i}" for i in range(6)]
    assert X.shape == (len(toy_df), 7)
    assert X.row(0, named=False)[3:6] == (0, 0, 0)
    assert all(int(v) == v and v >= 0 for v in X.row(0, named=False)[:6])


def test_onehot_shapes_and_sums(toy_df: pl.DataFrame) -> None:
    """Verify one-hot encoding produces binary matrix with expected shape."""
    enc = OneHotEncoder(dataset=toy_df, max_length=5)
    enc.run_process()
    X = enc.coded_dataset

    assert X.width == 101  # 5 * 20 + sequence column
    row0 = X.select(cs.exclude("sequence")).row(0, named=False)
    arr = np.array(row0)
    assert arr.sum() == 3
    assert set(np.unique(arr)) <= {0, 1}


def test_frequency_invariants() -> None:
    """Verify frequency encoder produces normalized residue frequencies."""

    df = pl.DataFrame({"sequence": ["ACCA"]})
    enc = FrequencyEncoder(dataset=df)
    enc.run_process()
    X = enc.coded_dataset

    feat_cols = [c for c in X.columns if c != "sequence"]
    freq_map = {col: X[col][0] for col in feat_cols}

    a_col = "freq_A"
    c_col = "freq_C"

    assert len(feat_cols) == 20
    total = sum(freq_map.values())
    assert total == pytest.approx(1.0, rel=1e-6)
    assert freq_map.get(a_col, 0) == pytest.approx(0.5, rel=1e-6)
    assert freq_map.get(c_col, 0) == pytest.approx(0.5, rel=1e-6)
    for col, val in freq_map.items():
        if col not in {a_col, c_col}:
            assert val == pytest.approx(0.0, abs=1e-12)
