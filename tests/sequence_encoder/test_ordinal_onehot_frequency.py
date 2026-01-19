from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sylphy.sequence_encoder import FrequencyEncoder, OneHotEncoder, OrdinalEncoder


def test_ordinal_shapes_and_padding(toy_df):
    """Verify ordinal encoding produces correct shape with zero padding."""
    enc = OrdinalEncoder(dataset=toy_df, max_length=6)
    enc.run_process()
    X = enc.coded_dataset

    assert list(X.columns[:6]) == [f"p_{i}" for i in range(6)]
    assert X.shape == (len(toy_df), 7)
    assert X.iloc[0, 3:6].tolist() == [0, 0, 0]
    assert all(int(v) == v and v >= 0 for v in X.iloc[0, :6])


def test_onehot_shapes_and_sums(toy_df):
    """Verify one-hot encoding produces binary matrix with expected shape."""
    enc = OneHotEncoder(dataset=toy_df, max_length=5)
    enc.run_process()
    X = enc.coded_dataset

    assert X.shape[1] == 101  # 5 * 20 + sequence column
    row0 = X.iloc[0, :100].to_numpy()
    assert row0.sum() == 3
    assert set(np.unique(row0)) <= {0, 1}


def test_frequency_invariants():
    """Verify frequency encoder produces normalized residue frequencies."""
    from sylphy.constants.tool_constants import LIST_RESIDUES

    df = pd.DataFrame({"sequence": ["ACCA"]})
    enc = FrequencyEncoder(dataset=df)
    enc.run_process()
    X = enc.coded_dataset

    feat_cols = [c for c in X.columns if c != "sequence"]
    v = (
        X.loc[0, LIST_RESIDUES].to_numpy()  # type: ignore[missing-attribute, bad-index]
        if set(LIST_RESIDUES).issubset(feat_cols)
        else X.loc[0, feat_cols[:20]].to_numpy()  # type: ignore[missing-attribute, bad-index]
    )

    assert len(v) == 20
    assert v.sum() == pytest.approx(1.0, rel=1e-6)

    idx = {res: i for i, res in enumerate(LIST_RESIDUES)}
    assert v[idx["A"]] == pytest.approx(0.5, rel=1e-6)
    assert v[idx["C"]] == pytest.approx(0.5, rel=1e-6)
    for res, i in idx.items():
        if res not in {"A", "C"}:
            assert v[i] == pytest.approx(0.0, abs=1e-12)
