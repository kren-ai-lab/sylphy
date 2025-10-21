from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sylphy.sequence_encoder import FrequencyEncoder, OneHotEncoder, OrdinalEncoder


def test_ordinal_shapes_and_padding(toy_df):
    enc = OrdinalEncoder(dataset=toy_df, max_length=6, debug=True)
    enc.run_process()
    X = enc.coded_dataset
    # 6 features (p_0..p_5) + 'sequence' column
    assert list(X.columns[:6]) == [f"p_{i}" for i in range(6)]
    assert X.shape == (len(toy_df), 6 + 1)
    # Row 0: "ACD" -> padded with zeros to length 6
    row0 = X.iloc[0, :6].tolist()
    assert row0[3:] == [0, 0, 0]
    # Values are non-negative integers
    assert all(int(v) == v and v >= 0 for v in row0)


def test_onehot_shapes_and_sums(toy_df):
    enc = OneHotEncoder(dataset=toy_df, max_length=5, debug=True)
    enc.run_process()
    X = enc.coded_dataset
    # Expect 5 * 20 = 100 features + 'sequence'
    assert X.shape[1] == 100 + 1
    # For "ACD" (length=3), one-hot has exactly 3 "1"s; the rest are zeros
    row0 = X.iloc[0, :100].to_numpy()
    assert row0.sum() == 3  # exactly one per residue
    assert set(np.unique(row0)) <= {0, 1}


def test_frequency_invariants():
    from sylphy.constants.tool_constants import LIST_RESIDUES

    df = pd.DataFrame({"sequence": ["ACCA"]})
    enc = FrequencyEncoder(dataset=df, debug=True)
    enc.run_process()
    X = enc.coded_dataset

    # Select feature columns (exclude 'sequence')
    feat_cols = [c for c in X.columns if c != "sequence"]

    # Prefer residue-named columns if present; otherwise assume canonical order
    if set(LIST_RESIDUES).issubset(set(feat_cols)):
        v = X.loc[0, LIST_RESIDUES].to_numpy()
    else:
        # Fallback: assume encoder uses canonical residue order across the first 20 cols
        v = X.loc[0, feat_cols[:20]].to_numpy()

    # Invariants: 20 residues, sum to 1.0, only A and C are 0.5 each for "ACCA"
    assert len(v) == 20
    assert v.sum() == pytest.approx(1.0, rel=1e-6)

    idx = {res: i for i, res in enumerate(LIST_RESIDUES)}
    a_i, c_i = idx["A"], idx["C"]
    assert v[a_i] == pytest.approx(0.5, rel=1e-6)
    assert v[c_i] == pytest.approx(0.5, rel=1e-6)

    # All other residues should be ~0
    for res, i in idx.items():
        if res not in {"A", "C"}:
            assert v[i] == pytest.approx(0.0, abs=1e-12)
