# tests/sequence_encoder/test_ordinal_onehot_frequency.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sylphy.sequence_encoder import OrdinalEncoder, OneHotEncoder, FrequencyEncoder


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
    df = pd.DataFrame({"sequence": ["ABBA".replace("B", "C")]})  # "ACCA"
    enc = FrequencyEncoder(dataset=df, max_length=4, debug=True)
    enc.run_process()
    X = enc.coded_dataset
    v = X.iloc[0, :4].tolist()
    # For "ACCA": counts are A=2/4, C=2/4 → positions [A,C,C,A] = [.5, .5, .5, .5]
    assert v == pytest.approx([0.5, 0.5, 0.5, 0.5], rel=1e-6)
