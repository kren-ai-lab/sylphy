# tests/sequence_encoder/test_fft_encoder.py
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from sylphy.sequence_encoder import FFTEncoder


def test_fft_encoder_half_spectrum_and_padding():
    # 3 numeric columns + sequence
    df = pd.DataFrame(
        {
            "p_0": [1.0, 0.0],
            "p_1": [0.0, 1.0],
            "p_2": [1.0, 1.0],
            "sequence": ["AAA", "CCC"],
        }
    )
    enc = FFTEncoder(dataset=df, sequence_column="sequence", debug=True)
    enc.run_process()
    X = enc.coded_dataset
    # Next power of two >= 3 is 4 → half-spectrum = 2
    assert X.drop(columns=["sequence"]).shape[1] == 2
    assert (X.drop(columns=["sequence"]).to_numpy() >= 0).all()
    # Sequence column preserved
    assert list(X["sequence"]) == ["AAA", "CCC"]
