# tests/sequence_encoder/test_kmers_tfidf.py
from __future__ import annotations

import pandas as pd

from protein_representation.sequence_encoder import KMersEncoders


def test_kmers_tfidf_basic():
    df = pd.DataFrame({"sequence": ["ABCDE", "BCDEF", "CDEFG"]})
    enc = KMersEncoders(dataset=df, size_kmer=3, debug=True)
    enc.run_process()
    X = enc.coded_dataset
    # Has at least some k-mer columns and preserves 'sequence'
    assert "sequence" in X.columns
    # A few typical 3-mers should appear capitalized
    expected = {"ABC", "BCD", "CDE", "DEF"}
    assert expected & set(X.columns)  # non-empty intersection
    # Features are float32-ish (tf-idf)
    assert X.drop(columns=["sequence"]).to_numpy().dtype.kind in ("f", "d")
