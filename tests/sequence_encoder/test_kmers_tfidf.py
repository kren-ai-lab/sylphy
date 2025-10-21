from __future__ import annotations

import pandas as pd

from sylphy.sequence_encoder import KMersEncoders


def test_kmers_tfidf_basic():
    """Verify k-mer encoder produces TF-IDF features for extracted k-mers."""
    df = pd.DataFrame({"sequence": ["ABCDE", "BCDEF", "CDEFG"]})
    enc = KMersEncoders(dataset=df, size_kmer=3, debug=True)
    enc.run_process()
    X = enc.coded_dataset

    assert "sequence" in X.columns
    expected = {"ABC", "BCD", "CDE", "DEF"}
    assert expected & set(X.columns)
    assert X.drop(columns=["sequence"]).to_numpy().dtype.kind in ("f", "d")
