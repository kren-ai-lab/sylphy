from __future__ import annotations

import polars as pl
import polars.selectors as cs

from sylphy.sequence_encoder import KMerEncoder


def test_kmers_tfidf_basic() -> None:
    """Verify k-mer encoder produces TF-IDF features for extracted k-mers."""
    df = pl.DataFrame({"sequence": ["ACDEF", "CDEFG", "DEFGH"]})
    enc = KMerEncoder(dataset=df, size_kmer=3, debug=True)
    enc.run_process()
    X = enc.coded_dataset

    assert "sequence" in X.columns
    expected = {"ACD", "CDE", "DEF"}
    assert expected & set(X.columns)
    numeric = X.select(cs.numeric())
    assert numeric.dtypes[0] in (pl.Float32, pl.Float64)


def test_kmers_tfidf_sparse_flag() -> None:
    """Verify as_sparse=True returns metadata-only coded_dataset and a sparse matrix."""
    import scipy.sparse

    df = pl.DataFrame({"sequence": ["ACDEF", "CDEFG"]})
    enc = KMerEncoder(dataset=df, size_kmer=3, as_sparse=True)
    enc.run_process()

    assert enc.sparse_matrix is not None
    assert scipy.sparse.issparse(enc.sparse_matrix)
    assert enc.coded_dataset.columns == ["sequence"]
