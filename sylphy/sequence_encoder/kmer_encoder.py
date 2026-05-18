"""Implement TF-IDF k-mer encoding for sequence datasets."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer

from .encoder_base import EncoderBase

if TYPE_CHECKING:
    import scipy.sparse


class KMerEncoder(EncoderBase):
    """TF-IDF encode k-merized sequences (word-level analyzer).

    By default returns a dense ``pl.DataFrame`` (``coded_dataset``).
    When ``as_sparse=True`` the vectorizer output is kept as a scipy sparse
    matrix and ``coded_dataset`` contains only the ``sequence_column`` metadata.
    Access the sparse matrix via ``self.sparse_matrix``.
    """

    def __init__(
        self,
        dataset: pl.DataFrame | None = None,
        sequence_column: str | None = "sequence",
        size_kmer: int = 3,
        *,
        as_sparse: bool = False,
        allow_extended: bool = False,
        allow_unknown: bool = False,
        debug: bool = False,
        debug_mode: int = logging.INFO,
    ) -> None:
        """Initialize the k-mer encoder."""
        super().__init__(
            dataset=dataset,
            sequence_column=sequence_column or "sequence",
            max_length=10**9,  # not used here
            allow_extended=allow_extended,
            allow_unknown=allow_unknown,
            debug=debug,
            debug_mode=debug_mode,
            name_logging=KMerEncoder.__name__,
        )
        self.size_kmer = size_kmer
        self.as_sparse = as_sparse
        self.sparse_matrix: scipy.sparse.csr_matrix | None = None

    @staticmethod
    def _tokenize(seq: str, k: int) -> str:
        """Return whitespace-separated overlapping k-mers for a sequence."""
        return " ".join(seq[i : i + k] for i in range(max(0, len(seq) - k + 1)))

    def run_process(self) -> None:
        """Vectorize k-mer text with TF-IDF and store the encoded dataset."""
        try:
            self.__logger__.info("Starting k-mer encoding (k=%d).", self.size_kmer)

            # Tokenize each sequence using a polars expression (no per-row Python apply)
            sequences = self.dataset[self.sequence_column].to_list()
            k = self.size_kmer
            tokenized = self.dataset.lazy().select(
                pl.col(self.sequence_column)
                .map_batches(
                    lambda s: pl.Series([self._tokenize(seq, k) for seq in s.to_list()]),
                    return_dtype=pl.String,
                )
                .alias("_kmer_seq")
            ).collect()["_kmer_seq"].to_list()

            vectorizer = TfidfVectorizer(
                analyzer="word",
                token_pattern=r"(?u)\b\w+\b",  # noqa: S106
                dtype=np.float32,
            )
            X = vectorizer.fit_transform(tokenized)
            feature_names = [c.upper() for c in vectorizer.get_feature_names_out()]

            if self.as_sparse:
                self.sparse_matrix = X
                self.coded_dataset = pl.DataFrame(
                    {self.sequence_column: sequences},
                    schema={self.sequence_column: pl.String},
                )
                self.__logger__.info(
                    "TF-IDF k-mer encoding completed (sparse). Features: %d.", len(feature_names)
                )
            else:
                dense = pl.from_numpy(X.toarray(), schema=feature_names).with_columns(
                    pl.Series(self.sequence_column, sequences)
                )
                self.coded_dataset = dense
                self.__logger__.info(
                    "TF-IDF k-mer encoding completed with %d features.", self.coded_dataset.width
                )
        except Exception as e:
            msg = f"[ERROR] K-mer encoding failed: {e}"
            self.__logger__.exception(msg)
            raise RuntimeError(msg) from e
