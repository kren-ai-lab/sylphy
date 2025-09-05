# protein_representation/sequence_encoder/kmers_encoder.py
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .base_encoder import Encoders


class KMersEncoders(Encoders):
    """
    TF-IDF encode k-merized sequences.
    """

    def __init__(
        self,
        dataset: Optional[pd.DataFrame] = None,
        sequence_column: Optional[str] = "sequence",
        size_kmer: int = 3,
        debug: bool = False,
        debug_mode: int = logging.INFO,
    ) -> None:
        super().__init__(
            dataset=dataset,
            sequence_column=sequence_column or "sequence",
            debug=debug,
            debug_mode=debug_mode,
            name_logging=KMersEncoders.__name__,
        )
        self.size_kmer = size_kmer

    @staticmethod
    def kmer(seq: str, kmer_length: int = 3) -> str:
        return " ".join(seq[i : i + kmer_length] for i in range(len(seq) - kmer_length + 1))

    def run_process(self) -> None:
        if not self.status:
            self.__logger__.warning("Encoding aborted due to failed validation.")
            return

        try:
            self.__logger__.info("Starting k-mer encoding (k=%d).", self.size_kmer)
            self.dataset["kmer_sequence"] = self.dataset[self.sequence_column].apply(
                lambda x: self.kmer(x, self.size_kmer)
            )

            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(self.dataset["kmer_sequence"]).astype("float32")

            self.coded_dataset = pd.DataFrame(
                data=X.toarray(),
                columns=[c.upper() for c in vectorizer.get_feature_names_out()],
            )
            self.coded_dataset[self.sequence_column] = self.dataset[self.sequence_column].values

            if "sequence" in self.coded_dataset.columns and self.sequence_column != "sequence":
                self.coded_dataset.drop(columns=["sequence"], inplace=True)

            self.__logger__.info("TF-IDF k-mer encoding completed with %d features.", self.coded_dataset.shape[1])
        except Exception as e:
            self.status = False
            self.message = f"[ERROR] K-mer encoding failed: {e}"
            self.__logger__.exception(self.message)
            raise RuntimeError(self.message)
