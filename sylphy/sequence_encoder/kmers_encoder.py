from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .base_encoder import Encoders


class KMersEncoders(Encoders):
    """
    TF-IDF encode k-merized sequences (word-level analyzer).
    """

    def __init__(
        self,
        dataset: Optional[pd.DataFrame] = None,
        sequence_column: Optional[str] = "sequence",
        size_kmer: int = 3,
        allow_extended: bool = False,
        allow_unknown: bool = False,
        debug: bool = False,
        debug_mode: int = logging.INFO,
    ) -> None:
        super().__init__(
            dataset=dataset,
            sequence_column=sequence_column or "sequence",
            max_length=10**9,  # not used here
            allow_extended=allow_extended,
            allow_unknown=allow_unknown,
            debug=debug,
            debug_mode=debug_mode,
            name_logging=KMersEncoders.__name__,
        )
        self.size_kmer = size_kmer

    @staticmethod
    def kmer(seq: str, k: int = 3) -> str:
        return " ".join(seq[i : i + k] for i in range(max(0, len(seq) - k + 1)))

    def run_process(self) -> None:
        if not self.status:
            self.__logger__.warning("Encoding aborted due to failed validation.")
            return

        try:
            self.__logger__.info("Starting k-mer encoding (k=%d).", self.size_kmer)
            self.dataset["kmer_sequence"] = self.dataset[self.sequence_column].apply(
                lambda x: self.kmer(x, self.size_kmer)
            )

            vectorizer = TfidfVectorizer(
                analyzer="word",
                token_pattern=r"(?u)\b\w+\b",
                dtype=np.float64,
            )
            X = vectorizer.fit_transform(self.dataset["kmer_sequence"])

            self.coded_dataset = pd.DataFrame(
                data=X.toarray(),
                columns=[c.upper() for c in vectorizer.get_feature_names_out()],
            )
            self.coded_dataset[self.sequence_column] = self.dataset[self.sequence_column].values

            if "sequence" in self.coded_dataset.columns and self.sequence_column != "sequence":
                self.coded_dataset.drop(columns=["sequence"], inplace=True)

            self.__logger__.info(
                "TF-IDF k-mer encoding completed with %d features.", self.coded_dataset.shape[1]
            )
        except Exception as e:
            self.status = False
            self.message = f"[ERROR] K-mer encoding failed: {e}"
            self.__logger__.exception(self.message)
            raise RuntimeError(self.message)
