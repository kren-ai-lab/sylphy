"""Implement residue-frequency encoding for protein sequences."""

from __future__ import annotations

import logging
import re

import polars as pl

from sylphy.constants import residues

from .encoder_base import EncoderBase


class FrequencyEncoder(EncoderBase):
    """Encode sequences by normalized per-residue frequency over the selected alphabet.

    Output is a single |alphabet|-dimensional vector per sequence:
    freq[r] = count(r in sequence) / len(sequence), r in alphabet.
    """

    def __init__(
        self,
        dataset: pl.DataFrame | None = None,
        sequence_column: str = "sequence",
        *,
        allow_extended: bool = False,
        allow_unknown: bool = False,
        debug: bool = False,
        debug_mode: int = logging.INFO,
    ) -> None:
        """Initialize the frequency encoder."""
        super().__init__(
            dataset=dataset,
            sequence_column=sequence_column,
            max_length=10**9,  # no practical cut here; base still guards
            allow_extended=allow_extended,
            allow_unknown=allow_unknown,
            debug=debug,
            debug_mode=debug_mode,
            name_logging=FrequencyEncoder.__name__,
        )
        self._alpha = list(residues(extended=self.allow_extended or self.allow_unknown))

    def run_process(self) -> None:
        """Encode each sequence as normalized residue frequencies."""
        try:
            self.__logger__.info("Starting frequency encoding (alphabet size=%d).", len(self._alpha))
            seq_col = pl.col(self.sequence_column)
            seq_len = seq_col.str.len_chars().cast(pl.Float32)
            freq_exprs = [
                (seq_col.str.count_matches(re.escape(r)).cast(pl.Float32) / seq_len).alias(f"freq_{r}")
                for r in self._alpha
            ]
            self.coded_dataset = self.dataset.select([*freq_exprs, seq_col])
            self.__logger__.info(
                "Frequency encoding completed with %d features.",
                self.coded_dataset.width,
            )
        except Exception as e:
            msg = f"[ERROR] Failed to encode sequences: {e}"
            self.__logger__.exception(msg)
            raise RuntimeError(msg) from e
