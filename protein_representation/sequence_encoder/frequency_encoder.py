# protein_representation/sequence_encoder/frequency_encoder.py
from __future__ import annotations

import logging
from typing import Optional, List

import pandas as pd

from .base_encoder import Encoders


class FrequencyEncoder(Encoders):
    """
    Encodes sequences by normalized per-residue frequency; zero-pads to `max_length`.
    """

    def __init__(
        self,
        dataset: Optional[pd.DataFrame] = None,
        sequence_column: str = "sequence",
        max_length: int = 1024,
        debug: bool = False,
        debug_mode: int = logging.INFO,
    ) -> None:
        super().__init__(
            dataset=dataset,
            sequence_column=sequence_column,
            max_length=max_length,
            debug=debug,
            debug_mode=debug_mode,
            name_logging=FrequencyEncoder.__name__,
        )

    def __zero_padding(self, current_length: int) -> List[float]:
        return [0.0 for _ in range(current_length, self.max_length)]

    def __get_residue_count(self, sequence: str, residue: str) -> float:
        return sequence.count(residue) / len(sequence) if sequence else 0.0

    def __coding_sequence(self, sequence: str) -> List[float]:
        coded = [self.__get_residue_count(sequence, r) for r in sequence]
        if len(sequence) < self.max_length:
            coded.extend(self.__zero_padding(len(coded)))
        return coded

    def run_process(self) -> None:
        if not self.status:
            self.__logger__.warning("Encoding skipped due to failed validation.")
            return

        try:
            self.__logger__.info("Starting frequency encoding.")
            matrix = [self.__coding_sequence(self.dataset.at[i, self.sequence_column]) for i in self.dataset.index]
            header = [f"p_{i}" for i in range(self.max_length)]
            self.coded_dataset = pd.DataFrame(matrix, columns=header)
            self.coded_dataset[self.sequence_column] = self.dataset[self.sequence_column].values
            self.__logger__.info("Frequency encoding completed.")
        except Exception as e:
            self.status = False
            self.message = f"[ERROR] Failed to encode sequences: {e}"
            self.__logger__.exception(self.message)
            raise RuntimeError(self.message)
