# protein_representation/sequence_encoder/one_hot_encoder.py
from __future__ import annotations

import logging
from typing import Optional, List

import pandas as pd

from .base_encoder import Encoders
from protein_representation.constants.tool_constants import POSITION_RESIDUES


class OneHotEncoder(Encoders):
    """
    One-hot encodes sequences; 20-dim per residue, zero-padded to `max_length`.
    """

    def __init__(
        self,
        dataset: Optional[pd.DataFrame] = None,
        sequence_column: Optional[str] = "sequence",
        max_length: int = 1024,
        debug: bool = False,
        debug_mode: int = logging.INFO,
    ) -> None:
        super().__init__(
            dataset=dataset,
            sequence_column=sequence_column or "sequence",
            max_length=max_length,
            debug=debug,
            debug_mode=debug_mode,
            name_logging=OneHotEncoder.__name__,
        )

    def __generate_vector_by_residue(self, residue: str) -> List[int]:
        vector = [0] * 20
        try:
            position = POSITION_RESIDUES[residue]
            vector[position] = 1
        except KeyError:
            self.__logger__.warning("Residue '%s' not found in mapping.", residue)
        return vector

    def __zero_padding(self, current_length: int) -> List[int]:
        return [0] * (self.max_length * 20 - current_length)

    def __coding_sequence(self, sequence: str) -> List[int]:
        coded = []
        for residue in sequence:
            coded.extend(self.__generate_vector_by_residue(residue))
        if len(sequence) < self.max_length:
            coded += self.__zero_padding(len(coded))
        return coded

    def run_process(self) -> None:
        if not self.status:
            self.__logger__.warning("Encoding skipped; dataset validation failed.")
            return

        try:
            self.__logger__.info("Starting one-hot encoding for %d sequences.", len(self.dataset))
            matrix = [self.__coding_sequence(self.dataset.at[i, self.sequence_column]) for i in self.dataset.index]
            header = [f"p_{i}" for i in range(len(matrix[0]))]
            self.coded_dataset = pd.DataFrame(matrix, columns=header)
            self.coded_dataset[self.sequence_column] = self.dataset[self.sequence_column].values
            self.__logger__.info("One-hot encoding completed with %d features.", self.coded_dataset.shape[1])
        except Exception as e:
            self.status = False
            self.message = f"[ERROR] One-hot encoding failed: {e}"
            self.__logger__.exception(self.message)
            raise RuntimeError(self.message)
