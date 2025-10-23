from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd

from sylphy.constants import residues

from .base_encoder import Encoders


class FrequencyEncoder(Encoders):
    """
    Encode sequences by normalized per-residue frequency over the selected alphabet.

    Output is a single |alphabet|-dimensional vector per sequence:
    freq[r] = count(r in sequence) / len(sequence), r in alphabet.
    """

    def __init__(
        self,
        dataset: Optional[pd.DataFrame] = None,
        sequence_column: str = "sequence",
        allow_extended: bool = False,
        allow_unknown: bool = False,
        debug: bool = False,
        debug_mode: int = logging.INFO,
    ) -> None:
        # Note: max_length is irrelevant for frequency features; keep base validation flow
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

    def __encode_sequence(self, sequence: str) -> List[float]:
        L = float(len(sequence)) if sequence else 1.0
        return [sequence.count(r) / L for r in self._alpha]

    def run_process(self) -> None:
        if not self.status:
            self.__logger__.warning("Encoding skipped due to failed validation.")
            return

        try:
            self.__logger__.info("Starting frequency encoding (alphabet size=%d).", len(self._alpha))
            matrix = [
                self.__encode_sequence(self.dataset.at[i, self.sequence_column]) for i in self.dataset.index
            ]
            header = [f"freq_{r}" for r in self._alpha]
            self.coded_dataset = pd.DataFrame(matrix, columns=header)
            self.coded_dataset[self.sequence_column] = self.dataset[self.sequence_column].values
            self.__logger__.info(
                "Frequency encoding completed with %d features.", self.coded_dataset.shape[1]
            )
        except Exception as e:
            self.status = False
            self.message = f"[ERROR] Failed to encode sequences: {e}"
            self.__logger__.exception(self.message)
            raise RuntimeError(self.message) from e
