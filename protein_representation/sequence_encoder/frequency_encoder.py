import pandas as pd
from typing import Optional, List

from .base_encoder import Encoders
import logging

class FrequencyEncoder(Encoders):
    """
    Encodes protein or peptide sequences based on residue frequency per position.

    Each residue in a sequence is represented by the frequency of its occurrence
    within the sequence. Sequences shorter than max_length are zero-padded.

    Inherits from
    ----------
    Encoders : Performs validation and filtering of sequences.

    Parameters
    ----------
    dataset : pd.DataFrame
        Input dataset containing sequences.
    sequence_column : str
        Column name where sequences are stored.
    ignore_columns : list, optional
        Columns to preserve during encoding.
    max_length : int, default=1024
        Maximum length to which sequences are padded.
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        sequence_column: str = "sequence",
        max_length: int = 1024,
        debug:bool=False,
        debug_mode:int=logging.INFO

    ) -> None:
        super().__init__(
            dataset=dataset,
            sequence_column=sequence_column,
            max_length=max_length,
            debug=debug,
            debug_mode=debug_mode,
            name_logging=FrequencyEncoder.__name__
        )

    def __zero_padding(self, current_length: int) -> List[float]:
        """
        Generate zero padding to reach max_length.

        Parameters
        ----------
        current_length : int
            The current length of the sequence vector.

        Returns
        -------
        List[float]
            A list of zeros to pad the sequence vector.
        """
        return [0.0 for _ in range(current_length, self.max_length)]

    def __get_residue_count(self, sequence: str, residue: str) -> float:
        """
        Calculate the normalized frequency of a residue in a sequence.

        Parameters
        ----------
        sequence : str
            The input sequence.
        residue : str
            The amino acid residue to count.

        Returns
        -------
        float
            Normalized frequency of the residue in the sequence.
        """
        return sequence.count(residue) / len(sequence) if sequence else 0.0

    def __coding_sequence(self, sequence: str) -> List[float]:
        """
        Encode a sequence as a vector of residue frequencies.

        Parameters
        ----------
        sequence : str
            The input sequence.

        Returns
        -------
        List[float]
            The frequency-encoded vector of the sequence.
        """
        coded_vector = [self.__get_residue_count(sequence, residue) for residue in sequence]

        if len(sequence) < self.max_length:
            coded_vector.extend(self.__zero_padding(len(coded_vector)))

        return coded_vector

    def run_process(self) -> None:
        """
        Perform frequency encoding of sequences.

        Encoded vectors are stored in self.coded_dataset.
        If validation failed during init, no encoding is applied.

        Raises
        ------
        RuntimeError
            If encoding fails due to internal error.
        """
        if not self.status:
            self.__logger__.warning("Encoding skipped due to failed validation.")
            return

        try:
            self.__logger__.info("Starting frequency encoding process.")
            matrix_coded = [
                self.__coding_sequence(self.dataset.at[index, self.sequence_column])
                for index in self.dataset.index
            ]

            header = [f"p_{i}" for i in range(self.max_length)]
            self.coded_dataset = pd.DataFrame(data=matrix_coded, columns=header)
            self.coded_dataset[self.sequence_column] = self.dataset[self.sequence_column].values
            
            self.__logger__.info("Frequency encoding completed successfully.")

        except Exception as e:
            self.status = False
            self.message = f"[ERROR] Failed to encode sequences: {str(e)}"
            self.__logger__.exception(self.message)
            raise RuntimeError(self.message)
