import pandas as pd
from typing import Optional, List, Literal
import logging
from protein_representation.misc.constants import Constant
from .logging_config import setup_logger
from protein_representation.misc.utils_lib import UtilsLib

class Encoders:
    """
    A class for preprocessing and validating protein or peptide sequences
    within a pandas DataFrame.

    The class performs checks for canonical amino acid residues and filters out
    sequences that exceed a specified maximum length. It retains specific columns
    from the original dataset and creates a processed dataset for downstream analysis.

    Parameters
    ----------
    dataset : pd.DataFrame
        The input dataset containing biological sequences.
    sequence_column : str, default="sequence"
        Name of the column that holds the protein/peptide sequences.
    ignore_columns : list, optional
        Columns to preserve in the encoded dataset without processing.
    max_length : int, default=1024
        Maximum allowed sequence length.

    Attributes
    ----------
    dataset : pd.DataFrame
        Cleaned and validated dataset.
    coded_dataset : pd.DataFrame
        Resulting dataset with preserved metadata columns.
    status : bool
        Indicates whether the encoding process was successful.
    message : str
        Status message explaining validation failure, if any.
    max_length : int
        Maximum sequence length allowed.
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        sequence_column: str = "sequence",
        max_length: int = 1024,
        debug:bool=False,
        debug_mode:int=logging.INFO,
        name_logging:str="sequence_encoder.encoder",
    ) -> None:
        
        self.dataset: pd.DataFrame = dataset
        self.sequence_column: str = sequence_column
        self.max_length: int = max_length
        
        self.status: bool = True
        self.message: str = ""
        self.coded_dataset: pd.DataFrame = pd.DataFrame()

        self.__logger__ = setup_logger(
            name=name_logging,
            level=debug_mode,
            enable=debug)

        try:            
            self.make_revisions()
        except Exception as e:
            self.status = False
            self.message = f"[ERROR] Initialization failed: {str(e)}"
            self.__logger__.exception(self.message)

    def make_revisions(self) -> None:
        """
        Perform sequence column validation and preprocessing.

        Raises
        ------
        ValueError
            If sequence column is not present in the dataset.
        """
        if self.sequence_column not in self.dataset.columns:
            self.status = False
            self.message = f"[ERROR] Column '{self.sequence_column}' not found in dataset."
            self.__logger__.error(self.message)
            raise ValueError(self.message)

        try:
            self.__logger__.info("Validating canonical residues.")
            self.check_canonical_residues()
            self.__logger__.info("Validating sequence lengths.")
            self.process_length_sequences()
        except Exception as e:
            self.status = False
            self.message = f"[ERROR] Failed during revision steps: {str(e)}"
            self.__logger__.exception(self.message)
            raise RuntimeError(self.message)

    def check_canonical_residues(self) -> None:
        """
        Filters the dataset to retain only those sequences that contain
        valid canonical residues as defined in Constant.LIST_RESIDUES.
        """
        try:
            canon_sequences = [
                all(residue in Constant.LIST_RESIDUES for residue in seq)
                for seq in self.dataset[self.sequence_column]
            ]
            self.dataset["is_canon"] = canon_sequences
            before_filter = len(self.dataset)
            self.dataset = self.dataset[self.dataset["is_canon"]].copy()
            after_filter = len(self.dataset)
            self.__logger__.info(f"Filtered non-canonical sequences: {before_filter - after_filter} removed.")
        except Exception as e:
            self.__logger__.exception("[ERROR] Failed during canonical residue check.")
            raise RuntimeError(f"Failed during canonical residue check: {str(e)}")

    def process_length_sequences(self) -> None:
        """
        Filters out sequences longer than the specified maximum length.
        Adds a 'length_sequence' column and keeps only valid sequences.
        """
        try:
            self.dataset["length_sequence"] = self.dataset[self.sequence_column].str.len()
            self.dataset["is_valid_length"] = (self.dataset["length_sequence"] <= self.max_length).astype(int)
            before_filter = len(self.dataset)
            self.dataset = self.dataset[self.dataset["is_valid_length"] == 1].copy()
            after_filter = len(self.dataset)
            self.__logger__.info(f"Filtered long sequences: {before_filter - after_filter} removed.")
        except Exception as e:
            self.__logger__.exception("[ERROR] Failed during length validation.")
            raise RuntimeError(f"Failed during length validation: {str(e)}")

    def export_encoder(
        self,
        path: str,
        file_format: Literal["csv", "npy"] = "csv"
    ) -> None:
        """
        Save the coded matrix to disk.
        """
        UtilsLib.export_data(
            df_encoded=self.coded_dataset,
            path=path,
            __logger__= self.__logger__,
            base_message= "Encoded generated",
            file_format= file_format
        )