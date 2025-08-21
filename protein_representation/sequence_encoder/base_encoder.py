# protein_representation/sequence_encoder/base_encoder.py
from __future__ import annotations

from typing import Optional, Literal
import logging
import pandas as pd

from protein_representation.logging import get_logger, add_context
from protein_representation.constants.tool_constants import (get_residue, get_index, LIST_DESCRIPTORS_SEQUENCE_NON_NUMERIC,
                                                             LIST_DESCRIPTORS_SEQUENCE, POSITION_RESIDUES, LIST_RESIDUES, BASE_URL_AAINDEX,
                                                             BASE_URL_CLUSTERS_DESCRIPTORS)

from protein_representation.misc.utils_lib import UtilsLib


class Encoders:
    """
    Common pre-processing and validation for protein/peptide sequence encoders.

    This class validates canonical amino acids and maximum length constraints,
    preserving selected columns for downstream encoders.

    Parameters
    ----------
    dataset : pd.DataFrame, optional
        Input dataset containing sequences in `sequence_column`. If None, the
        instance is created with `status=False` and a message is logged.
    sequence_column : str, default="sequence"
        Column that holds the sequence strings.
    max_length : int, default=1024
        Maximum allowed sequence length; longer sequences are filtered out.
    debug : bool, default=False
        If True, the *child* logger is set to `debug_mode`.
    debug_mode : int, default=logging.INFO
        Logging level for this encoder's child logger (e.g., logging.DEBUG).
    name_logging : str, default="sequence_encoder.encoder"
        Child logger suffix; emitted as
        ``protein_representation.sequence_encoder.<name_logging>``.

    Attributes
    ----------
    dataset : pd.DataFrame
        Cleaned & validated dataset. Unchanged if validation fails.
    coded_dataset : pd.DataFrame
        Output feature matrix (to be filled by concrete encoders).
    status : bool
        Validation status.
    message : str
        Last status/error message.
    max_length : int
        Maximum permitted sequence length.
    __logger__ : logging.Logger
        Child logger for this encoder.
    """

    def __init__(
        self,
        dataset: Optional[pd.DataFrame] = None,
        sequence_column: str = "sequence",
        max_length: int = 1024,
        debug: bool = False,
        debug_mode: int = logging.INFO,
        name_logging: str = "sequence_encoder.encoder",
    ) -> None:

        # Ensure the top-level library logger is configured once,
        # and use a child logger for this encoder.
        _ = get_logger("protein_representation")
        self.__logger__ = logging.getLogger(
            f"protein_representation.sequence_encoder.{name_logging}"
        )
        self.__logger__.setLevel(debug_mode if debug else logging.NOTSET)
        add_context(self.__logger__, component="sequence_encoder", encoder=name_logging)

        self.dataset = dataset if dataset is not None else pd.DataFrame()
        self.sequence_column = sequence_column
        self.max_length = max_length

        self.status = True
        self.message = ""
        self.coded_dataset = pd.DataFrame()

        # Fast-exit if dataset was not provided
        if dataset is None:
            self.status = False
            self.message = "[ERROR] No dataset provided to encoder."
            self.__logger__.error(self.message)
            return

        try:
            self.make_revisions()
        except Exception as e:
            self.status = False
            self.message = f"[ERROR] Initialization failed: {e}"
            self.__logger__.exception(self.message)

    # ----------------------------
    # Validation steps
    # ----------------------------

    def make_revisions(self) -> None:
        """Run residue-canonicity and length validations."""
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
            self.message = f"[ERROR] Failed during revision steps: {e}"
            self.__logger__.exception(self.message)
            raise RuntimeError(self.message)

    def check_canonical_residues(self) -> None:
        """Keep only sequences composed of canonical residues."""
        try:
            canon_mask = [
                all(res in LIST_RESIDUES for res in seq)
                for seq in self.dataset[self.sequence_column]
            ]
            self.dataset["is_canon"] = canon_mask
            before = len(self.dataset)
            self.dataset = self.dataset[self.dataset["is_canon"]].copy()
            removed = before - len(self.dataset)
            self.__logger__.info("Filtered non-canonical sequences: %d removed.", removed)
        except Exception:
            self.__logger__.exception("[ERROR] Failed during canonical residue check.")
            raise RuntimeError("Failed during canonical residue check.")

    def process_length_sequences(self) -> None:
        """Filter out sequences longer than `max_length`."""
        try:
            self.dataset["length_sequence"] = self.dataset[self.sequence_column].str.len()
            self.dataset["is_valid_length"] = (self.dataset["length_sequence"] <= self.max_length).astype(int)
            before = len(self.dataset)
            self.dataset = self.dataset[self.dataset["is_valid_length"] == 1].copy()
            removed = before - len(self.dataset)
            self.__logger__.info("Filtered long sequences: %d removed.", removed)
        except Exception:
            self.__logger__.exception("[ERROR] Failed during length validation.")
            raise RuntimeError("Failed during length validation.")

    # ----------------------------
    # IO
    # ----------------------------

    def export_encoder(self, path: str, file_format: Literal["csv", "npy"] = "csv") -> None:
        """Persist the encoded matrix to disk."""
        UtilsLib.export_data(
            df_encoded=self.coded_dataset,
            path=path,
            __logger__=self.__logger__,
            base_message="Encoded generated",
            file_format=file_format,
        )
