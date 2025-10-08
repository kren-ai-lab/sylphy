from __future__ import annotations

from typing import Optional, Literal
import logging
import pandas as pd

from sylphy.logging import get_logger, add_context
from sylphy.constants import residues
from sylphy.misc.utils_lib import UtilsLib


class Encoders:
    """
    Common pre-processing and validation for protein/peptide sequence encoders.

    This class validates the input alphabet and maximum length constraints,
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
    allow_extended : bool, default=False
        If True, accept extended amino acids (B, Z, X, U, O).
    allow_unknown : bool, default=False
        If True, allow 'X' even when `allow_extended=False`.
    debug : bool, default=False
        If True, the child logger is set to `debug_mode`.
    debug_mode : int, default=logging.INFO
        Logging level for this encoder's child logger (e.g., logging.DEBUG).
    name_logging : str, default="sequence_encoder.encoder"
        Child logger suffix; emitted as
        ``sylphy.sequence_encoder.<name_logging>``.

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
    allow_extended : bool
        Whether extended alphabet is enabled.
    allow_unknown : bool
        Whether 'X' is allowed when not using extended alphabet.
    __logger__ : logging.Logger
        Child logger for this encoder.
    """

    def __init__(
        self,
        dataset: Optional[pd.DataFrame] = None,
        sequence_column: str = "sequence",
        max_length: int = 1024,
        allow_extended: bool = False,
        allow_unknown: bool = False,
        debug: bool = False,
        debug_mode: int = logging.INFO,
        name_logging: str = "sequence_encoder.encoder",
    ) -> None:
        # Ensure top-level logger is initialized once and create a child logger
        _ = get_logger("sylphy")
        self.__logger__ = logging.getLogger(f"sylphy.sequence_encoder.{name_logging}")
        self.__logger__.setLevel(debug_mode if debug else logging.NOTSET)
        add_context(self.__logger__, component="sequence_encoder", encoder=name_logging)

        self.dataset = dataset if dataset is not None else pd.DataFrame()
        self.sequence_column = sequence_column
        self.max_length = max_length
        self.allow_extended = allow_extended
        self.allow_unknown = allow_unknown

        self.status = True
        self.message = ""
        self.coded_dataset = pd.DataFrame()

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
        """Run alphabet validation and length filtering."""
        if self.sequence_column not in self.dataset.columns:
            self.status = False
            self.message = f"[ERROR] Column '{self.sequence_column}' not found in dataset."
            self.__logger__.error(self.message)
            raise ValueError(self.message)

        try:
            self.__logger__.info("Validating alphabet (%s).",
                                 "extended" if self.allow_extended or self.allow_unknown else "canonical")
            self.check_allowed_alphabet()
            self.__logger__.info("Validating sequence lengths (≤ %d).", self.max_length)
            self.process_length_sequences()
        except Exception as e:
            self.status = False
            self.message = f"[ERROR] Failed during revision steps: {e}"
            self.__logger__.exception(self.message)
            raise RuntimeError(self.message)

    def check_allowed_alphabet(self) -> None:
        """Keep only sequences composed of the selected alphabet."""
        try:
            alpha = set(residues(extended=self.allow_extended or self.allow_unknown))
            if not self.allow_extended and self.allow_unknown:
                alpha.add("X")  # allow unknown explicitly if requested

            def _ok(seq: str) -> bool:
                return all((r in alpha) for r in seq)

            mask = [ _ok(seq) for seq in self.dataset[self.sequence_column] ]
            self.dataset["is_canon"] = mask
            before = len(self.dataset)
            self.dataset = self.dataset[self.dataset["is_canon"]].copy()
            removed = before - len(self.dataset)
            self.__logger__.info("Filtered sequences outside alphabet: %d removed.", removed)
        except Exception:
            self.__logger__.exception("[ERROR] Failed during alphabet validation.")
            raise RuntimeError("Failed during alphabet validation.")

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

    def export_encoder(self, path: str, file_format: Literal["csv", "npy", "npz", "parquet"] = "csv") -> None:
        """Persist the encoded matrix to disk."""
        UtilsLib.export_data(
            df_encoded=self.coded_dataset,
            path=path,
            base_message="Encoded features",
            file_format=file_format,
        )
