"""Implement the shared base class for sequence encoders."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import polars as pl

from sylphy.constants import residues
from sylphy.logging import add_context, get_logger
from sylphy.misc.utils_lib import UtilsLib

if TYPE_CHECKING:
    from pathlib import Path

    from sylphy.types import FileFormat


class EncoderBase(ABC):
    """Common pre-processing and validation for protein/peptide sequence encoders.

    This class validates the input alphabet and maximum length constraints,
    preserving selected columns for downstream encoders.

    Args:
        dataset: Input dataset containing sequences in ``sequence_column``.
        sequence_column: Column that holds sequence strings.
        max_length: Maximum allowed sequence length.
        allow_extended: Whether extended amino acids are accepted.
        allow_unknown: Whether ``X`` is allowed when extended alphabet is disabled.
        debug: Whether to set the child logger level to ``debug_mode``.
        debug_mode: Logging level for the child encoder logger.
        name_logging: Child logger suffix under ``sylphy.sequence_encoder``.

    Attributes:
        dataset: Cleaned and validated dataset.
        coded_dataset: Output feature matrix populated by subclasses.
        max_length: Maximum permitted sequence length.
        allow_extended: Whether extended alphabet mode is enabled.
        allow_unknown: Whether ``X`` is allowed when not using extended mode.
        __logger__: Child logger for this encoder.

    """

    def __init__(
        self,
        dataset: pl.DataFrame | None = None,
        sequence_column: str = "sequence",
        max_length: int = 1024,
        *,
        allow_extended: bool = False,
        allow_unknown: bool = False,
        debug: bool = False,
        debug_mode: int = logging.INFO,
        name_logging: str = "sequence_encoder.encoder",
    ) -> None:
        """Initialize the base encoder and run input validation."""
        _ = get_logger("sylphy")
        self.__logger__ = logging.getLogger(f"sylphy.sequence_encoder.{name_logging}")
        self.__logger__.setLevel(debug_mode if debug else logging.NOTSET)
        add_context(self.__logger__, component="sequence_encoder", encoder=name_logging)

        self.dataset: pl.DataFrame = dataset if dataset is not None else pl.DataFrame()
        self.sequence_column = sequence_column
        self.max_length = max_length
        self.allow_extended = allow_extended
        self.allow_unknown = allow_unknown

        self.coded_dataset: pl.DataFrame = pl.DataFrame()

        if dataset is None:
            msg = "[ERROR] No dataset provided to encoder."
            self.__logger__.error(msg)
            raise ValueError(msg)

        self.make_revisions()

    # ----------------------------
    # Validation steps
    # ----------------------------

    def make_revisions(self) -> None:
        """Run alphabet validation and length filtering."""
        if self.sequence_column not in self.dataset.columns:
            msg = f"[ERROR] Column '{self.sequence_column}' not found in dataset."
            self.__logger__.error(msg)
            raise ValueError(msg)

        try:
            self.__logger__.info(
                "Validating alphabet (%s).",
                "extended" if self.allow_extended or self.allow_unknown else "canonical",
            )
            self.check_allowed_alphabet()
            self.__logger__.info("Validating sequence lengths (≤ %d).", self.max_length)
            self.process_length_sequences()
        except Exception as e:
            msg = f"[ERROR] Failed during revision steps: {e}"
            self.__logger__.exception(msg)
            raise RuntimeError(msg) from e

    def check_allowed_alphabet(self) -> None:
        """Keep only sequences composed of the selected alphabet."""
        try:
            alpha = set(residues(extended=self.allow_extended or self.allow_unknown))
            if not self.allow_extended and self.allow_unknown:
                alpha.add("X")

            # Build a regex that matches any character NOT in the allowed alphabet
            escaped = re.escape("".join(sorted(alpha)))
            invalid_pattern = f"[^{escaped}]"

            before = len(self.dataset)
            self.dataset = self.dataset.filter(
                ~pl.col(self.sequence_column).str.contains(invalid_pattern)
            )
            removed = before - len(self.dataset)
            self.__logger__.info("Filtered sequences outside alphabet: %d removed.", removed)
        except Exception as exc:
            self.__logger__.exception("[ERROR] Failed during alphabet validation.")
            msg = "Failed during alphabet validation."
            raise RuntimeError(msg) from exc

    def process_length_sequences(self) -> None:
        """Filter out sequences longer than `max_length`."""
        try:
            before = len(self.dataset)
            self.dataset = self.dataset.filter(
                pl.col(self.sequence_column).str.len_chars() <= self.max_length
            )
            removed = before - len(self.dataset)
            self.__logger__.info("Filtered long sequences: %d removed.", removed)
        except Exception as exc:
            self.__logger__.exception("[ERROR] Failed during length validation.")
            msg = "Failed during length validation."
            raise RuntimeError(msg) from exc

    # ----------------------------
    # IO
    # ----------------------------

    @abstractmethod
    def run_process(self) -> None:
        """Run the encoder-specific processing routine."""

    def export_encoder(
        self,
        path: str | Path,
        file_format: FileFormat = "csv",
    ) -> None:
        """Persist the encoded matrix to disk."""
        UtilsLib.export_data(
            df_encoded=self.coded_dataset,
            path=path,
            base_message="Encoded features",
            file_format=file_format,
        )
