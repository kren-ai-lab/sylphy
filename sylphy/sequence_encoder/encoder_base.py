"""Implement the shared base class for sequence encoders."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from sylphy.constants import residues
from sylphy.logging import add_context, get_logger
from sylphy.misc.utils_lib import UtilsLib

if TYPE_CHECKING:
    from pathlib import Path

    from sylphy.types import FileFormat


def _encode_lut_1d(
    lut: np.ndarray, sequences: list[str], max_length: int, *, uppercase: bool = False
) -> np.ndarray:
    N = len(sequences)
    matrix = np.zeros((N, max_length), dtype=lut.dtype)
    for i, seq in enumerate(sequences):
        s = seq.upper() if uppercase else seq
        raw = np.frombuffer(s.encode("ascii"), dtype=np.uint8)
        length = min(len(raw), max_length)
        matrix[i, :length] = lut[raw[:length]]
    return matrix


def _encode_lut_onehot(lut: np.ndarray, sequences: list[str], max_length: int, n_channels: int) -> np.ndarray:
    N = len(sequences)
    matrix = np.zeros((N, max_length * n_channels), dtype=np.uint8)
    for i, seq in enumerate(sequences):
        raw = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
        length = min(len(raw), max_length)
        indices = lut[raw[:length]]
        valid = indices >= 0
        pos = np.nonzero(valid)[0]
        matrix[i, pos * n_channels + indices[valid]] = 1
    return matrix


def encode_ordinal(sequences: list[str], alphabet: tuple[str, ...], max_length: int) -> np.ndarray:
    """Encode sequences as ordinal alphabet indices, zero-padded to ``max_length``.

    Args:
        sequences: Validated amino acid sequences.
        alphabet: Ordered residue alphabet.
        max_length: Output length per sequence.

    Returns:
        int32 array of shape ``(N, max_length)``.

    """
    lut = np.zeros(256, dtype=np.int32)
    for idx, res in enumerate(alphabet):
        lut[ord(res)] = idx
        lut[ord(res.lower())] = idx
    return _encode_lut_1d(lut, sequences, max_length)


def encode_onehot(sequences: list[str], alphabet: tuple[str, ...], max_length: int) -> np.ndarray:
    """Encode sequences as flattened one-hot vectors, zero-padded to ``max_length``.

    Args:
        sequences: Validated amino acid sequences.
        alphabet: Ordered residue alphabet.
        max_length: Output length per sequence.

    Returns:
        uint8 array of shape ``(N, max_length * len(alphabet))``.

    """
    n_channels = len(alphabet)
    lut = np.full(256, -1, dtype=np.intp)
    for idx, res in enumerate(alphabet):
        lut[ord(res)] = idx
        lut[ord(res.lower())] = idx
    return _encode_lut_onehot(lut, sequences, max_length, n_channels)


def encode_physicochemical(sequences: list[str], prop_map: dict[str, float], max_length: int) -> np.ndarray:
    """Encode sequences using a residue → float property mapping, zero-padded to ``max_length``.

    Args:
        sequences: Validated amino acid sequences.
        prop_map: Uppercase residue → numeric property value.
        max_length: Output length per sequence.

    Returns:
        float32 array of shape ``(N, max_length)``.

    """
    lut = np.zeros(256, dtype=np.float32)
    for res, val in prop_map.items():
        try:
            f = float(val)
        except (TypeError, ValueError):
            f = 0.0
        lut[ord(res.upper())] = f
        lut[ord(res.lower())] = f
    return _encode_lut_1d(lut, sequences, max_length, uppercase=True)


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
            has_invalid = pl.col(self.sequence_column).str.contains(invalid_pattern).fill_null(value=True)
            self.dataset = self.dataset.filter(~has_invalid)
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

    def _finalize_encoding(self, sequences: list[str], matrix: np.ndarray) -> None:
        """Populate ``coded_dataset`` from an encoded numpy matrix."""
        col_names = [f"p_{i}" for i in range(matrix.shape[1])]
        self.coded_dataset = pl.from_numpy(matrix, schema=col_names).with_columns(
            pl.Series(self.sequence_column, sequences)
        )

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
