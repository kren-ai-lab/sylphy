"""Implement FFT-based encoding over numeric sequence features."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import polars.selectors as cs
from scipy.fft import fft

from sylphy.logging import add_context, get_logger
from sylphy.misc.utils_lib import UtilsLib

if TYPE_CHECKING:
    from pathlib import Path

    from sylphy.types import FileFormat


class FFTEncoder:
    """Apply FFT to each numeric row (after removing the `sequence_column`).

    Notes:
        Expects a numeric matrix in the dataframe except for ``sequence_column``,
        which is preserved and re-attached to the output.

    """

    def __init__(
        self,
        dataset: pl.DataFrame,
        sequence_column: str = "sequence",
        *,
        debug: bool = False,
        debug_mode: int = logging.INFO,
    ) -> None:
        """Initialize the FFT encoder and precompute padding parameters."""
        self.sequence_column = sequence_column
        _ = get_logger("sylphy")
        self.__logger__ = logging.getLogger("sylphy.sequence_encoder.FFTEncoder")
        self.__logger__.setLevel(debug_mode if debug else logging.NOTSET)
        add_context(self.__logger__, component="sequence_encoder", encoder="FFTEncoder")

        # Store sequences as a Series; strip from numeric frame
        self.sequence_series: pl.Series = dataset[sequence_column]
        self._numeric: pl.DataFrame = dataset.select(cs.numeric())

        self.max_length = self._numeric.width
        self.stop_value: int = 0
        self.coded_dataset: pl.DataFrame | None = None

        self.init_process()

    def _get_near_pow(self) -> None:
        self.__logger__.info("Computing nearest power-of-two for padding.")
        self.stop_value = int(2 ** int(np.ceil(np.log2(max(1, self.max_length)))))
        self.__logger__.info("FFT stop value set to %d.", self.stop_value)

    def init_process(self) -> None:
        """Compute FFT output length for the dataset."""
        self.__logger__.info("Initializing FFT encoding process.")
        self._get_near_pow()

    def encoding_dataset(self) -> None:
        """Encode the dataset by applying FFT row-wise."""
        try:
            self.__logger__.info("Encoding dataset with FFT.")
            arr = self._numeric.to_numpy()
            pad = self.stop_value - self.max_length
            if pad > 0:
                arr = np.pad(arr, ((0, 0), (0, pad)), mode="constant")
            n_out = self.stop_value // 2
            yf = fft(arr, axis=1, workers=-1)
            output = np.abs(yf[:, :n_out]).astype(np.float32)
            col_names = [f"p_{i}" for i in range(n_out)]
            self.coded_dataset = pl.from_numpy(output, schema=col_names).with_columns(self.sequence_series)
            self.__logger__.info("FFT encoding complete. Shape: %s", self.coded_dataset.shape)
        except Exception as e:
            self.__logger__.error("Failed to encode dataset with FFT: %s", e)
            msg = "FFT encoding failed."
            raise RuntimeError(msg) from e

    def run_process(self) -> None:
        """Run FFT encoding."""
        self.encoding_dataset()

    def export_encoder(
        self,
        path: str | Path,
        file_format: FileFormat = "csv",
        *,
        df_encoder: pl.DataFrame | None = None,
    ) -> None:
        """Export encoded FFT features to disk."""
        data = df_encoder if df_encoder is not None else self.coded_dataset
        if data is None:
            msg = "No encoded FFT dataset available for export."
            raise ValueError(msg)

        UtilsLib.export_data(
            df_encoded=data,
            path=path,
            base_message="FFT encoder output",
            file_format=file_format,
        )
