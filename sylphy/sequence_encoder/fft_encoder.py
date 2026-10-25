"""Implement FFT-based encoding over numeric sequence features."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import polars.selectors as cs
from scipy.fft import fft

from sylphy.exceptions import SylphyRuntimeError
from sylphy.feature_io import check_ids, write_features
from sylphy.logging import add_context, get_logger

if TYPE_CHECKING:
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
        id_col: str = "id",
        debug: bool = False,
        debug_mode: int = logging.INFO,
    ) -> None:
        """Initialize the FFT encoder and precompute padding parameters."""
        self.sequence_column = sequence_column
        self.id_col = id_col
        self._ids = dataset.select(pl.col(id_col)) if id_col in dataset.columns else pl.DataFrame()
        _ = get_logger("sylphy")
        self.__logger__ = logging.getLogger("sylphy.sequence_encoder.FFTEncoder")
        self.__logger__.setLevel(debug_mode if debug else logging.NOTSET)
        add_context(self.__logger__, component="sequence_encoder", encoder="FFTEncoder")

        # Store sequences as a Series; strip from numeric frame
        self.sequence_series: pl.Series = dataset[sequence_column]
        self._numeric: pl.DataFrame = dataset.drop(id_col, strict=False).select(cs.numeric())

        self.max_length = self._numeric.width
        self.stop_value: int = 0
        self.coded_dataset: pl.DataFrame = pl.DataFrame()

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
            raise SylphyRuntimeError(msg) from e

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
        """Export encoded FFT features to disk, keyed by the constructor's ``id_col``.

        Output matches the CLI: tables have ``id`` first and no sequence column.

        Raises:
            DataError: If the input dataset had no ``id_col``, or its ids are empty or duplicated.

        """
        data = df_encoder if df_encoder is not None else self.coded_dataset
        ids = check_ids(self._ids, self.id_col)
        write_features(data.drop(self.sequence_column, strict=False), ids, Path(path), file_format)
