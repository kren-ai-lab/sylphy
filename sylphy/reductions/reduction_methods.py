"""Define shared base behavior for dimensionality-reduction classes."""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from sylphy.constants.tool_configs import get_config
from sylphy.logging import add_context, get_logger

ReturnType = Literal["numpy", "pandas"]
Preprocess = Literal["none", "standardize", "normalize", "robust"]


class Reductions:
    """Base utilities for dimensionality reduction workflows.

    Responsibilities:
        - Hold and validate the input matrix (2D numeric array).
        - Apply optional preprocessing.
        - Provide unified component logging.
        - Convert transformed arrays into NumPy or pandas outputs.

    Args:
        dataset: Input feature matrix of shape ``(N, D)``.
        return_type: Output container used by helper methods.
        preprocess: Optional preprocessing strategy.
        random_state: Random seed for supported models.
        debug: Whether to enable this component logger.
        debug_mode: Logging level used when ``debug`` is enabled.
        name_logging: Child logger suffix under ``sylphy.reductions``.

    """

    def __init__(
        self,
        dataset: np.ndarray | pd.DataFrame,
        *,
        return_type: ReturnType = "numpy",
        preprocess: Preprocess = "none",
        random_state: int | None = None,
        debug: bool = True,
        debug_mode: int = logging.INFO,
        name_logging: str = "Reductions",
    ) -> None:
        """Initialize reduction state, logger, validation, and preprocessing."""
        # Ensure the package logger exists exactly once, then get a child
        _ = get_logger("sylphy")
        self.__logger__ = logging.getLogger(f"sylphy.reductions.{name_logging}")
        self.__logger__.setLevel(debug_mode if debug else logging.NOTSET)
        add_context(self.__logger__, component="reductions", cls=name_logging)

        self.return_type: ReturnType = return_type
        self.preprocess: Preprocess = preprocess
        self.random_state: int = int(get_config().seed if random_state is None else random_state)

        # Normalize dataset → np.ndarray (float32), validate 2D numeric
        if isinstance(dataset, pd.DataFrame):
            dataset = dataset.to_numpy()
        arr = np.asarray(dataset)
        if arr.ndim != 2:
            msg = f"Expected 2D array, got shape {arr.shape}"
            raise ValueError(msg)
        if not np.issubdtype(arr.dtype, np.number):
            msg = "Dataset must be numeric."
            raise TypeError(msg)
        self.dataset: np.ndarray = arr.astype(np.float32, copy=False)

        self.__logger__.info(
            "Initialized %s with dataset shape=%s, dtype=%s | preprocess=%s | seed=%d",
            name_logging,
            self.dataset.shape,
            self.dataset.dtype,
            self.preprocess,
            self.random_state,
        )

        # Apply optional preprocessing in-place
        self._scaler: StandardScaler | MinMaxScaler | RobustScaler | None = None
        self._apply_preprocess()

    # -----------------------------
    # Preprocessing
    # -----------------------------
    def _apply_preprocess(self) -> None:
        if self.preprocess == "none":
            return
        if self.preprocess == "standardize":
            self._scaler = StandardScaler(with_mean=True, with_std=True)
        elif self.preprocess == "normalize":
            self._scaler = MinMaxScaler(feature_range=(0, 1))
        elif self.preprocess == "robust":
            self._scaler = RobustScaler(with_centering=True, with_scaling=True)
        else:
            msg = f"Unknown preprocess option '{self.preprocess}'."
            raise ValueError(msg)

        self.__logger__.info("Applying preprocess: %s", self.preprocess)
        scaler = self._scaler
        if scaler is None:
            return
        self.dataset = scaler.fit_transform(self.dataset).astype(np.float32, copy=False)

    # -----------------------------
    # Output helpers
    # -----------------------------
    def _make_headers(self, n_components: int) -> list[str]:
        return [f"p_{i + 1}" for i in range(n_components)]

    def generate_dataset_post_reduction(
        self,
        transform_values: np.ndarray | list[list[float]],
        n_components: int | None = None,
    ) -> np.ndarray | pd.DataFrame:
        """Build the final reduced output.

        - If ``return_type='numpy'`` → returns a numpy array (N, K).
        - If ``return_type='pandas'`` → returns a DataFrame with columns ``p_1..p_K``.
        """
        transform_array = np.asarray(transform_values)
        if transform_array.ndim != 2:
            msg = f"Expected 2D reduced array, got shape {transform_array.shape}"
            raise ValueError(msg)

        k = transform_array.shape[1]
        if n_components is None:
            n_components = k
        if k != n_components:
            msg = f"Expected {n_components} components, but got {k}"
            raise ValueError(msg)

        try:
            if self.return_type == "numpy":
                self.__logger__.info("Prepared NumPy output with %d components.", n_components)
                return transform_array
            headers = self._make_headers(n_components)
            self.__logger__.info("Prepared pandas DataFrame with %d components.", n_components)
            return pd.DataFrame(data=transform_array, columns=pd.Index(headers))

        except Exception as e:
            self.__logger__.error("Failed to build post-reduction output: %s", e)
            raise
