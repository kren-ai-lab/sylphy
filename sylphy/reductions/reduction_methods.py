from __future__ import annotations

from typing import List, Union, Optional, Literal
import logging
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from sylphy.logging import get_logger, add_context
from sylphy.constants.tool_configs import get_config

ReturnType = Literal["numpy", "pandas"]
Preprocess = Literal["none", "standardize", "normalize", "robust"]


class Reductions:
    """
    Base utilities for dimensionality reduction workflows.

    Responsibilities
    ----------------
    - Hold and validate the input matrix (2D numeric array).
    - Optional preprocessing: standardization/normalization/robust scaling.
    - Provide a unified logger consistent with the library.
    - Convert transformed arrays into **NumPy** (default) or **pandas** outputs
      with standardized column names (``p_1, p_2, ...``).

    Parameters
    ----------
    dataset : np.ndarray | pd.DataFrame
        Input feature matrix of shape (N, D).
    return_type : {"numpy", "pandas"}, default "numpy"
        Output container used by helper methods.
    preprocess : {"none","standardize","normalize","robust"}, default "none"
        Optional preprocessing to apply before the reduction.
    random_state : int | None, default None
        Random seed for models that accept it. If None, uses ToolConfig.seed.
    debug : bool, default True
        If True, enable this component's logger (child logger).
    debug_mode : int, default logging.INFO
        Logging level when ``debug=True`` (e.g., ``logging.DEBUG``).
    name_logging : str, default="Reductions"
        Suffix used for the child logger name ``sylphy.reductions.<name_logging>``.
    """

    def __init__(
        self,
        dataset: Union[np.ndarray, pd.DataFrame],
        *,
        return_type: ReturnType = "numpy",
        preprocess: Preprocess = "none",
        random_state: Optional[int] = None,
        debug: bool = True,
        debug_mode: int = logging.INFO,
        name_logging: str = "Reductions",
    ) -> None:
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
            dataset = dataset.values
        arr = np.asarray(dataset)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {arr.shape}")
        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError("Dataset must be numeric.")
        self.dataset: np.ndarray = arr.astype(np.float32, copy=False)

        self.__logger__.info(
            "Initialized %s with dataset shape=%s, dtype=%s | preprocess=%s | seed=%d",
            name_logging, self.dataset.shape, self.dataset.dtype, self.preprocess, self.random_state
        )

        # Apply optional preprocessing in-place
        self._scaler = None
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
            self._scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        elif self.preprocess == "robust":
            self._scaler = RobustScaler(with_centering=True, with_scaling=True)
        else:
            raise ValueError(f"Unknown preprocess option '{self.preprocess}'.")

        self.__logger__.info("Applying preprocess: %s", self.preprocess)
        self.dataset = self._scaler.fit_transform(self.dataset).astype(np.float32, copy=False)

    # -----------------------------
    # Output helpers
    # -----------------------------
    def _make_headers(self, n_components: int) -> List[str]:
        return [f"p_{i+1}" for i in range(n_components)]

    def generate_dataset_post_reduction(
        self,
        transform_values: Union[np.ndarray, List[List[float]]],
        n_components: Optional[int] = None,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Build the final reduced output.

        - If ``return_type='numpy'`` → returns a numpy array (N, K).
        - If ``return_type='pandas'`` → returns a DataFrame with columns ``p_1..p_K``.
        """
        try:
            transform_array = np.asarray(transform_values)
            if transform_array.ndim != 2:
                raise ValueError(f"Expected 2D reduced array, got shape {transform_array.shape}")

            k = transform_array.shape[1]
            if n_components is None:
                n_components = k
            if k != n_components:
                raise ValueError(f"Expected {n_components} components, but got {k}")

            if self.return_type == "numpy":
                self.__logger__.info("Prepared NumPy output with %d components.", n_components)
                return transform_array
            else:
                headers = self._make_headers(n_components)
                self.__logger__.info("Prepared pandas DataFrame with %d components.", n_components)
                return pd.DataFrame(data=transform_array, columns=headers)

        except Exception as e:
            self.__logger__.error("Failed to build post-reduction output: %s", e)
            raise
