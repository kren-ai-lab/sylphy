from __future__ import annotations

import inspect
import logging
import traceback
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import (
    NMF,
    PCA,
    FactorAnalysis,
    FastICA,
    IncrementalPCA,
    LatentDirichletAllocation,
    MiniBatchNMF,
    MiniBatchSparsePCA,
    SparsePCA,
    TruncatedSVD,
)

from .reduction_methods import Preprocess, Reductions, ReturnType


class LinearReduction(Reductions):
    """
    Linear dimensionality reductions (PCA/ICA/NMF/LDA/etc.) with a consistent API.

    All methods log their parameters and shapes, return either NumPy arrays or
    DataFrames depending on ``return_type``. For estimators where the trained model
    is also valuable, we return ``(fitted_model, transformed)``.
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
    ) -> None:
        super().__init__(
            dataset=dataset,
            return_type=return_type,
            preprocess=preprocess,
            random_state=random_state,
            debug=debug,
            debug_mode=debug_mode,
            name_logging=LinearReduction.__name__,
        )
        self.__logger__.info("Initialized LinearReduction with dataset shape=%s", self.dataset.shape)

    # -----------------------------
    # Helpers
    # -----------------------------
    def _init_with_seed(self, cls: type[Any], kwargs: dict[str, Any]) -> Any:
        sig = inspect.signature(cls.__init__)
        params = set(sig.parameters.keys())
        k = dict(kwargs)
        if "random_state" in params and "random_state" not in k:
            k["random_state"] = self.random_state
        return cls(**k)

    def _apply_model(
        self,
        model: Any,
        method_name: str,
        n_components: int | None = None,
    ) -> tuple[object, np.ndarray | pd.DataFrame | None]:
        """Fit/transform wrapper with logging and error handling."""
        try:
            params = getattr(model, "get_params", lambda: {})()
            self.__logger__.info("Applying %s with params=%s", method_name, params)
            transformed = model.fit_transform(self.dataset)
            k = (
                n_components
                if n_components is not None
                else getattr(model, "n_components", transformed.shape[1])
            )
            self.__logger__.info("%s successful. Output shape=%s", method_name, transformed.shape)
            return model, self.generate_dataset_post_reduction(transformed, k)
        except Exception as e:
            self.__logger__.error("%s failed: %s", method_name, e)
            self.__logger__.debug(traceback.format_exc())
            return model, None

    # -----------------------------
    # Public API
    # -----------------------------
    def apply_pca(self, **kwargs) -> tuple[PCA, np.ndarray | pd.DataFrame | None]:
        model = self._init_with_seed(PCA, kwargs)
        return self._apply_model(model, "PCA", kwargs.get("n_components"))

    def apply_incremental_pca(self, **kwargs) -> tuple[IncrementalPCA, np.ndarray | pd.DataFrame | None]:
        model = self._init_with_seed(IncrementalPCA, kwargs)
        return self._apply_model(model, "IncrementalPCA", kwargs.get("n_components"))

    def apply_sparse_pca(self, **kwargs) -> tuple[SparsePCA, np.ndarray | pd.DataFrame | None]:
        model = self._init_with_seed(SparsePCA, kwargs)
        return self._apply_model(model, "SparsePCA", kwargs.get("n_components"))

    def apply_minibatch_sparse_pca(
        self, **kwargs
    ) -> tuple[MiniBatchSparsePCA, np.ndarray | pd.DataFrame | None]:
        model = self._init_with_seed(MiniBatchSparsePCA, kwargs)
        return self._apply_model(model, "MiniBatchSparsePCA", kwargs.get("n_components"))

    def apply_fast_ica(self, **kwargs) -> tuple[FastICA, np.ndarray | pd.DataFrame | None]:
        model = self._init_with_seed(FastICA, kwargs)
        return self._apply_model(model, "FastICA", kwargs.get("n_components"))

    def apply_truncated_svd(self, **kwargs) -> tuple[TruncatedSVD, np.ndarray | pd.DataFrame | None]:
        model = self._init_with_seed(TruncatedSVD, kwargs)
        return self._apply_model(model, "TruncatedSVD", kwargs.get("n_components"))

    def apply_factor_analysis(self, **kwargs) -> tuple[FactorAnalysis, np.ndarray | pd.DataFrame | None]:
        model = self._init_with_seed(FactorAnalysis, kwargs)
        return self._apply_model(model, "FactorAnalysis", kwargs.get("n_components"))

    def apply_nmf(self, **kwargs) -> tuple[NMF, np.ndarray | pd.DataFrame | None]:
        # NMF requires non-negative data; users should ensure this upstream or via preprocess
        model = self._init_with_seed(NMF, kwargs)
        return self._apply_model(model, "NMF", kwargs.get("n_components"))

    def apply_minibatch_nmf(self, **kwargs) -> tuple[MiniBatchNMF, np.ndarray | pd.DataFrame | None]:
        model = self._init_with_seed(MiniBatchNMF, kwargs)
        return self._apply_model(model, "MiniBatchNMF", kwargs.get("n_components"))

    def apply_latent_dirichlet_allocation(
        self, **kwargs
    ) -> tuple[LatentDirichletAllocation, np.ndarray | pd.DataFrame | None]:
        model = self._init_with_seed(LatentDirichletAllocation, kwargs)
        return self._apply_model(model, "LatentDirichletAllocation", kwargs.get("n_components"))
