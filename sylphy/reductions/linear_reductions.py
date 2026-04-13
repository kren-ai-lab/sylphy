"""Implement linear dimensionality-reduction wrappers."""

from __future__ import annotations

import inspect
import logging
import traceback
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast

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

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


class _SupportsFitTransform(Protocol):
    def fit_transform(self, X: Any, y: Any = None) -> Any:
        ...


ModelT = TypeVar("ModelT", bound=_SupportsFitTransform)


class LinearReduction(Reductions):
    """Linear dimensionality reductions (PCA/ICA/NMF/LDA/etc.) with a consistent API.

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
        """Initialize linear-reduction utilities."""
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
    def _init_with_seed(self, cls: type[ModelT], kwargs: dict[str, object]) -> ModelT:
        sig = inspect.signature(cls.__init__)
        params = set(sig.parameters.keys())
        k = dict(kwargs)
        if "random_state" in params and "random_state" not in k:
            k["random_state"] = self.random_state
        return cls(**k)

    def _apply_model(
        self,
        model: ModelT,
        method_name: str,
        n_components: int | None = None,
    ) -> tuple[ModelT, np.ndarray | pd.DataFrame | None]:
        """Fit/transform wrapper with logging and error handling."""
        try:
            params = getattr(model, "get_params", dict)()
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
    def apply_pca(self, **kwargs: object) -> tuple[PCA, np.ndarray | pd.DataFrame | None]:
        """Apply PCA and return the fitted model with transformed data."""
        model = self._init_with_seed(PCA, kwargs)
        return self._apply_model(model, "PCA", cast("int | None", kwargs.get("n_components")))

    def apply_incremental_pca(
        self, **kwargs: object,
    ) -> tuple[IncrementalPCA, np.ndarray | pd.DataFrame | None]:
        """Apply IncrementalPCA and return the fitted model with transformed data."""
        model = self._init_with_seed(IncrementalPCA, kwargs)
        return self._apply_model(model, "IncrementalPCA", cast("int | None", kwargs.get("n_components")))

    def apply_sparse_pca(self, **kwargs: object) -> tuple[SparsePCA, np.ndarray | pd.DataFrame | None]:
        """Apply SparsePCA and return the fitted model with transformed data."""
        model = self._init_with_seed(SparsePCA, kwargs)
        return self._apply_model(model, "SparsePCA", cast("int | None", kwargs.get("n_components")))

    def apply_minibatch_sparse_pca(
        self, **kwargs: object,
    ) -> tuple[MiniBatchSparsePCA, np.ndarray | pd.DataFrame | None]:
        """Apply MiniBatchSparsePCA and return the fitted model with transformed data."""
        model = self._init_with_seed(MiniBatchSparsePCA, kwargs)
        return self._apply_model(model, "MiniBatchSparsePCA", cast("int | None", kwargs.get("n_components")))

    def apply_fast_ica(self, **kwargs: object) -> tuple[FastICA, np.ndarray | pd.DataFrame | None]:
        """Apply FastICA and return the fitted model with transformed data."""
        model = self._init_with_seed(FastICA, kwargs)
        return self._apply_model(model, "FastICA", cast("int | None", kwargs.get("n_components")))

    def apply_truncated_svd(self, **kwargs: object) -> tuple[TruncatedSVD, np.ndarray | pd.DataFrame | None]:
        """Apply TruncatedSVD and return the fitted model with transformed data."""
        model = self._init_with_seed(TruncatedSVD, kwargs)
        return self._apply_model(model, "TruncatedSVD", cast("int | None", kwargs.get("n_components")))

    def apply_factor_analysis(
        self, **kwargs: object,
    ) -> tuple[FactorAnalysis, np.ndarray | pd.DataFrame | None]:
        """Apply FactorAnalysis and return the fitted model with transformed data."""
        model = self._init_with_seed(FactorAnalysis, kwargs)
        return self._apply_model(model, "FactorAnalysis", cast("int | None", kwargs.get("n_components")))

    def apply_nmf(self, **kwargs: object) -> tuple[NMF, np.ndarray | pd.DataFrame | None]:
        """Apply NMF and return the fitted model with transformed data."""
        # NMF requires non-negative data; users should ensure this upstream or via preprocess
        model = self._init_with_seed(NMF, kwargs)
        return self._apply_model(model, "NMF", cast("int | None", kwargs.get("n_components")))

    def apply_minibatch_nmf(self, **kwargs: object) -> tuple[MiniBatchNMF, np.ndarray | pd.DataFrame | None]:
        """Apply MiniBatchNMF and return the fitted model with transformed data."""
        model = self._init_with_seed(MiniBatchNMF, kwargs)
        return self._apply_model(model, "MiniBatchNMF", cast("int | None", kwargs.get("n_components")))

    def apply_latent_dirichlet_allocation(
        self, **kwargs: object,
    ) -> tuple[LatentDirichletAllocation, np.ndarray | pd.DataFrame | None]:
        """Apply LatentDirichletAllocation and return model with transformed data."""
        model = self._init_with_seed(LatentDirichletAllocation, kwargs)
        return self._apply_model(
            model, "LatentDirichletAllocation", cast("int | None", kwargs.get("n_components")),
        )
