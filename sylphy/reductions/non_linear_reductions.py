"""Implement non-linear dimensionality-reduction wrappers."""

from __future__ import annotations

import inspect
import logging
import traceback
from typing import TYPE_CHECKING, Any, Callable, TypeVar, cast  # noqa: UP035

import numpy as np
import umap.umap_ as umap
from clustpy.partition import DipExt
from sklearn.decomposition import DictionaryLearning, MiniBatchDictionaryLearning
from sklearn.manifold import MDS, TSNE, Isomap, LocallyLinearEmbedding, SpectralEmbedding

from .reduction_methods import Preprocess, Reductions, ReturnType

if TYPE_CHECKING:
    import polars as pl


ModelT = TypeVar("ModelT")


class NonLinearReductions(Reductions):
    """Non-linear dimensionality reductions with a unified interface.

    Supported methods
    -----------------
    - t-SNE, Isomap, MDS, LLE, Spectral Embedding, UMAP
    - DictionaryLearning, MiniBatchDictionaryLearning
    - DipExt (ClustPy)
    """

    def __init__(
        self,
        dataset: np.ndarray | pl.DataFrame,
        *,
        return_type: ReturnType = "numpy",
        preprocess: Preprocess = "none",
        random_state: int | None = None,
        debug: bool = True,
        debug_mode: int = logging.INFO,
    ) -> None:
        """Initialize non-linear reduction utilities."""
        super().__init__(
            dataset=dataset,
            return_type=return_type,
            preprocess=preprocess,
            random_state=random_state,
            debug=debug,
            debug_mode=debug_mode,
            name_logging=NonLinearReductions.__name__,
        )
        self.__logger__.info("Initialized NonLinearReductions with dataset shape=%s", self.dataset.shape)

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
    ) -> np.ndarray | pl.DataFrame | None:
        try:
            params_getter = cast("Callable[[], object]", getattr(model, "get_params", dict))
            params = params_getter()
            self.__logger__.info("Applying %s with params=%s", method_name, params)
            fit_transform = cast("Callable[[np.ndarray], object]", getattr(model, "fit_transform"))  # noqa: B009
            transformed = np.asarray(fit_transform(self.dataset))
            k = (
                n_components
                if n_components is not None
                else getattr(model, "n_components", transformed.shape[1])
            )
            self.__logger__.info("%s successful. Output shape=%s", method_name, transformed.shape)
            return self.generate_dataset_post_reduction(transformed, k)
        except Exception as e:  # noqa: BLE001
            self.__logger__.error("%s failed: %s", method_name, e)
            self.__logger__.debug(traceback.format_exc())
            return None

    # -----------------------------
    # Public API (one wrapper per method)
    # -----------------------------
    def apply_tsne(self, **kwargs: object) -> np.ndarray | pl.DataFrame | None:
        """Apply t-SNE and return transformed data."""
        n = self.dataset.shape[0]
        perplexity = kwargs.get("perplexity", 30)
        if isinstance(perplexity, (int, float)) and perplexity >= max(1, n):
            new_p = max(5, min(30, max(1, n // 3)))
            self.__logger__.warning("t-SNE perplexity=%s >= n=%s; using %s instead.", perplexity, n, new_p)
            kwargs["perplexity"] = new_p
        model = self._init_with_seed(TSNE, kwargs)
        return self._apply_model(model, "t-SNE", cast("int | None", kwargs.get("n_components")))

    def apply_isomap(self, **kwargs: object) -> np.ndarray | pl.DataFrame | None:
        """Apply Isomap and return transformed data."""
        model = Isomap(**cast("dict[str, Any]", kwargs))  # no random_state
        return self._apply_model(model, "Isomap", cast("int | None", kwargs.get("n_components")))

    def apply_mds(self, **kwargs: object) -> np.ndarray | pl.DataFrame | None:
        """Apply MDS and return transformed data."""
        model = self._init_with_seed(MDS, kwargs)
        return self._apply_model(model, "MDS", cast("int | None", kwargs.get("n_components")))

    def apply_lle(self, **kwargs: object) -> np.ndarray | pl.DataFrame | None:
        """Apply LLE and return transformed data."""
        model = self._init_with_seed(LocallyLinearEmbedding, kwargs)
        return self._apply_model(model, "LLE", cast("int | None", kwargs.get("n_components")))

    def apply_spectral(self, **kwargs: object) -> np.ndarray | pl.DataFrame | None:
        """Apply SpectralEmbedding and return transformed data."""
        model = self._init_with_seed(SpectralEmbedding, kwargs)
        return self._apply_model(model, "SpectralEmbedding", cast("int | None", kwargs.get("n_components")))

    def apply_umap(self, **kwargs: object) -> np.ndarray | pl.DataFrame | None:
        """Apply UMAP and return transformed data."""
        # umap-learn supports 'random_state'
        model = self._init_with_seed(umap.UMAP, kwargs)
        return self._apply_model(model, "UMAP", cast("int | None", kwargs.get("n_components")))

    def apply_dictionary_learning(self, **kwargs: object) -> np.ndarray | pl.DataFrame | None:
        """Apply DictionaryLearning and return transformed data."""
        model = self._init_with_seed(DictionaryLearning, kwargs)
        return self._apply_model(model, "DictionaryLearning", cast("int | None", kwargs.get("n_components")))

    def apply_mini_batch_dictionary_learning(
        self,
        **kwargs: object,
    ) -> np.ndarray | pl.DataFrame | None:
        """Apply MiniBatchDictionaryLearning and return transformed data."""
        model = self._init_with_seed(MiniBatchDictionaryLearning, kwargs)
        return self._apply_model(
            model,
            "MiniBatchDictionaryLearning",
            cast("int | None", kwargs.get("n_components")),
        )

    def apply_dip_ext(self, **kwargs: object) -> np.ndarray | pl.DataFrame | None:
        """Apply DipExt and return transformed data."""
        model = DipExt(**cast("dict[str, Any]", kwargs))  # library-specific, no random_state in __init__
        return self._apply_model(model, "DipExt", cast("int | None", kwargs.get("n_components")))
