# utils_lib.py
from __future__ import annotations

import datetime as dt
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.utils import shuffle

_LOG = logging.getLogger("sylphy.misc.utils")


class UtilsLib:
    """
    Utility collection for common data manipulation tasks used across the library.

    Features
    --------
    • Reproducible random/stratified sampling (per-label or globally proportionate).
    • Pairwise distance estimation with support for metric parameters.
    • Safe job-id creation (UTC) with optional prefix.
    • Safe folder deletion with guardrails and optional subtree restriction.
    • Data export helpers for CSV/NPY/NPZ/Parquet.

    Notes
    -----
    The methods are class methods to allow use without instantiation and to facilitate testing.
    """

    # -------------------------------------------------------------------------
    # Sampling
    # -------------------------------------------------------------------------
    @classmethod
    def random_selection(
        cls,
        df: pd.DataFrame,
        label_name: Optional[str] = None,
        labels: Optional[Sequence[Any]] = None,
        n_samples: int = 100,
        *,
        per_label: bool = True,
        replace: bool = False,
        random_state: Optional[int] = 42,
    ) -> pd.DataFrame:
        """
        Randomly select rows from a DataFrame with optional stratification.

        Parameters
        ----------
        df : pd.DataFrame
            Input table.
        label_name : str, optional
            Column used for stratification. If None, plain random sampling is performed.
        labels : Sequence, optional
            Subset of label values to include. If None and label_name is provided, all
            unique labels present in `df[label_name]` are used.
        n_samples : int, default=100
            Number of rows to sample. If `per_label=True`, this is "n per label".
        per_label : bool, default=True
            If True, take `n_samples` *per label*. If False, take `n_samples` *in total*
            proportionally to each label frequency (rounded; at least 1 per present label).
        replace : bool, default=False
            Whether to sample with replacement.
        random_state : int or None, default=42
            Random seed (or None for non-deterministic).

        Returns
        -------
        pd.DataFrame
            The sampled rows (index reset).

        Raises
        ------
        ValueError
            If `label_name` is provided but not found in `df`.
            If `labels` contains values not present in `df[label_name]`.
        """
        if label_name is None:
            _LOG.info("Sampling without stratification (n=%d).", n_samples)
            return shuffle(df, random_state=random_state).head(n_samples)

        if label_name not in df.columns:
            _LOG.error("Label column '%s' not found.", label_name)
            raise ValueError(f"Label column '{label_name}' not found in DataFrame.")

        # Determine the set of labels to consider
        present = pd.unique(df[label_name])
        if labels is None:
            use_labels = list(present)
        else:
            unknown = [lab for lab in labels if lab not in present]
            if unknown:
                _LOG.error("Unknown labels for '%s': %s", label_name, unknown)
                raise ValueError(f"The following labels were not found in column '{label_name}': {unknown}")
            use_labels = list(labels)

        # Per-label sampling
        if per_label:
            parts = []
            for lab in use_labels:
                subset = df[df[label_name] == lab]
                k = n_samples if replace else min(n_samples, len(subset))
                if k == 0:
                    _LOG.warning("Skipping label '%s' (no rows).", lab)
                    continue
                parts.append(subset.sample(n=k, replace=replace, random_state=random_state))
                _LOG.info("Sampled %d rows for label '%s'.", k, lab)
            out = pd.concat(parts, axis=0).reset_index(drop=True)
            _LOG.info("Total sampled rows (per_label): %d", len(out))
            return out

        # Global proportional sampling
        # Compute desired counts per label proportional to their frequency
        counts = df[df[label_name].isin(use_labels)][label_name].value_counts().sort_index()
        total = counts.sum()
        if total == 0:
            _LOG.warning("No rows matched the requested labels; returning empty frame.")
            return df.iloc[0:0].copy()

        # Initial allocation
        alloc = (counts / total * n_samples).round().astype(int)
        # Ensure at least 1 for present labels when possible
        alloc = alloc.mask((counts > 0) & (alloc == 0), 1)
        # Adjust to match exactly n_samples
        diff = int(n_samples - int(alloc.sum()))
        if diff != 0:
            # Distribute the difference starting from the largest groups (or smallest)
            order = counts.sort_values(ascending=(diff < 0)).index
            for lab in order:
                if diff == 0:
                    break
                new_v = alloc[lab] + (1 if diff > 0 else -1)
                if new_v >= 0:
                    alloc[lab] = new_v
                    diff += -1 if diff > 0 else 1

        parts = []
        for lab, k in alloc.items():
            if k <= 0:
                continue
            subset = df[df[label_name] == lab]
            k = k if replace else min(k, len(subset))
            if k <= 0:
                continue
            parts.append(subset.sample(n=k, replace=replace, random_state=random_state))
            _LOG.info("Sampled %d rows for label '%s' (global mode).", k, lab)

        out = pd.concat(parts, axis=0).reset_index(drop=True)
        _LOG.info("Total sampled rows (global): %d", len(out))
        return out

    # -------------------------------------------------------------------------
    # Distances
    # -------------------------------------------------------------------------
    @classmethod
    def estimated_distance(
        cls,
        matrix_data: np.ndarray,
        metric: Literal[
            "cityblock",
            "cosine",
            "euclidean",
            "l1",
            "l2",
            "manhattan",
            "nan_euclidean",
            "braycurtis",
            "canberra",
            "chebyshev",
            "correlation",
            "dice",
            "hamming",
            "jaccard",
            "kulsinski",
            "mahalanobis",
            "minkowski",
            "rogerstanimoto",
            "russellrao",
            "seuclidean",
            "sokalmichener",
            "sokalsneath",
            "sqeuclidean",
            "yule",
        ] = "euclidean",
        *,
        metric_params: Optional[Dict[str, Any]] = None,
        n_jobs: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute pairwise distances between rows of a numeric matrix.

        Parameters
        ----------
        matrix_data : np.ndarray
            2D numeric matrix with shape (n_samples, n_features).
        metric : str, default='euclidean'
            Distance metric to use (as supported by sklearn.metrics.pairwise_distances).
        metric_params : dict, optional
            Additional keyword parameters for the distance metric, e.g.:
              • mahalanobis: VI=np.linalg.inv(np.cov(X, rowvar=False))
              • seuclidean : V=np.var(X, axis=0, ddof=1)
        n_jobs : int, optional
            Number of parallel jobs (if supported by the backend).

        Returns
        -------
        np.ndarray
            Square matrix of pairwise distances (shape: n_samples × n_samples).

        Raises
        ------
        TypeError
            If `matrix_data` is not a NumPy ndarray.
        ValueError
            If input is not 2D, contains non-finite values (when required),
            or if `metric` is unsupported.
        """
        if not isinstance(matrix_data, np.ndarray):
            _LOG.error("Input must be a NumPy ndarray.")
            raise TypeError("Input must be a NumPy ndarray.")
        if matrix_data.ndim != 2:
            raise ValueError(f"matrix_data must be 2D; got shape {matrix_data.shape}")

        supported_metrics = {
            "cityblock",
            "cosine",
            "euclidean",
            "l1",
            "l2",
            "manhattan",
            "nan_euclidean",
            "braycurtis",
            "canberra",
            "chebyshev",
            "correlation",
            "dice",
            "hamming",
            "jaccard",
            "kulsinski",
            "mahalanobis",
            "minkowski",
            "rogerstanimoto",
            "russellrao",
            "seuclidean",
            "sokalmichener",
            "sokalsneath",
            "sqeuclidean",
            "yule",
        }
        if metric not in supported_metrics:
            _LOG.error("Unsupported metric '%s'.", metric)
            raise ValueError(f"Unsupported metric '{metric}'. Must be one of: {sorted(supported_metrics)}")

        params: Dict[str, Any] = dict(metric_params or {})

        # Provide sensible defaults for metrics that require parameters.
        if metric == "mahalanobis" and "VI" not in params:
            # Compute inverse covariance across features (rowvar=False)
            cov = np.cov(matrix_data, rowvar=False)
            try:
                VI = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                # Regularize diagonal in near-singular cases
                eps = 1e-8
                VI = np.linalg.inv(cov + np.eye(cov.shape[0]) * eps)
            params["VI"] = VI

        if metric == "seuclidean" and "V" not in params:
            # Per-feature variance (ddof=1) for standardized Euclidean
            params["V"] = np.var(matrix_data, axis=0, ddof=1)

        # Most metrics expect finite values (except specialized ones like nan_euclidean).
        if metric != "nan_euclidean" and not np.isfinite(matrix_data).all():
            raise ValueError(
                "matrix_data contains non-finite values; consider 'nan_euclidean' or clean the data."
            )

        _LOG.info("Computing pairwise distances with metric '%s'.", metric)
        return pairwise_distances(matrix_data, metric=metric, n_jobs=n_jobs, **params)

    # -------------------------------------------------------------------------
    # IDs and filesystem helpers
    # -------------------------------------------------------------------------
    @classmethod
    def create_jobid(cls, prefix: Optional[str] = None) -> str:
        """
        Create a unique job identifier composed of an UTC timestamp and a shortened UUID.

        Parameters
        ----------
        prefix : str, optional
            Optional string prefix (e.g., a component name).

        Returns
        -------
        str
            Identifier like: '20250101T120305123456Z-abcdef123456' or with prefix
            'encoder-20250101T120305123456Z-abcdef123456'.
        """
        now = dt.datetime.now(dt.timezone.utc)
        ts = now.strftime("%Y%m%dT%H%M%S%fZ")
        jid = f"{ts}-{uuid.uuid4().hex[:12]}"
        if prefix:
            jid = f"{prefix}-{jid}"
        _LOG.info("Creating job_id %s", jid)
        return jid

    @classmethod
    def delete_folder(
        cls,
        path_to_folder: Union[str, Path],
        *,
        missing_ok: bool = True,
        restrict_to: Optional[Path] = None,
    ) -> bool:
        """
        Safely delete a folder tree with guardrails.

        Parameters
        ----------
        path_to_folder : str | Path
            Directory to remove recursively.
        missing_ok : bool, default=True
            If True, return False when the folder does not exist instead of raising.
        restrict_to : Path, optional
            If provided, the folder must be within this directory (resolved). This
            is useful to restrict deletions to cache/temp roots.

        Returns
        -------
        bool
            True if the folder was removed; False if it did not exist.

        Raises
        ------
        ValueError
            If the target path is unsafe (e.g., root) or outside `restrict_to`.
        """
        folder = Path(path_to_folder).expanduser().resolve()

        # Basic safety: do not allow dangerous targets
        forbidden = {Path("/").resolve(), Path.home().resolve()}
        if os.name == "nt":
            # best-effort Windows safeguards
            forbidden.add(Path(os.environ.get("SystemDrive", "C:") + "\\").resolve())

        if folder in forbidden or str(folder).strip() in {"", ".", ".."}:
            raise ValueError(f"Refusing to delete unsafe path: {folder}")

        if restrict_to is not None:
            base = Path(restrict_to).expanduser().resolve()
            try:
                folder.relative_to(base)
            except Exception as exc:
                raise ValueError(
                    f"Refusing to delete outside of restricted base: {base} (target: {folder})"
                ) from exc

        if not folder.exists():
            if missing_ok:
                _LOG.info("Folder does not exist, nothing to delete: %s", folder)
                return False
            raise ValueError(f"Folder not found: {folder}")

        if not folder.is_dir():
            raise ValueError(f"Not a directory: {folder}")

        shutil.rmtree(folder)
        _LOG.info("Deleted folder: %s", folder)
        return True

    # -------------------------------------------------------------------------
    # Export helpers
    # -------------------------------------------------------------------------
    @classmethod
    def export_data(
        cls,
        df_encoded: pd.DataFrame,
        path: Union[str, Path],
        *,
        base_message: str = "Encoded data",
        file_format: Optional[Literal["csv", "npy", "npz", "parquet"]] = None,
        overwrite: bool = True,
    ) -> Path:
        """
        Persist an encoded table to disk.

        Parameters
        ----------
        df_encoded : pd.DataFrame
            Data to export.
        path : str | Path
            Destination path. If `file_format` is None, the format is inferred from suffix.
        base_message : str, default="Encoded data"
            Human-readable label for logging messages.
        file_format : {"csv","npy","npz","parquet"} or None, optional
            Output format. If None, inferred from file suffix; default CSV if no suffix.
        overwrite : bool, default=True
            If False and the target exists, raise a FileExistsError.

        Returns
        -------
        Path
            The actual file path written.

        Raises
        ------
        ValueError
            If the format is unsupported.
        FileExistsError
            If `overwrite=False` and destination exists.
        """
        dest = Path(path).expanduser()
        suffix = dest.suffix.lower()
        if file_format is None:
            if suffix in {".csv", ".npy", ".npz", ".parquet"}:
                file_format = suffix.lstrip(".")  # type: ignore[assignment]
            else:
                file_format = "csv"

        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and not overwrite:
            raise FileExistsError(f"Destination exists: {dest}")

        try:
            if file_format == "csv":
                df_encoded.to_csv(dest, index=False)
                _LOG.info("%s exported to CSV: %s", base_message, dest)

            elif file_format == "npy":
                np.save(dest, df_encoded.values, allow_pickle=True)
                _LOG.info("%s exported to NPY: %s", base_message, dest)

            elif file_format == "npz":
                # Save values plus column names for round-trip friendliness
                np.savez_compressed(dest, values=df_encoded.values, columns=df_encoded.columns.to_numpy())
                _LOG.info("%s exported to NPZ: %s", base_message, dest)

            elif file_format == "parquet":
                # Requires pyarrow or fastparquet installed
                df_encoded.to_parquet(dest, index=False)
                _LOG.info("%s exported to Parquet: %s", base_message, dest)

            else:
                raise ValueError(f"Unsupported file format '{file_format}'.")
        except Exception as e:
            _LOG.error("Failed to export %s to %s (%s): %s", base_message, dest, file_format, e)
            raise

        return dest

    def get_cache_dir() -> Path:
        env = os.getenv("SYLPHY_CACHE_DIR")
        if env:
            return Path(env).expanduser()

        try:
            from sylphy._siteconfig import CACHE_DIR

            if CACHE_DIR:
                return Path(CACHE_DIR).expanduser()
        except Exception:
            pass

        try:
            from platformdirs import user_cache_dir

            base = Path(user_cache_dir("sylphy"))
        except Exception:
            base = Path.home() / ".cache" / "sylphy"
        return base
