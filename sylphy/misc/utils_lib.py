"""Provide utility helpers for sampling, distance, IDs, and exports."""

from __future__ import annotations

import datetime as dt
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.utils import shuffle

from sylphy.core.optional_dependencies import wrap_optional_dependency_error

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sylphy.types import FileFormat

_LOG = logging.getLogger("sylphy.misc.utils")
_MATRIX_NDIM = 2


class UtilsLib:
    """Utility collection for common data manipulation tasks used across the library.

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
        label_name: str | None = None,
        labels: Sequence[Any] | None = None,
        n_samples: int = 100,
        *,
        per_label: bool = True,
        replace: bool = False,
        random_state: int | None = 42,
    ) -> pd.DataFrame:
        """Randomly select rows from a DataFrame with optional stratification."""
        if label_name is None:
            _LOG.info("Sampling without stratification (n=%d).", n_samples)
            shuffled = cast("pd.DataFrame", shuffle(df, random_state=random_state))
            return shuffled.head(n_samples)

        return cls._stratified_selection(
            df,
            label_name=label_name,
            labels=labels,
            n_samples=n_samples,
            per_label=per_label,
            replace=replace,
            random_state=random_state,
        )

    @classmethod
    def _stratified_selection(
        cls,
        df: pd.DataFrame,
        label_name: str,
        labels: Sequence[Any] | None,
        n_samples: int,
        *,
        per_label: bool,
        replace: bool,
        random_state: int | None,
    ) -> pd.DataFrame:
        """Select rows using stratified sampling rules."""
        if label_name not in df.columns:
            _LOG.error("Label column '%s' not found.", label_name)
            msg = f"Label column '{label_name}' not found in DataFrame."
            raise ValueError(msg)

        # Determine the set of labels to consider
        present = cast("np.ndarray", pd.unique(df[label_name]))
        use_labels = cls._resolve_labels(present, labels, label_name)

        if per_label:
            return cls._sample_per_label(
                df, label_name, use_labels, n_samples, replace=replace, seed=random_state,
            )

        return cls._sample_globally_proportionate(
            df, label_name, use_labels, n_samples, replace=replace, seed=random_state,
        )

    @staticmethod
    def _resolve_labels(present: np.ndarray, requested: Sequence[Any] | None, col: str) -> list[Any]:
        if requested is None:
            return list(present)
        unknown = [lab for lab in requested if lab not in present]
        if unknown:
            _LOG.error("Unknown labels for '%s': %s", col, unknown)
            msg = f"The following labels were not found in column '{col}': {unknown}"
            raise ValueError(msg)
        return list(requested)

    @staticmethod
    def _sample_per_label(
        df: pd.DataFrame,
        col: str,
        labels: list[Any],
        n: int,
        *,
        replace: bool,
        seed: int | None,
    ) -> pd.DataFrame:
        parts: list[pd.DataFrame] = []
        for lab in labels:
            subset = df.loc[df[col] == lab]
            k = n if replace else min(n, len(subset))
            if k <= 0:
                _LOG.warning("Skipping label '%s' (no rows).", lab)
                continue
            parts.append(subset.sample(n=k, replace=replace, random_state=seed))
            _LOG.info("Sampled %d rows for label '%s'.", k, lab)

        if not parts:
            return pd.DataFrame(columns=df.columns)
        out = pd.concat(parts, axis=0).reset_index(drop=True)
        _LOG.info("Total sampled rows (per_label): %d", len(out))
        return out

    @staticmethod
    def _sample_globally_proportionate(
        df: pd.DataFrame,
        col: str,
        labels: list[Any],
        n: int,
        *,
        replace: bool,
        seed: int | None,
    ) -> pd.DataFrame:
        counts = df.loc[df[col].isin(labels), col].value_counts().sort_index()
        total = counts.sum()
        if total == 0:
            _LOG.warning("No rows matched labels; returning empty frame.")
            return df.iloc[0:0].copy()

        alloc = cast("pd.Series", (cast("Any", counts / total) * n).round().astype(int))
        alloc = alloc.mask((counts > 0) & (alloc == 0), 1)
        diff = int(n - int(alloc.sum()))
        if diff != 0:
            order = counts.sort_values(ascending=(diff < 0)).index
            for lab in order:
                if diff == 0:
                    break
                new_v = alloc[lab] + (1 if diff > 0 else -1)
                if new_v >= 0:
                    alloc[lab] = new_v
                    diff += -1 if diff > 0 else 1

        parts: list[pd.DataFrame] = []
        for lab, k in alloc.items():
            if k <= 0:
                continue
            subset = df.loc[df[col] == lab]
            actual_k = k if replace else min(k, len(subset))
            if actual_k > 0:
                parts.append(subset.sample(n=actual_k, replace=replace, random_state=seed))
                _LOG.info("Sampled %d rows for label '%s' (global mode).", actual_k, lab)

        if not parts:
            return pd.DataFrame(columns=df.columns)
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
        metric_params: dict[str, Any] | None = None,
        n_jobs: int | None = None,
    ) -> np.ndarray:
        """Compute pairwise distances between rows of a numeric matrix."""
        if not isinstance(matrix_data, np.ndarray):
            _LOG.error("Input must be a NumPy ndarray.")
            msg = "Input must be a NumPy ndarray."
            raise TypeError(msg)
        if matrix_data.ndim != _MATRIX_NDIM:
            msg = f"matrix_data must be 2D; got shape {matrix_data.shape}"
            raise ValueError(msg)

        supported_metrics = {
            "cityblock", "cosine", "euclidean", "l1", "l2", "manhattan",
            "nan_euclidean", "braycurtis", "canberra", "chebyshev",
            "correlation", "dice", "hamming", "jaccard", "kulsinski",
            "mahalanobis", "minkowski", "rogerstanimoto", "russellrao",
            "seuclidean", "sokalmichener", "sokalsneath", "sqeuclidean", "yule",
        }
        if metric not in supported_metrics:
            _LOG.error("Unsupported metric '%s'.", metric)
            msg = f"Unsupported metric '{metric}'. Must be one of: {sorted(supported_metrics)}"
            raise ValueError(msg)

        params: dict[str, Any] = dict(metric_params or {})
        if metric == "mahalanobis" and "VI" not in params:
            cov = np.cov(matrix_data, rowvar=False)
            try:
                VI = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                VI = np.linalg.inv(cov + np.eye(cov.shape[0]) * 1e-8)
            params["VI"] = VI

        if metric == "seuclidean" and "V" not in params:
            params["V"] = np.var(matrix_data, axis=0, ddof=1)

        if metric != "nan_euclidean" and not np.isfinite(matrix_data).all():
            msg = "matrix_data contains non-finite values; consider 'nan_euclidean' or clean the data."
            raise ValueError(msg)

        _LOG.info("Computing pairwise distances with metric '%s'.", metric)
        return pairwise_distances(matrix_data, metric=metric, n_jobs=n_jobs, **params)

    # -------------------------------------------------------------------------
    # IDs and filesystem helpers
    # -------------------------------------------------------------------------
    @classmethod
    def create_jobid(cls, prefix: str | None = None) -> str:
        """Create a unique job identifier."""
        now = dt.datetime.now(dt.UTC)
        ts = now.strftime("%Y%m%dT%H%M%S%fZ")
        jid = f"{ts}-{uuid.uuid4().hex[:12]}"
        if prefix:
            jid = f"{prefix}-{jid}"
        _LOG.info("Creating job_id %s", jid)
        return jid

    @classmethod
    def delete_folder(
        cls,
        path_to_folder: str | Path,
        *,
        missing_ok: bool = True,
        restrict_to: Path | None = None,
    ) -> bool:
        """Safely delete a folder tree with guardrails."""
        folder = Path(path_to_folder).expanduser().resolve()
        forbidden = {Path("/").resolve(), Path.home().resolve()}
        if os.name == "nt":
            forbidden.add(Path(os.environ.get("SYSTEMDRIVE", "C:") + "\\").resolve())

        if folder in forbidden or str(folder).strip() in {"", ".", ".."}:
            msg = f"Refusing to delete unsafe path: {folder}"
            raise ValueError(msg)

        if restrict_to is not None:
            base = Path(restrict_to).expanduser().resolve()
            try:
                folder.relative_to(base)
            except Exception as exc:
                msg = f"Refusing to delete outside restricted base: {base} (target: {folder})"
                raise ValueError(msg) from exc

        if not folder.exists():
            if missing_ok:
                return False
            msg = f"Folder not found: {folder}"
            raise ValueError(msg)

        if not folder.is_dir():
            msg = f"Not a directory: {folder}"
            raise ValueError(msg)

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
        path: str | Path,
        *,
        base_message: str = "Encoded data",
        file_format: FileFormat | None = None,
        overwrite: bool = True,
    ) -> Path:
        """Persist an encoded table to disk."""
        dest = Path(path).expanduser()
        if file_format is None:
            file_format = cls._infer_format(dest)

        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and not overwrite:
            msg = f"Destination exists: {dest}"
            raise FileExistsError(msg)

        if file_format not in {"csv", "npy", "npz", "parquet"}:
            msg = f"Unsupported file format '{file_format}'."
            raise ValueError(msg)

        try:
            cls._do_export(df_encoded, dest, file_format, base_message)
        except Exception as e:
            wrapped = wrap_optional_dependency_error(
                e, feature="Parquet export", extra="parquet", packages=("pyarrow", "fastparquet"),
            )
            if wrapped is not None:
                _LOG.error("Failed export to %s (%s): %s", dest, file_format, wrapped)
                raise wrapped from e
            _LOG.error("Failed export to %s (%s): %s", dest, file_format, e)
            raise

        return dest

    @staticmethod
    def _infer_format(dest: Path) -> FileFormat:
        s = dest.suffix.lower()
        if s in {".csv", ".npy", ".npz", ".parquet"}:
            return cast("FileFormat", s.lstrip("."))
        return "csv"

    @staticmethod
    def _do_export(df: pd.DataFrame, dest: Path, fmt: str, msg: str) -> None:
        if fmt == "csv":
            df.to_csv(dest, index=False)
        elif fmt == "npy":
            np.save(dest, df.to_numpy(), allow_pickle=True)
        elif fmt == "npz":
            np.savez_compressed(dest, values=df.to_numpy(), columns=df.columns.to_numpy())
        elif fmt == "parquet":
            df.to_parquet(dest, index=False)
        _LOG.info("%s exported to %s: %s", msg, fmt.upper(), dest)
