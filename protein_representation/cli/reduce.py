# protein_representation/cli/reduce.py
"""
Dimensionality reduction for embedding matrices.

Features
--------
- Load from NPY/NPZ/CSV/TSV/JSON(L)
- Optional standardization (StandardScaler)
- Linear & non-linear methods via central factory
- Method kwargs via --params JSON or --params-file
- Save reduced outputs as .npy or .csv
- (Linear only) Save fitted model with --save-model
- List available methods with --list-methods
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import typer

from protein_representation.reductions import (
    reduce_dimensionality,
    get_available_methods,
    is_linear_method,
)

try:
    from rich.console import Console
    from rich.table import Table
    from rich.theme import Theme
except Exception:
    Console = None  # type: ignore

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None  # type: ignore

try:
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover
    StandardScaler = None  # type: ignore

app = typer.Typer(name="reduce", help="Dimensionality reduction for embedding matrices.", no_args_is_help=True)

def _console(enabled: bool) -> Optional[Console]:  # type: ignore[name-defined]
    if not enabled or Console is None:
        return None
    theme = Theme({"info": "cyan", "ok": "green", "warn": "yellow", "err": "red"})
    return Console(theme=theme, stderr=False)

def _infer_sep(path: Path) -> str:
    return "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","

def _load_matrix(input_path: Path, npz_key: Optional[str], columns: Optional[str], con: Optional[Console]) -> np.ndarray:
    suf = input_path.suffix.lower()
    if suf == ".npy":
        X = np.load(input_path)
        con and con.print(f"[info]Loaded NPY with shape {X.shape}")
        return X
    if suf == ".npz":
        data = np.load(input_path)
        keys = list(data.keys())
        key = npz_key or (keys[0] if len(keys) == 1 else None)
        if key is None:
            raise ValueError(f"NPZ has multiple arrays {keys}; specify --npz-key.")
        X = data[key]
        con and con.print(f"[info]Loaded NPZ[{key}] with shape {X.shape}")
        return X
    if suf in {".csv", ".tsv", ".tab", ".jsonl", ".json"}:
        if suf in {".csv", ".tsv", ".tab"}:
            df = pd.read_csv(input_path, sep=_infer_sep(input_path))
        else:
            df = pd.read_json(input_path, lines=(suf == ".jsonl"))
        if columns:
            cols = [c.strip() for c in columns.split(",") if c.strip()]
            missing = [c for c in cols if c not in df.columns]
            if missing:
                raise ValueError(f"Columns not found: {missing}")
            mat = df[cols].values
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found. Use --columns to select.")
            mat = df[numeric_cols].values
        con and con.print(f"[info]Loaded table with shape {mat.shape}")
        return mat
    raise ValueError(f"Unsupported input format: {suf}")

def _level_from_str(name: str) -> int:
    name = (name or "").strip().upper()
    mapping = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return mapping.get(name, logging.DEBUG)

def _parse_params(params: Optional[str], params_file: Optional[Path]) -> Dict:
    cfg: Dict = {}
    if params_file is not None:
        cfg.update(json.loads(Path(params_file).read_text(encoding="utf-8")))
    if params:
        cfg.update(json.loads(params))
    return cfg

def _print_methods_table(con: Optional[Console]) -> None:
    methods_all = get_available_methods()
    methods_lin = set(get_available_methods("linear"))
    methods_non = set(get_available_methods("nonlinear"))
    if con:
        from rich.table import Table
        table = Table(title="Available reduction methods")
        table.add_column("Method")
        table.add_column("Kind")
        for m in methods_all:
            kind = "linear" if m in methods_lin else ("nonlinear" if m in methods_non else "unknown")
            table.add_row(m, kind)
        con.print(table)
    else:
        print("Available methods:")
        for m in methods_all:
            kind = "linear" if m in methods_lin else ("nonlinear" if m in methods_non else "unknown")
            print(f"- {m:30s} [{kind}]")

@app.command()
def run(
    input: Path = typer.Option(..., "--input", "-i", help="Input NPY/NPZ/CSV/TSV/JSON(L)."),
    out: Path = typer.Option(..., "--out", "-o", help="Output (.npy for numpy, .csv for pandas)."),
    method: str = typer.Option(..., "--method", "-m", help="Reduction method (use --list-methods to see options)."),
    n_components: Optional[int] = typer.Option(None, "--n-components", help="Target dimensionality."),
    params: Optional[str] = typer.Option(None, "--params", help="JSON string with method kwargs."),
    params_file: Optional[Path] = typer.Option(None, "--params-file", help="Path to JSON file with method kwargs."),
    npz_key: Optional[str] = typer.Option(None, "--npz-key", help="Key to load from .npz file."),
    columns: Optional[str] = typer.Option(None, "--columns", help="Comma-separated column names for tabular input."),
    standardize: bool = typer.Option(False, "--standardize/--no-standardize", help="Apply StandardScaler before reduction."),
    return_type: str = typer.Option("numpy", "--return-type", help="numpy|pandas"),
    rich: bool = typer.Option(True, "--rich/--no-rich", help="Rich console output."),
    debug: bool = typer.Option(True, "--debug/--no-debug", help="Enable component loggers."),
    log_level: str = typer.Option("DEBUG", "--log-level", help="DEBUG|INFO|WARNING|ERROR|CRITICAL."),
    save_model: Optional[Path] = typer.Option(None, "--save-model", help="For linear methods, save fitted estimator (joblib .pkl)."),
    list_methods: bool = typer.Option(False, "--list-methods", help="List available methods and exit."),
) -> None:
    con = _console(rich)
    if list_methods:
        _print_methods_table(con)
        raise typer.Exit(code=0)

    X = _load_matrix(input, npz_key=npz_key, columns=columns, con=con)
    X = np.asarray(X, dtype=np.float32)

    if standardize:
        if StandardScaler is None:
            raise RuntimeError("scikit-learn is required for --standardize")
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        con and con.print("[info]Applied StandardScaler to input matrix.")

    kwargs = _parse_params(params, params_file)
    if n_components is not None and "n_components" not in kwargs:
        kwargs["n_components"] = int(n_components)

    out.parent.mkdir(parents=True, exist_ok=True)
    model, reduced = reduce_dimensionality(
        method=method.strip().lower(),
        dataset=X,
        return_type=return_type,
        debug=debug,
        debug_mode=_level_from_str(log_level),
        **kwargs,
    )
    if reduced is None:
        con and con.print("[err]Reduction failed; no output produced.")
        raise typer.Exit(code=1)

    if save_model is not None:
        if not is_linear_method(method):
            raise typer.BadParameter("--save-model is only valid for linear methods.")
        if joblib is None:
            raise RuntimeError("joblib is required for --save-model")
        if model is None:
            raise RuntimeError("Internal error: linear method returned no model.")
        joblib.dump(model, save_model)
        con and con.print(f"[ok]Saved model to {save_model}")

    if return_type.lower() == "pandas":
        if out.suffix.lower() != ".csv":
            out = out.with_suffix(".csv")
        import pandas as pd
        assert isinstance(reduced, pd.DataFrame)
        reduced.to_csv(out, index=False)
        con and con.print(f"[ok]Saved CSV to {out} with shape {reduced.shape}")
    else:
        if out.suffix.lower() != ".npy":
            out = out.with_suffix(".npy")
        arr = np.asarray(reduced)
        np.save(out, arr)
        con and con.print(f"[ok]Saved NPY to {out} with shape {arr.shape}")

if __name__ == "__main__":
    app()
