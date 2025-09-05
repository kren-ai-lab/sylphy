# tests/cli/test_reduce_cli.py
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from typer.testing import CliRunner

from sylphy.cli.reduce import app


def test_reduce_pca_numpy_roundtrip(tmp_path):
    X = np.random.RandomState(0).randn(12, 6).astype("float32")
    inp = tmp_path / "X.npy"
    np.save(inp, X)
    out = tmp_path / "X_pca.npy"

    runner = CliRunner()
    args = [
        "run",
        "--input", str(inp),
        "--out", str(out),
        "--method", "pca",
        "--n-components", "2",
        "--return-type", "numpy",
        "--debug",
        "--log-level", "INFO",
    ]
    res = runner.invoke(app, args)
    assert res.exit_code == 0, res.stdout
    Y = np.load(out)
    assert Y.shape == (12, 2)


def test_reduce_isomap_pandas_from_csv(tmp_path):
    df = pd.DataFrame(np.random.RandomState(1).randn(10, 5).astype("float32"), columns=[f"f{i}" for i in range(5)])
    inp = tmp_path / "X.csv"
    df.to_csv(inp, index=False)
    out = tmp_path / "X_iso.csv"

    runner = CliRunner()
    args = [
        "run",
        "--input", str(inp),
        "--out", str(out),
        "--method", "isomap",
        "--n-components", "2",
        "--return-type", "pandas",
        "--columns", "f0,f1,f2,f3,f4",
        "--debug",
        "--log-level", "INFO",
    ]
    res = runner.invoke(app, args)
    assert res.exit_code == 0, res.stdout

    got = pd.read_csv(out)
    assert got.shape == (10, 2)  # p_1, p_2


def test_reduce_params_from_json_string(tmp_path):
    X = np.random.RandomState(2).randn(8, 4).astype("float32")
    inp = tmp_path / "X.npy"
    np.save(inp, X)
    out = tmp_path / "X_svd.npy"

    params = json.dumps({"n_components": 3, "random_state": 0})
    runner = CliRunner()
    args = [
        "run",
        "--input", str(inp),
        "--out", str(out),
        "--method", "truncated_svd",
        "--params", params,
        "--return-type", "numpy",
    ]
    res = runner.invoke(app, args)
    assert res.exit_code == 0, res.stdout
    Y = np.load(out)
    assert Y.shape == (8, 3)
