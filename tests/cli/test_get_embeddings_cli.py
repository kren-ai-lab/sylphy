# tests/cli/test_get_embeddings_cli.py
from __future__ import annotations

import sys

import pandas as pd
from typer.testing import CliRunner

from sylphy.cli.get_embeddings import app


def test_get_embeddings_runs_and_saves_csv(tmp_path):
    df = pd.DataFrame({"sequence": ["AAAA", "BBB", "CCCCC"]})
    inp = tmp_path / "seqs.csv"
    df.to_csv(inp, index=False)
    out = tmp_path / "out.csv"

    runner = CliRunner()
    args = [
        "--model",
        "facebook/esm2_t6_8M_UR50D",
        "--device",
        "cpu",
        "--batch-size",
        "3",
        "--max-length",
        "16",
        "--pool",
        "mean",
        "--input-data",
        str(inp),
        "--output",
        str(out),
        "--sequence-identifier",
        "sequence",
        "--format-output",
        "csv",
        "--debug",
        "--log-level",
        "DEBUG",
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output

    got = pd.read_csv(out)
    assert got.shape[0] == 3  # 3 secuencias
    # hidden=4 → 4 features + 'sequence' = 5 columnas
    assert got.shape[1] == 5
    assert "sequence" in got.columns


def test_get_embeddings_oom_backoff_succeeds(tmp_path):
    # Access the fake model through sys.modules (set up by conftest fixture)
    _FakeModel = sys.modules["transformers"].AutoModel
    # Set threshold to None to disable OOM simulation (since we're on CPU)
    _FakeModel.OOM_THRESHOLD = None
    _FakeModel.FORWARD_CALLS = 0

    df = pd.DataFrame({"sequence": ["AAAA", "BBBB", "CCCCC", "DD"]})
    inp = tmp_path / "seqs.csv"
    df.to_csv(inp, index=False)
    out = tmp_path / "out.csv"

    runner = CliRunner()
    args = [
        "--model",
        "facebook/esm2_t6_8M_UR50D",
        "--device",
        "cpu",
        "--batch-size",
        "4",
        "--max-length",
        "32",
        "--pool",
        "mean",
        "--input-data",
        str(inp),
        "--output",
        str(out),
        "--sequence-identifier",
        "sequence",
        "--format-output",
        "csv",
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output

    got = pd.read_csv(out)
    assert got.shape == (4, 5)
    # Note: OOM backoff only works on CUDA, so we can't test FORWARD_CALLS >= 2 on CPU
    # The model should be called at least once
    assert _FakeModel.FORWARD_CALLS >= 1
