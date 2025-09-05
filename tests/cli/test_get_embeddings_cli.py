# tests/cli/test_get_embeddings_cli.py
from __future__ import annotations

import pandas as pd
from typer.testing import CliRunner

from sylphy.cli.get_embeddings import app
from transformers import AutoModel as _FakeModel


def test_get_embeddings_runs_and_saves_csv(tmp_path):
    df = pd.DataFrame({"sequence": ["AAAA", "BBB", "CCCCC"]})
    inp = tmp_path / "seqs.csv"
    df.to_csv(inp, index=False)
    out = tmp_path / "out.csv"

    runner = CliRunner()
    args = [
        "run",
        "--model", "facebook/esm2_t6_8M_UR50D",
        "--device", "cpu",
        "--batch-size", "3",
        "--max-length", "16",
        "--pool", "mean",
        "--input-data", str(inp),
        "--output", str(out),
        "--sequence-identifier", "sequence",
        "--format-output", "csv",
        "--debug",
        "--log-level", "DEBUG",
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.stdout

    got = pd.read_csv(out)
    assert got.shape[0] == 3  # 3 secuencias
    # hidden=4 → 4 features + 'sequence' = 5 columnas
    assert got.shape[1] == 5
    assert "sequence" in got.columns


def test_get_embeddings_oom_backoff_succeeds(tmp_path, monkeypatch):
    _FakeModel.OOM_THRESHOLD = 2
    _FakeModel.FORWARD_CALLS = 0

    import pandas as pd
    df = pd.DataFrame({"sequence": ["AAAA", "BBBB", "CCCCC", "DD"]})
    inp = tmp_path / "seqs.csv"
    df.to_csv(inp, index=False)
    out = tmp_path / "out.csv"

    runner = CliRunner()
    args = [
        "run",
        "--model", "facebook/esm2_t6_8M_UR50D",
        "--device", "cpu",
        "--batch-size", "4", 
        "--max-length", "32",
        "--pool", "mean",
        "--input-data", str(inp),
        "--output", str(out),
        "--sequence-identifier", "sequence",
        "--format-output", "csv",
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.stdout

    got = pd.read_csv(out)
    assert got.shape == (4, 5)
    assert _FakeModel.FORWARD_CALLS >= 2  
