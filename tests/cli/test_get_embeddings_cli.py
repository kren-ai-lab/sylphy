from __future__ import annotations

import sys
from pathlib import Path  # noqa: TC003
from typing import Any, cast

import polars as pl
from typer.testing import CliRunner

from sylphy.cli.get_embeddings import app


def test_get_embeddings_runs_and_saves_csv(tmp_path: Path) -> None:
    """Verify CLI extracts embeddings and saves to CSV with correct shape."""
    df = pl.DataFrame({"sequence": ["AAAA", "BBB", "CCCCC"]})
    inp = tmp_path / "seqs.csv"
    df.write_csv(inp)
    out = tmp_path / "out.csv"

    result = CliRunner().invoke(
        app,
        [
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
        ],
    )
    assert result.exit_code == 0, result.output

    got = pl.read_csv(out)
    assert got.shape == (3, 5)  # 3 sequences, hidden_size=4 + sequence column
    assert "sequence" in got.columns


def test_get_embeddings_oom_backoff_succeeds(tmp_path: Path) -> None:
    """Verify that the CLI successfully processes sequences and calls the model."""
    transformers_mod = cast("Any", sys.modules["transformers"])
    _FakeModel = transformers_mod.AutoModel
    _FakeModel.OOM_THRESHOLD = None
    _FakeModel.FORWARD_CALLS = 0

    df = pl.DataFrame({"sequence": ["AAAA", "BBBB", "CCCCC", "DD"]})
    inp = tmp_path / "seqs.csv"
    df.write_csv(inp)
    out = tmp_path / "out.csv"

    result = CliRunner().invoke(
        app,
        [
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
        ],
    )
    assert result.exit_code == 0, result.output
    assert pl.read_csv(out).shape == (4, 5)
    assert _FakeModel.FORWARD_CALLS >= 1
