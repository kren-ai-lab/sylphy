from __future__ import annotations

import sys
from pathlib import Path  # noqa: TC003
from typing import Any, cast

import polars as pl
from typer.testing import CliRunner

from sylphy.cli.embed import app


def test_embed_runs_and_saves_csv(tmp_path: Path) -> None:
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
            "--input",
            str(inp),
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output

    got = pl.read_csv(out)
    assert got.shape == (3, 5)  # 3 sequences, hidden_size=4 + sequence column
    assert "sequence" in got.columns


def test_embed_oom_backoff_succeeds(tmp_path: Path) -> None:
    """Verify CLI processes sequences with OOM backoff enabled."""
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
            "--input",
            str(inp),
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert pl.read_csv(out).shape == (4, 5)
    assert _FakeModel.FORWARD_CALLS >= 1


def test_embed_parquet_input(tmp_path: Path) -> None:
    """Load sequences from parquet input file."""
    df = pl.DataFrame({"sequence": ["AAAA", "BBB"]})
    inp = tmp_path / "seqs.parquet"
    df.write_parquet(inp)
    out = tmp_path / "out.csv"

    result = CliRunner().invoke(
        app,
        [
            "--model",
            "facebook/esm2_t6_8M_UR50D",
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--input",
            str(inp),
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert pl.read_csv(out).shape[0] == 2


def test_embed_parquet_output(tmp_path: Path) -> None:
    """Save embeddings to parquet format."""
    df = pl.DataFrame({"sequence": ["AAAA", "BBB"]})
    inp = tmp_path / "seqs.csv"
    df.write_csv(inp)
    out = tmp_path / "out.parquet"

    result = CliRunner().invoke(
        app,
        [
            "--model",
            "facebook/esm2_t6_8M_UR50D",
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--input",
            str(inp),
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert pl.read_parquet(out).shape[0] == 2


def test_embed_output_missing_extension_errors(tmp_path: Path) -> None:
    """Output path without extension should exit with error."""
    df = pl.DataFrame({"sequence": ["AAAA"]})
    inp = tmp_path / "seqs.csv"
    df.write_csv(inp)

    result = CliRunner().invoke(
        app,
        [
            "--model",
            "facebook/esm2_t6_8M_UR50D",
            "--input",
            str(inp),
            "--output",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code != 0


def test_embed_output_unsupported_extension_errors(tmp_path: Path) -> None:
    """Output path with unsupported extension should exit with error."""
    df = pl.DataFrame({"sequence": ["AAAA"]})
    inp = tmp_path / "seqs.csv"
    df.write_csv(inp)

    result = CliRunner().invoke(
        app,
        [
            "--model",
            "facebook/esm2_t6_8M_UR50D",
            "--input",
            str(inp),
            "--output",
            str(tmp_path / "out.txt"),
        ],
    )
    assert result.exit_code != 0


def test_embed_custom_seq_col(tmp_path: Path) -> None:
    """--seq-col selects the correct column."""
    df = pl.DataFrame({"prot": ["AAAA", "BBB", "CCCCC"]})
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
            "--input",
            str(inp),
            "--output",
            str(out),
            "--seq-col",
            "prot",
        ],
    )
    assert result.exit_code == 0, result.output
