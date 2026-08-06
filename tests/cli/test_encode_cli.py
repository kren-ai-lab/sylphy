from __future__ import annotations

from pathlib import Path  # noqa: TC003

import polars as pl
import pytest
from typer.testing import CliRunner

from sylphy.cli.encode import app


def test_encode_onehot_csv_input(tmp_path: Path) -> None:
    """One-hot encode sequences from CSV, output to CSV."""
    df = pl.DataFrame({"sequence": ["ACD", "WYYVV", "KLMNP"]})
    inp = tmp_path / "seqs.csv"
    df.write_csv(inp)
    out = tmp_path / "onehot.csv"

    res = CliRunner().invoke(
        app,
        ["--method", "one_hot", "--input", str(inp), "--output", str(out), "--max-length", "5"],
    )
    assert res.exit_code == 0, res.stdout
    got = pl.read_csv(out)
    assert got.shape == (3, 5 * 20 + 1)  # max_length * 20 AAs + sequence column


def test_encode_ordinal_npy_output(tmp_path: Path) -> None:
    """Ordinal encode to npy format."""
    df = pl.DataFrame({"sequence": ["AAAA", "BBB"]})
    inp = tmp_path / "seq.csv"
    df.write_csv(inp)
    out = tmp_path / "ordinal.npy"

    res = CliRunner().invoke(
        app,
        ["--method", "ordinal", "--input", str(inp), "--output", str(out), "--max-length", "4"],
    )
    assert res.exit_code == 0, res.stdout
    assert out.exists()


def test_encode_parquet_input(tmp_path: Path) -> None:
    """Load sequences from parquet input."""
    df = pl.DataFrame({"sequence": ["ACD", "WYYVV"]})
    inp = tmp_path / "seqs.parquet"
    df.write_parquet(inp)
    out = tmp_path / "onehot.csv"

    res = CliRunner().invoke(
        app,
        ["--method", "one_hot", "--input", str(inp), "--output", str(out), "--max-length", "5"],
    )
    assert res.exit_code == 0, res.stdout
    assert pl.read_csv(out).shape[0] == 2


def test_encode_tsv_input(tmp_path: Path) -> None:
    """Load sequences from TSV input."""
    inp = tmp_path / "seqs.tsv"
    inp.write_text("sequence\nACD\nWYYVV\n", encoding="utf-8")
    out = tmp_path / "onehot.csv"

    res = CliRunner().invoke(
        app,
        ["--method", "one_hot", "--input", str(inp), "--output", str(out), "--max-length", "5"],
    )
    assert res.exit_code == 0, res.stdout
    assert pl.read_csv(out).shape[0] == 2


def test_encode_output_missing_extension_errors(tmp_path: Path) -> None:
    """Output path without extension should exit with error."""
    df = pl.DataFrame({"sequence": ["ACD"]})
    inp = tmp_path / "seqs.csv"
    df.write_csv(inp)

    res = CliRunner().invoke(
        app,
        ["--method", "one_hot", "--input", str(inp), "--output", str(tmp_path / "out")],
    )
    assert res.exit_code != 0


def test_encode_output_unsupported_extension_errors(tmp_path: Path) -> None:
    """Output path with unsupported extension should exit with error."""
    df = pl.DataFrame({"sequence": ["ACD"]})
    inp = tmp_path / "seqs.csv"
    df.write_csv(inp)

    res = CliRunner().invoke(
        app,
        ["--method", "one_hot", "--input", str(inp), "--output", str(tmp_path / "out.txt")],
    )
    assert res.exit_code != 0


def test_encode_custom_seq_col(tmp_path: Path) -> None:
    """--seq-col flag selects correct column."""
    df = pl.DataFrame({"prot": ["ACD", "WYYVV", "KLMNP"]})
    inp = tmp_path / "seqs.csv"
    df.write_csv(inp)
    out = tmp_path / "onehot.csv"

    res = CliRunner().invoke(
        app,
        [
            "--method",
            "one_hot",
            "--input",
            str(inp),
            "--output",
            str(out),
            "--seq-col",
            "prot",
            "--max-length",
            "5",
        ],
    )
    assert res.exit_code == 0, res.stdout


@pytest.mark.parametrize("method", ["one_hot", "ordinal", "frequency", "kmers"])
def test_encode_basic_methods(tmp_path: Path, method: str) -> None:
    """Smoke test each non-physchem method produces a file."""
    df = pl.DataFrame({"sequence": ["AACD", "WYYVV", "KLMNP"]})
    inp = tmp_path / "seqs.csv"
    df.write_csv(inp)
    out = tmp_path / f"{method}.csv"

    res = CliRunner().invoke(
        app,
        ["--method", method, "--input", str(inp), "--output", str(out), "--max-length", "5"],
    )
    assert res.exit_code == 0, res.stdout
    assert out.exists()
