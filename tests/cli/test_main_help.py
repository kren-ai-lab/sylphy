# tests/cli/test_main_help.py
from __future__ import annotations

from typer.testing import CliRunner

from protein_representation.cli.main import app


def test_main_help_lists_subcommands():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    text = result.stdout
    assert "get-embedding" in text
    assert "encode-sequences" in text
    assert "reduce" in text
    assert "cache" in text
