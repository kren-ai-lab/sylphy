# tests/cli/test_main_help.py
from __future__ import annotations

from typer.testing import CliRunner

from sylphy import __version__
from sylphy.cli.main import app


def test_main_help_lists_subcommands() -> None:
    """Verify main CLI help displays all subcommands."""
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    text = result.stdout
    assert "embed" in text
    assert "encode" in text
    assert "cache" in text


def test_version_flag() -> None:
    """Verify --version flag displays version and exits."""
    runner = CliRunner()
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert f"sylphy {__version__}" in result.stdout
