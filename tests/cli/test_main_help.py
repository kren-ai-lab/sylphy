# tests/cli/test_main_help.py
from __future__ import annotations

from typer.testing import CliRunner

from sylphy import __version__
from sylphy.cli.main import app


def test_main_help_lists_subcommands():
    """Verify main CLI help displays all subcommands."""
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    text = result.stdout
    assert "get-embedding" in text
    assert "encode-sequences" in text
    assert "cache" in text


def test_version_flag():
    """Verify --version flag displays version and exits."""
    runner = CliRunner()
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert f"sylphy version {__version__}" in result.stdout


def test_version_short_flag():
    """Verify -v flag displays version and exits."""
    runner = CliRunner()
    result = runner.invoke(app, ["-v"])
    assert result.exit_code == 0
    assert f"sylphy version {__version__}" in result.stdout
