"""sylphy validate-config — check a ``--config`` file without touching network, GPU or input data."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path  # noqa: TC003

import typer

from sylphy.cli._config_file import report_issues
from sylphy.validation import validate_config


class Target(StrEnum):
    """Command whose config shape is validated."""

    encode_cmd = "encode"
    embed_cmd = "embed"


def validate(
    target: Target = typer.Argument(..., help="Command the config is for."),  # noqa: B008
    config: Path = typer.Argument(..., help="YAML/JSON config file."),  # noqa: B008
    *,
    as_json: bool = typer.Option(False, "--json", help="Print problems as a JSON array on stdout."),  # noqa: FBT003
) -> None:
    """Validate CONFIG as accepted by `sylphy TARGET --config CONFIG`."""
    raise typer.Exit(report_issues(validate_config(config, target.value), as_json=as_json))
