"""sylphy validate-config — check a ``--config`` file without touching network, GPU or input data."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path  # noqa: TC003

import typer
import typer.main

from sylphy.cli._config_file import check_config, read_config, report_issues
from sylphy.cli._shared import CONFIG_EXCLUDE


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
    if target is Target.encode_cmd:
        from sylphy.cli.encode import app  # noqa: PLC0415
    else:
        from sylphy.cli.embed import app  # noqa: PLC0415

    raw, issues = read_config(config)
    issues = issues or check_config(typer.main.get_command(app), raw, exclude=CONFIG_EXCLUDE)
    raise typer.Exit(report_issues(issues, as_json=as_json))
