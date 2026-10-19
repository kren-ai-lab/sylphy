"""Public config validation: the check behind ``sylphy validate-config``, usable from Python."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import typer.main

from sylphy.cli._config_file import ConfigIssue, check_config, read_config

if TYPE_CHECKING:
    import os

__all__ = ["ConfigIssue", "validate_config"]


def validate_config(
    config: Mapping[str, Any] | str | os.PathLike[str], target: Literal["encode", "embed"]
) -> list[ConfigIssue]:
    """Check ``config`` as accepted by ``sylphy TARGET --config``; empty when valid.

    ``config`` is a mapping or a path to a YAML/JSON file. Problems (including a missing or unparsable
    file, reported with path ``""``) are returned, never raised.
    """
    from sylphy.cli._shared import CONFIG_EXCLUDE  # noqa: PLC0415

    if target == "encode":
        from sylphy.cli.encode import app  # noqa: PLC0415
    elif target == "embed":
        from sylphy.cli.embed import app  # noqa: PLC0415
    else:
        return [ConfigIssue("", f"unknown target {target!r}, expected 'encode' or 'embed'")]

    if isinstance(config, Mapping):
        raw, issues = dict(config), []
    else:
        raw, issues = read_config(Path(config))
    return issues or check_config(typer.main.get_command(app), raw, exclude=CONFIG_EXCLUDE)
