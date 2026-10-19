"""``--config`` files whose keys mirror a Typer command's options.

Keys are option parameter names (``--max-length`` -> ``max_length``). Values are checked with the
option's own Click type (through Typer), and accepted files become the context's ``default_map``, so the
precedence is defaults < config file < options on the command line. Copied into each library, not imported.
"""
# ruff: noqa: ANN401

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import typer
import yaml

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class ConfigIssue:
    """One problem found in a config file; ``path`` is ``""`` for the root."""

    path: str
    message: str


def read_config(path: Path) -> tuple[dict[str, Any], list[ConfigIssue]]:
    """Parse a YAML (or JSON) config whose root must be a mapping. An empty file is ``{}``."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return {}, [ConfigIssue("", f"cannot read config: {exc}")]
    if raw is None:
        return {}, []
    if not isinstance(raw, dict):
        return {}, [ConfigIssue("", f"config root must be a mapping, got {type(raw).__name__}")]
    return raw, []


def check_config(command: Any, raw: dict[str, Any], *, exclude: frozenset[str]) -> list[ConfigIssue]:
    """Check every key and value against ``command``'s options.

    ``exclude`` names options that must stay on the CLI.
    """
    ctx = typer.Context(command)
    # duck-typed: Typer >=0.27 vendors Click, so isinstance checks against `click` fail
    options = {
        p.name: p for p in command.params if getattr(p, "param_type_name", None) == "option" and p.name
    }
    issues: list[ConfigIssue] = []
    for key, value in raw.items():
        param = options.get(key)
        if param is None:
            issues.append(ConfigIssue(str(key), "unknown key"))
        elif key in exclude:
            issues.append(ConfigIssue(key, f"pass {param.opts[0]} on the command line, not in the config"))
        elif param.multiple and not isinstance(value, list):
            issues.append(ConfigIssue(key, f"expected a list, got {type(value).__name__}"))
        else:
            try:
                param.type_cast_value(ctx, value)
            except typer.BadParameter as exc:
                issues.append(ConfigIssue(key, exc.format_message()))
            except (AttributeError, TypeError):  # Click's BOOL type calls .strip() on non-strings
                issues.append(ConfigIssue(key, f"invalid value {value!r}"))
    return issues


def apply_config(ctx: typer.Context, path: Path | None, *, exclude: frozenset[str]) -> dict[str, Any]:
    """Validate ``path`` and install it as ``ctx.default_map``; use from an eager ``--config`` callback.

    Raises ``typer.BadParameter`` (exit code 2) listing every issue.
    """
    if path is None:
        return {}
    raw, issues = read_config(path)
    issues = issues or check_config(ctx.command, raw, exclude=exclude)
    if issues:
        lines = "\n".join(f"  {issue.path or '<root>'}: {issue.message}" for issue in issues)
        msg = f"{path}:\n{lines}"
        raise typer.BadParameter(msg, ctx=ctx, param_hint="'--config'")
    ctx.default_map = {**(ctx.default_map or {}), **raw}
    return raw


def report_issues(issues: list[ConfigIssue], *, as_json: bool) -> int:
    """Print issues in the contract's format and return the exit code (0 valid, 2 invalid)."""
    if as_json:
        typer.echo(json.dumps([asdict(issue) for issue in issues]))
    else:
        for issue in issues:
            typer.echo(f"{issue.path or '<root>'}: {issue.message}", err=True)
    if not issues and not as_json:
        typer.echo("config is valid", err=True)
    return 2 if issues else 0
