"""Register the main Typer entrypoint and top-level CLI commands."""

from __future__ import annotations

import sys
import traceback
from typing import TYPE_CHECKING, Any

import typer
from typer.core import TyperGroup
from typer.exceptions import TyperException

from sylphy import __version__
from sylphy.cli._shared import HELP_CONTEXT_SETTINGS
from sylphy.cli.cache import app as cache_app
from sylphy.cli.embed import app as embed_app
from sylphy.cli.encode import app as encode_app
from sylphy.cli.validate import validate
from sylphy.exceptions import SylphyError

if TYPE_CHECKING:
    from typer._click.core import Context

EXIT_RUNTIME = 3  # runtime and data errors (shared ecosystem CLI contract)
EXIT_INTERNAL = 4  # unexpected errors


def _is_runtime_error(exc: Exception) -> bool:
    from polars.exceptions import PolarsError  # noqa: PLC0415

    return isinstance(exc, SylphyError | OSError | PolarsError | UnicodeDecodeError)


class _ExitCodeGroup(TyperGroup):
    """Map errors escaping a command to exit codes: 3 for runtime/data errors, 4 for anything else."""

    def invoke(self, ctx: Context) -> Any:  # noqa: ANN401
        try:
            return super().invoke(ctx)
        except (TyperException, typer.Exit, typer.Abort):
            raise
        except Exception as exc:
            if _is_runtime_error(exc):
                typer.echo(f"Error: {exc}", err=True)
                raise typer.Exit(EXIT_RUNTIME) from exc
            traceback.print_exception(exc, file=sys.stderr)
            raise typer.Exit(EXIT_INTERNAL) from exc


app = typer.Typer(
    cls=_ExitCodeGroup,
    name="sylphy",
    add_completion=False,
    context_settings=HELP_CONTEXT_SETTINGS,
    help="Tools to numerically represent protein sequences (encoders, embeddings, reductions, cache).",
)


def version_callback(value: bool | None) -> None:  # noqa: FBT001
    """Show version and exit."""
    if value:
        typer.echo(f"sylphy {__version__}")
        raise typer.Exit


@app.callback()
def main(
    _version: bool | None = typer.Option(  # noqa: FBT001
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """Sylphy CLI main callback."""


app.add_typer(cache_app, name="cache")
app.add_typer(encode_app, name="encode")
app.add_typer(embed_app, name="embed")
app.command(name="validate-config")(validate)

if __name__ == "__main__":
    app()
