"""Register the main Typer entrypoint and top-level CLI commands."""

from __future__ import annotations

import typer

from sylphy import __version__
from sylphy.cli._shared import HELP_CONTEXT_SETTINGS
from sylphy.cli.cache import app as cache_app
from sylphy.cli.embed import embed
from sylphy.cli.encode import encode

app = typer.Typer(
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
app.command("encode")(encode)
app.command("embed")(embed)

if __name__ == "__main__":
    app()
