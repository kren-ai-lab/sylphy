"""Register the main Typer entrypoint and top-level CLI commands."""

from __future__ import annotations

import typer

from sylphy import __version__
from sylphy.cli._shared import HELP_CONTEXT_SETTINGS
from sylphy.cli.cache import app as cache_app
from sylphy.cli.encoder_sequences import encode_sequences
from sylphy.cli.get_embeddings import get_embedding

app = typer.Typer(
    name="sylphy",
    add_completion=False,
    context_settings=HELP_CONTEXT_SETTINGS,
    help="Tools to numerically represent protein sequences (encoders, embeddings, reductions, cache).",
)


def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        typer.echo(f"sylphy {__version__}")
        raise typer.Exit


@app.callback()
def main(
    _version: bool = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """Sylphy CLI main callback."""


# Cache management
app.add_typer(
    cache_app,
    name="cache",
    help="Inspect and manage the library cache (list, stats, prune, remove).",
)

app.command(
    name="encode-sequences",
    help="Encode sequences (one-hot, ordinal, freq, kmers, physchem, FFT)",
)(encode_sequences)

app.command(
    name="get-embedding",
    help="Extract embeddings from pretrained protein models",
)(get_embedding)

if __name__ == "__main__":
    app()
