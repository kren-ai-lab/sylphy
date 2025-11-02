# protein_representation/cli/main.py
from __future__ import annotations

import typer

from sylphy import __version__
from sylphy.cli.cache import app as cache_app
from sylphy.cli.encoder_sequences import app as encoder_sequence_app
from sylphy.cli.get_embeddings import app as embedding_extractor_app

app = typer.Typer(
    name="sylphy",
    add_completion=False,
    help="Tools to numerically represent protein sequences (encoders, embeddings, reductions, cache).",
)


def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        typer.echo(f"sylphy version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """Sylphy CLI main callback."""
    pass


# Cache management
app.add_typer(
    cache_app,
    name="cache",
    help="Inspect and manage the library cache (list, stats, prune, remove).",
)

app.add_typer(
    encoder_sequence_app,
    name="encode-sequences",
    help="Encode sequences (one-hot, ordinal, freq, kmers, physchem, FFT)",
)

app.add_typer(
    embedding_extractor_app, name="get-embedding", help="Extract embeddings from pretrained protein models"
)

if __name__ == "__main__":
    app()
