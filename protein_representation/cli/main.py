# protein_representation/cli/main.py
from __future__ import annotations

import typer

from protein_representation.cli.cache import app as cache_app
from protein_representation.cli.encoder_sequences import app as encoder_app
from protein_representation.cli.get_embeddings import app as embedding_app
from protein_representation.cli.reduce import app as reduce_app

app = typer.Typer(
    name="protein_representation",
    add_completion=False,
    help="Tools to numerically represent protein sequences (encoders, embeddings, reductions, cache).",
)

app.add_typer(
    embedding_app,
    name="get-embedding",
    help="Extract embeddings from pretrained protein language models.",
)

app.add_typer(
    encoder_app,
    name="encode-sequences",
    help="Encode sequences using classical strategies (one-hot, k-mers, physicochemical, FFT...).",
)

app.add_typer(
    reduce_app,
    name="reduce",
    help="Apply dimensionality reduction to embedding/feature matrices.",
)

app.add_typer(
    cache_app,
    name="cache",
    help="Inspect and manage the library cache (list, stats, prune, remove).",
)

if __name__ == "__main__":
    app()
