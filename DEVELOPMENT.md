# Development

This guide covers local setup, day-to-day commands, and the code structure behind Sylphy.

## Local Setup

Sylphy uses `uv` for environment management and `taskipy` for common developer commands.

Install development dependencies:

```bash
uv sync --extra dev
```

If you also need embeddings, parquet, and reduction extras locally:

```bash
uv sync --extra dev --extra all
```

Editable install with `pip` also works:

```bash
pip install -e ".[dev]"
pip install -e ".[all,dev]"
```

## Common Commands

Run tests:

```bash
uv run task test
uv run task test-v
uv run task test-cov
uv run pytest tests/cli/test_get_embeddings_cli.py -v
uv run pytest tests/cli/test_get_embeddings_cli.py::test_get_embeddings_runs_and_saves_csv -vv
```

Lint, format, and type-check:

```bash
uv run task lint
uv run task lint-fix
uv run task format
uv run task sort-imports
uv run task ty
uv run task pyrefly
```

Build the package:

```bash
uv build
```

Smoke-test the CLI:

```bash
uv run sylphy --help
uv run sylphy --version
```

## Project Layout

```text
sylphy/
├── cli/                  # Typer commands
├── constants/            # Shared constants and config values
├── core/                 # Registry, config, optional dependency helpers
├── embedding_extractor/  # Protein language model backends
├── logging/              # Unified logging setup
├── misc/                 # Export and cache helpers
├── reductions/           # Dimensionality reduction methods and factory
├── sequence_encoder/     # Classical encoders and factory
└── types.py              # Shared type aliases and enums

tests/                    # Mirrors the source tree
examples/                 # Scripts and notebooks for manual exploration
```

## Design Rules

- Keep the public API small and explicit.
- Avoid side effects on import.
- Preserve lazy loading for heavy embedding dependencies.
- Prefer factory entry points over ad hoc object construction.
- Use logging instead of `print`.
- Keep stochastic behavior reproducible with explicit seeds or `random_state=0`.

Core factory entry points:

- `create_encoder(...)`
- `create_embedding(...)`
- `reduce_dimensionality(...)`

## Testing Notes

The test suite is designed to run offline.

- Hugging Face and PyTorch interactions are mocked in tests.
- Test layout mirrors the source layout.
- New features should ship with focused unit tests and CLI coverage when applicable.

## Extending Sylphy

Adding a new sequence encoder:

1. Inherit from `Encoders` in `sylphy/sequence_encoder/base_encoder.py`.
2. Implement `run_process()` and populate `self.coded_dataset`.
3. Register the encoder in `sylphy/sequence_encoder/factory.py`.
4. Export it from `sylphy/sequence_encoder/__init__.py` if needed.
5. Add tests under `tests/sequence_encoder/`.

Adding a new embedding backend:

1. Inherit from `EmbeddingBased` in `sylphy/embedding_extractor/embedding_based.py`.
2. Override initialization or `embedding_process()` only when necessary.
3. Register the backend in `sylphy/embedding_extractor/embedding_factory.py`.
4. Add lazy exports in `sylphy/embedding_extractor/__init__.py`.
5. Add tests under `tests/embedding_extractor/`.

Adding a new reduction method:

1. Implement it in the appropriate reductions module.
2. Register it in `sylphy/reductions/factory.py`.
3. Include it in public availability helpers.
4. Add tests under `tests/reductions/`.

## Environment Variables

Useful variables while developing:

- `SYLPHY_CACHE_ROOT` to override the cache directory
- `SYLPHY_MODEL_<NAME>` to point a registered model alias at a local path
- `SYLPHY_DEVICE` to force `cpu` or `cuda`
- `SYLPHY_SEED` to set a reproducibility seed
- `SYLPHY_LOG_LEVEL`, `SYLPHY_LOG_FILE`, `SYLPHY_LOG_JSON` for logging behavior

## Additional Context

For more architecture detail, see [CLAUDE.md](CLAUDE.md).
