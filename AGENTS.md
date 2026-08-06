# AGENTS.md

This file provides guidance to AI Agents when working with code in this repository.

Design philosophy: minimal public API, lazy loading of heavy dependencies, no side effects on import, fully testable and configurable.

Package management is `uv`; task running is `taskipy` (see `[tool.taskipy.tasks]` in `pyproject.toml` for the current task list).

## Gotchas

### Lazy loading
Heavy dependencies (PyTorch, Transformers) are loaded lazily in `embedding_extractor/__init__.py` using `__getattr__` to keep imports fast. When accessing classes like `ESMEmbedding`, the module is imported on-demand and cached. Do not move these imports to module top level.

### Two-stage encoding (FFT)
FFT encoders expect **numeric** input, not sequences. Common workflow:
1. First encode with the `physicochemical` encoder to get a numeric matrix
2. Pass `phys.coded_dataset` to the `fft` encoder

### Tests run offline
The test suite runs **offline** with mocked HuggingFace and PyTorch dependencies. Tests must never download a model; add fixtures instead.

## Code Style

- **Type annotations**: fully type-annotated with `from __future__ import annotations`
- **Docstrings**: Google-style docstrings for all public functions/classes
- **No side effects**: imports are clean; no initialization on import
- **Reproducibility**: use `random_state=0` in stochastic reducers
- **Logging over print**: use `logger.info/debug` instead of print statements

## Common Patterns

### Creating a new sequence encoder
1. Inherit from `EncoderBase` in `sequence_encoder/encoder_base.py`
2. Implement `run_process(self) -> None` to populate `self.coded_dataset`
3. Register in `sequence_encoder/factory.py` encoder mapping
4. Add exports to `sequence_encoder/__init__.py`
5. Add tests in `tests/sequence_encoder/`

### Creating a new embedding backend
1. Inherit from `EmbeddingBase` in `embedding_extractor/embedding_based.py`
2. Override `__init__` if custom initialization needed
3. For non-tokenizer models, set `requires_tokenizer=False` and override `embedding_process()`
4. Register in `embedding_extractor/embedding_factory.py` model family mapping
5. Add lazy export to `embedding_extractor/__init__.py`
6. Add tests in `tests/embedding_extractor/`

### Adding a new reduction method
1. Add method to `LinearReduction` or `NonLinearReductions` class
2. Update method mappings in `reductions/factory.py`
3. Add to `get_available_methods()` output
4. Add tests in `tests/reductions/`

## Environment Variables

### Core Configuration
- `SYLPHY_CACHE_ROOT`: override the default cache directory for model weights and intermediate files
  - If not set, defaults to OS-specific cache directories:
    - Linux: `~/.cache/sylphy` (or `$XDG_CACHE_HOME/sylphy`)
    - macOS: `~/Library/Caches/sylphy`
    - Windows: `%LOCALAPPDATA%\sylphy\Cache`
- `SYLPHY_MODEL_<NAME>`: override model path (e.g., `SYLPHY_MODEL_ESM2_SMALL`)
- `SYLPHY_DEVICE`: force device selection (`cpu` or `cuda`), overrides auto-detection
- `SYLPHY_SEED`: set random seed for reproducibility (default: `42`)

### Logging Configuration
- `SYLPHY_LOG_FILE`: path for log file output
- `SYLPHY_LOG_LEVEL`: logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`)
- `SYLPHY_LOG_JSON`: enable JSON-formatted logs (`true`/`false`, `1`/`0`)
- `SYLPHY_LOG_STDERR`: log to stderr instead of stdout (`true`/`false`, `1`/`0`)
- `SYLPHY_LOG_UTC`: use UTC timestamps in logs (`true`/`false`, `1`/`0`)
- `SYLPHY_LOG_MAX_BYTES`: max bytes per log file before rotation (default: 10MB)
- `SYLPHY_LOG_BACKUPS`: number of backup log files to keep (default: 3)

### External Dependencies
- `HF_HOME`, `TRANSFORMERS_CACHE`: HuggingFace cache location
