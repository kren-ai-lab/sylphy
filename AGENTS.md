# AGENTS.md

This file provides guidance to AI Agents when working with code in this repository.

## Project Overview

**Sylphy** is a Python toolkit for transforming protein sequences into numerical representations for machine learning and bioinformatics workflows. It unifies three core components:

- **Classical sequence encoders** (one-hot, ordinal, frequency, k-mers, physicochemical, FFT)
- **Embedding extraction** from pretrained models (ESM2, ProtT5, ProtBERT, Ankh2, Mistral-Prot, ESM-C)
- **Dimensionality reduction** (linear and non-linear methods)

Design philosophy: minimal public API, lazy loading of heavy dependencies, no side effects on import, fully testable and configurable.

## Development Commands

This project uses `uv` as the package manager and `taskipy` for task running.

### Testing
```bash
# Run all tests (quiet)
uv run task test

# Run tests with verbose output
uv run task test-v

# Run tests with coverage report (generates HTML report)
uv run task test-cov

# Run a single test file
uv run pytest tests/cli/test_get_embeddings_cli.py -v

# Run a specific test
uv run pytest tests/cli/test_get_embeddings_cli.py::test_get_embeddings_runs_and_saves_csv -vv
```

### Linting & Formatting
```bash
# Run linter (check only)
uv run task lint

# Run linter and auto-fix issues
uv run task lint-fix

# Format code (includes import sorting)
uv run task format

# Sort imports only
uv run task sort-imports

# Run type checker
uv run task typecheck
```

### Building & Distribution
```bash
# Build distribution packages
uv build

# Install in editable mode for development
pip install -e .
```

### CLI Usage
```bash
# Show help and version
sylphy --help
sylphy --version

# Extract embeddings
sylphy get-embedding run \
  --model facebook/esm2_t6_8M_UR50D \
  --input-data data/sequences.csv \
  --sequence-identifier sequence \
  --output out/embeddings.parquet \
  --device cuda --precision fp16 --batch-size 16 --pool mean

# Encode sequences
sylphy encode-sequences run \
  --encoder one_hot \
  --input-data data/sequences.csv \
  --sequence-identifier sequence \
  --output out/onehot.csv

# Manage cache
sylphy cache ls          # List cached files
sylphy cache stats       # Show cache statistics
sylphy cache path        # Show cache directory path
sylphy cache prune       # Prune cache
sylphy cache rm          # Remove files by pattern/age
sylphy cache clear       # Clear entire cache
```

## Architecture

### Package Structure

```
sylphy/
├── cli/                    # Typer-based CLI commands (main.py, get_embeddings.py, encoder_sequences.py, cache.py)
├── constants/              # Configuration constants and tool definitions
├── core/                   # Core infrastructure
│   ├── config.py          # Cache configuration with appdirs
│   ├── model_spec.py      # ModelSpec dataclass for model metadata
│   └── model_registry.py  # Model registration, aliasing, and resolution
├── embedding_extractor/    # Embedding extraction backends
│   ├── embedding_based.py # Base class for all embedding extractors
│   ├── esm_based.py       # ESM2 models
│   ├── ankh2_based.py     # Ankh2 models
│   ├── prot5_based.py     # ProtT5 models
│   ├── bert_based.py      # ProtBERT models
│   ├── mistral_based.py   # Mistral-Prot models
│   ├── esmc_based.py      # ESM-C models (non-tokenizer backend)
│   └── embedding_factory.py # Factory pattern for creating embedders
├── logging/               # Unified hierarchical logging
│   └── logging_config.py  # setup_logger, get_logger, add_context
├── misc/                  # Utilities (UtilsLib for cache, file exports)
├── reductions/            # Dimensionality reduction
│   ├── reduction_methods.py    # Base class and enums
│   ├── linear_reductions.py    # PCA, TruncatedSVD, NMF
│   ├── non_linear_reductions.py # UMAP, Isomap, t-SNE, etc.
│   └── factory.py              # reduce_dimensionality factory
├── sequence_encoder/      # Classical sequence encoders
│   ├── base_encoder.py    # Common validation and preprocessing (Encoders class)
│   ├── one_hot_encoder.py # OneHotEncoder
│   ├── ordinal_encoder.py # OrdinalEncoder
│   ├── frequency_encoder.py # FrequencyEncoder
│   ├── kmers_encoder.py   # KMersEncoders (TF-IDF)
│   ├── physicochemical_encoder.py # PhysicochemicalEncoder (AAIndex properties)
│   ├── fft_encoder.py     # FFTEncoder (expects numeric input)
│   └── factory.py         # create_encoder factory
└── types.py               # Type definitions (PrecisionType, PoolType, LayerAggType, FileFormat)
```

### Key Design Patterns

#### 1. Lazy Loading
Heavy dependencies (PyTorch, Transformers) are loaded lazily in `embedding_extractor/__init__.py` using `__getattr__` to keep imports fast. When accessing classes like `ESMBasedEmbedding`, the module is imported on-demand and cached.

#### 2. Factory Pattern
All major components use factories:
- `create_encoder(encoder_type, ...)` → sequence encoders
- `create_embedding(model_name, ...)` or `EmbeddingFactory(...)` → embedding extractors
- `reduce_dimensionality(method, ...)` → dimensionality reducers

#### 3. Model Registry
The `core.model_registry` provides a centralized system for:
- Registering models with `ModelSpec(name, provider, ref)`
- Creating aliases (`register_alias("esm2_small", "esm2_t6")`)
- Resolving model paths with env var overrides (`SYLPHY_MODEL_<NAME>`)
- Downloading HuggingFace models if needed

#### 4. Unified Logging
- `setup_logger(name="sylphy", level="INFO")` configures the root logger once
- `get_logger("sylphy.component.subcomponent")` creates child loggers
- `add_context(logger, key=value, ...)` attaches structured metadata
- Optional `SYLPHY_LOG_FILE` env var for file output

#### 5. Two-Stage Encoding (FFT)
FFT encoders expect numeric input. Common workflow:
1. First encode with `physicochemical` encoder to get numeric matrix
2. Pass `phys.coded_dataset` to `fft` encoder

### Base Classes

#### `Encoders` (sequence_encoder/base_encoder.py)
- Common validation for all sequence encoders
- Parameters: `dataset`, `sequence_column`, `max_length`, `allow_extended`, `allow_unknown`
- Validates amino acid alphabet and filters sequences by length
- Subclasses implement `run_process()` and populate `coded_dataset`
- Export via `export_encoder(path)` (supports CSV/Parquet)

#### `EmbeddingBased` (embedding_extractor/embedding_based.py)
- Base for all embedding extractors
- Handles model/tokenizer loading from HuggingFace or local paths
- Manages device placement (cuda/cpu) and precision (fp32/fp16/bf16)
- OOM backoff: automatically reduces batch size on CUDA out-of-memory
- Pooling strategies: "mean", "cls", "eos"
- Layer selection: single int, list of ints, or "mean"/"sum"/"concat" aggregation
- Non-tokenizer backends (ESM-C) override `embedding_process()`
- Export via `export_encoder(path)` (supports CSV/Parquet)

#### `Reductions` (reductions/reduction_methods.py)
- Base for dimensionality reduction methods
- Returns: numpy array, pandas DataFrame, or fitted model + data
- Preprocessing options: standardization, normalization
- Linear methods: PCA, TruncatedSVD, NMF, Factor Analysis
- Non-linear methods: UMAP, t-SNE, Isomap, MDS, etc.

### CLI Structure

The CLI uses Typer with subcommands:
- `sylphy cache` → cache management (ls, stats, prune, rm, clear, path)
- `sylphy encode-sequences` → classical sequence encoders
- `sylphy get-embedding` → embedding extraction

Each CLI module follows the pattern:
1. Define `app = typer.Typer(...)`
2. Implement `run()` command with options
3. Register in `cli/main.py`

### Cache Management

Cache paths follow OS-specific conventions via `appdirs`:
- Linux: `~/.cache/sylphy/`
- macOS: `~/Library/Caches/sylphy/`
- Windows: `%LOCALAPPDATA%\sylphy\Cache`

Override with:
```python
from sylphy import set_cache_root, temporary_cache_root

set_cache_root("/custom/path")  # global override

with temporary_cache_root("/tmp/cache"):  # context manager
    ...
```

### Testing Architecture

The test suite runs **offline** with mocked HuggingFace and PyTorch dependencies:
- `tests/conftest.py` provides global fixtures
- Each submodule has its own `conftest.py` with module-specific fixtures
- Tests use fixtures to avoid actual model downloads
- Mock responses for tokenizers and models
- Coverage target enforced via `pytest-cov`

Test organization mirrors source structure:
```
tests/
├── cli/                    # CLI command tests
├── core/                   # Registry and config tests
├── embedding_extractor/    # Embedding extraction tests
├── logging/                # Logging configuration tests
├── reductions/             # Dimensionality reduction tests
└── sequence_encoder/       # Sequence encoder tests
```

## Dependencies

### Core (always required)
- `pandas`, `numpy`, `scipy`, `scikit-learn`
- `typer`, `rich` (CLI)
- `appdirs`, `huggingface-hub` (cache/registry)

### Optional Groups
- `embeddings`: `torch`, `transformers`, `sentencepiece`, `esm` (ESM-C)
- `reductions`: `umap-learn`, `clustpy`
- `parquet`: `pyarrow`, `fastparquet`
- `dev`: `pyrefly`, `pytest`, `ruff`, `taskipy`, `build`, `twine`
- `tests`: `pytest`, `pytest-cov`
- `all`: all optional dependencies

Install with:
```bash
pip install -e ".[embeddings,reductions,parquet]"
```

## Code Style

- **Python 3.11+** required (3.12 target)
- **Type annotations**: fully type-annotated with `from __future__ import annotations`
- **Docstrings**: NumPy-style docstrings for all public functions/classes
- **Linting**: Ruff with line length 110
- **No side effects**: imports are clean; no initialization on import
- **Reproducibility**: use `random_state=0` in stochastic reducers
- **Logging over print**: use `logger.info/debug` instead of print statements

## Common Patterns

### Creating a new sequence encoder
1. Inherit from `Encoders` in `sequence_encoder/base_encoder.py`
2. Implement `run_process(self) -> None` to populate `self.coded_dataset`
3. Register in `sequence_encoder/factory.py` encoder mapping
4. Add exports to `sequence_encoder/__init__.py`
5. Add tests in `tests/sequence_encoder/`

### Creating a new embedding backend
1. Inherit from `EmbeddingBased` in `embedding_extractor/embedding_based.py`
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
