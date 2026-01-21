# Sylphy 🧬

**Lightweight Python toolkit for protein sequence representation** — transform sequences into numerical formats for
machine learning and bioinformatics.

Three core components:

- **Classical encoders** — one-hot, ordinal, frequency, k-mers, physicochemical, FFT
- **Embedding extraction** — ESM2, ProtT5, ProtBERT, Ankh2, Mistral-Prot, ESM-C
- **Dimensionality reduction** — PCA, UMAP, t-SNE, and more

## Quick Example

```python
import pandas as pd
from sylphy.embedding_extractor import create_embedding

# Extract embeddings from protein sequences
df = pd.DataFrame({"sequence": ["MKTAYIAKQR", "GAVLIMPFWK", "PEPTIDE"]})

embedder = create_embedding(
    model_name="facebook/esm2_t6_8M_UR50D",
    dataset=df,
    column_seq="sequence",
    name_device="cuda",
    precision="fp16"
)

embedder.run_process(batch_size=8, pool="mean")
embeddings = embedder.coded_dataset  # pandas DataFrame with embeddings
embedder.export_encoder("embeddings.parquet")
```

## Installation

**Recommended:** Use a virtual environment to isolate dependencies:

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate
```

Install directly from GitHub:

```bash
# Basic installation
pip install "sylphy @ git+https://github.com/kren-ai-lab/sylphy"

# With optional variants
pip install "sylphy[embeddings,parquet] @ git+https://github.com/kren-ai-lab/sylphy"
```

Download the latest `.whl` file from [Releases](https://github.com/ProteinEngineering-PESB2/sylphy_library/releases):

```bash
# Basic installation
pip install sylphy-<version>-py3-none-any.whl

# With optional variants
pip install sylphy-<version>-py3-none-any.whl[embeddings,parquet]
```

The basic installation includes classical sequence encoders and core utilities. For additional features, install optional variants:

### Installation Variants

| Variant          | Description                                                                                                                                                  |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **`embeddings`** | Adds PyTorch, Transformers, and ESM-C SDK for protein language model embedding extraction (ESM2, ProtT5, ProtBERT, Ankh2, Mistral-Prot, ESM-C).              |
| **`parquet`**    | Enables Parquet file format support via PyArrow and FastParquet for efficient storage and loading of large datasets.                                         |
| **`reductions`** | Adds UMAP and ClustPy for advanced non-linear dimensionality reduction methods. **Requires a C++ compiler and Python development headers** to build ClustPy. |
| **`all`**        | Installs all optional dependencies (embeddings + parquet + reductions). **Requires compilation tools** for ClustPy.                                          |
| **`tests`**      | Installs pytest and pytest-cov for running the test suite with coverage reports.                                                                             |
| **`dev`**        | Development tools including pytest, mypy, ruff, taskipy, and build utilities for contributing to Sylphy.                                                     |

**Example installations:**

```bash
# Embeddings + Parquet support
pip install sylphy-<version>-py3-none-any.whl[embeddings,parquet]

# Full installation with all features
pip install sylphy-<version>-py3-none-any.whl[all]
```

**Requirements:**
- Python 3.11–3.12
- Optional: CUDA for GPU-accelerated embedding extraction
- For `reductions` variant: C++ compiler and Python development headers
  ```bash
  # Ubuntu/Debian
  sudo apt-get install build-essential python3-dev

  # Fedora/RHEL
  sudo dnf install gcc gcc-c++ python3-devel
  ```

## Usage

### Sequence Encoders

Transform sequences using classical encoding methods:

```python
from sylphy.sequence_encoder import create_encoder

encoder = create_encoder(
    "one_hot",  # or: ordinal, kmers, frequency, physicochemical, fft
    dataset=df,
    sequence_column="sequence",
    max_length=1024
)

encoder.run_process()
encoded = encoder.coded_dataset
encoder.export_encoder("encoded.csv")
```

**FFT encoding** requires numeric input (use a two-stage pipeline):

```python
# Stage 1: physicochemical properties
phys = create_encoder("physicochemical", dataset=df, name_property="ANDN920101")
phys.run_process()

# Stage 2: FFT on numeric matrix
fft = create_encoder("fft", dataset=phys.coded_dataset)
fft.run_process()
```

### Embedding Extraction

Extract embeddings from pretrained protein language models:

```python
from sylphy.embedding_extractor import create_embedding

embedder = create_embedding(
    model_name="facebook/esm2_t6_8M_UR50D",
    dataset=df,
    column_seq="sequence",
    name_device="cuda",
    precision="fp16",  # fp32, fp16, or bf16
    oom_backoff=True  # auto-reduce batch size on OOM
)

embedder.run_process(
    max_length=1024,
    batch_size=16,
    pool="mean"  # mean, cls, or eos
)
```

**Supported models:** ESM2 • Ankh2 • ProtT5 • ProtBERT • Mistral-Prot • ESM-C

### Dimensionality Reduction

Reduce high-dimensional embeddings for visualization:

```python
from sylphy.reductions import reduce_dimensionality

model, reduced = reduce_dimensionality(
    method="umap",  # pca, truncated_svd, umap, tsne, isomap, etc.
    dataset=embeddings,
    n_components=2,
    random_state=42,
    return_type="numpy"  # numpy or pandas
)
```

## Command-Line Interface

```bash
# Extract embeddings
sylphy get-embedding run \
  --model facebook/esm2_t6_8M_UR50D \
  --input-data sequences.csv \
  --sequence-identifier sequence \
  --output embeddings.parquet \
  --device cuda --precision fp16 --batch-size 16

# Encode sequences
sylphy encode-sequences run \
  --encoder one_hot \
  --input-data sequences.csv \
  --sequence-identifier sequence \
  --output encoded.csv

# Manage cache
sylphy cache ls        # List cached files
sylphy cache stats     # Cache statistics
sylphy cache prune     # Prune cache (remove old files or reduce size)
sylphy cache rm        # Remove files by pattern or age
sylphy cache clear     # Clear entire cache

# Version info
sylphy --version
```

## Configuration

### Cache Management

Models and intermediate files are cached at:

- **Linux:** `~/.cache/sylphy`
- **macOS:** `~/Library/Caches/sylphy`
- **Windows:** `%LOCALAPPDATA%\sylphy`

**Programmatic control:**

```python
from sylphy import get_config, set_cache_root, temporary_cache_root

# View current cache location
cfg = get_config()
print(cfg.cache_paths.cache_root)

# Change cache directory
set_cache_root("/custom/cache/path")

# Temporary override
with temporary_cache_root("/tmp/cache"):
    # operations use temporary cache
    pass
```

**Environment variables:**

```bash
export SYLPHY_CACHE_ROOT=/custom/cache     # Override cache location
export SYLPHY_DEVICE=cuda                  # Force device (cpu/cuda)
export SYLPHY_LOG_FILE=/tmp/sylphy.log     # Enable file logging
export SYLPHY_SEED=42                      # Random seed
```

### Model Registry

Register custom models and aliases:

```python
from sylphy import ModelSpec, register_model, register_alias, resolve_model

# Register a model
register_model(ModelSpec(
    name="esm2_small",
    provider="huggingface",
    ref="facebook/esm2_t6_8M_UR50D"
))

# Create alias
register_alias("my_model", "esm2_small")

# Resolve to path
path = resolve_model("my_model")
```

Override model paths via environment:

```bash
export SYLPHY_MODEL_ESM2_SMALL=/path/to/local/model
```

### Logging

Optional unified logging configuration:

```python
from sylphy.logging import setup_logger

setup_logger(name="sylphy", level="INFO")  # DEBUG, INFO, WARNING, ERROR
```

## Examples

The `examples/` directory contains complete working examples:

- **`1_quick_start_encoders.ipynb`** — Jupyter notebook demonstrating all classical encoders
- **`2_simple_demo_embedding_extractor.py`** — Extract embeddings using all supported model families
- **`3_quick_start_reduction_process.ipynb`** — Dimensionality reduction workflows
- **`4_demo_embedding_different_layers.py`** — Layer selection and aggregation strategies
- **`encoder_sequences_using_sylphy.py`** — Batch encoding with multiple encoder types
- **`extract_embedding_using_sylphy.py`** — Production-ready embedding extraction script

Run examples:

```bash
# Python scripts
python examples/2_simple_demo_embedding_extractor.py

# Jupyter notebooks
jupyter notebook examples/1_quick_start_encoders.ipynb
```

## Development

### Setup

Clone the repository and install in editable mode:

```bash
git clone https://github.com/kren-ai-lab/sylphy.git
cd sylphy

# Install with development dependencies
pip install -e ".[dev]"

# Or install with all features for testing
pip install -e ".[all,dev]"
```

**Note:** The `-e` flag installs in editable mode, meaning changes to the source code take effect immediately without reinstalling.

### Testing

```bash
# Run tests
pytest                # All tests (offline, mocked)
pytest -v             # Verbose
pytest --cov=sylphy   # With coverage

# Using taskipy shortcuts
uv run task test      # Run tests (quiet)
uv run task test-v    # Run tests (verbose)
uv run task test-cov  # Run tests with coverage report
```

### Code Quality

```bash
# Linting and formatting
ruff check sylphy/    # Lint
ruff format sylphy/   # Format
mypy sylphy/          # Type check

# Using taskipy shortcuts
uv run task lint      # Lint check
uv run task lint-fix  # Lint and auto-fix
uv run task format    # Format code
```

### Architecture

- Fully typed with annotations
- NumPy-style docstrings
- Factory pattern for all components
- Lazy imports for heavy dependencies
- Offline tests with mocked PyTorch/HF

## API Reference

Main imports:

```python
from sylphy import (
    # Configuration / registry
    get_config, set_cache_root, temporary_cache_root,
    ModelSpec, register_model, resolve_model,
)
from sylphy.sequence_encoder import create_encoder
from sylphy.embedding_extractor import create_embedding
from sylphy.reductions import reduce_dimensionality
from sylphy.logging import setup_logger, get_logger
```

See [CLAUDE.md](CLAUDE.md) for detailed architecture documentation.

## License

**GPL-3.0-only** — See [LICENSE](LICENSE) for details.

## Acknowledgements

Built with:

- **Hugging Face** Transformers ecosystem
- **Meta** ESM-C SDK
- **scikit-learn** • **PyTorch** • **UMAP** • **ClustPy**

Developed by **KREN AI Lab** at Universidad de Magallanes, Chile.
