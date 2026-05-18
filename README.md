# Sylphy 🧬
[![PyPI](https://img.shields.io/pypi/v/sylphy?style=flat-square)](https://pypi.org/project/sylphy/)
[![PyVersions](https://img.shields.io/pypi/pyversions/sylphy?style=flat-square)](https://github.com/kren-ai-lab/sylphy)
[![Tests](https://img.shields.io/github/actions/workflow/status/kren-ai-lab/sylphy/tests.yml?style=flat-square)](https://github.com/kren-ai-lab/sylphy/actions/workflows/tests.yml)
![License](https://img.shields.io/github/license/kren-ai-lab/sylphy?style=flat-square)

Sylphy is a Python toolkit for turning protein sequences into machine-learning-ready representations.

It covers three main workflows:

- Classical sequence encoders: one-hot, ordinal, frequency, k-mers, physicochemical, FFT
- Embedding extraction from pretrained protein models: ESM2, ProtT5, ProtBERT, Ankh2, Mistral-Prot, ESM-C
- Dimensionality reduction for downstream analysis and visualization

## Installation

Sylphy supports Python 3.11 and 3.12.

```bash
pip install sylphy
```

Install optional extras as needed:

- `embeddings` for PyTorch and Transformers-based embedding extraction
- `reductions` for UMAP and related optional reducers
- `all` for all optional runtime dependencies

The `reductions` extra may require a C++ compiler and Python development headers because of optional native dependencies such as ClustPy.

```bash
pip install 'sylphy[embeddings]'
pip install 'sylphy[all]'
```

On Debian or Ubuntu systems, install the build prerequisites with:

```bash
sudo apt-get install build-essential python3-dev
```

On Fedora or RHEL systems:

```bash
sudo dnf install gcc gcc-c++ python3-devel
```

## Quick Start

Classical sequence encoding:

```python
import polars as pl
from sylphy.sequence_encoder import create_encoder

df = pl.DataFrame({"sequence": ["MKTAYIAKQR", "GAVLIMPFWK", "PEPTIDE"]})

encoder = create_encoder(
    "one_hot",  # or: ordinal, kmers, frequency, physicochemical, fft
    dataset=df,
    sequence_column="sequence",
)
encoder.run_process()
encoded = encoder.coded_dataset
```

Embedding extraction:

```python
import polars as pl
from sylphy.embedding_extractor import create_embedding

df = pl.DataFrame({"sequence": ["MKTAYIAKQR", "GAVLIMPFWK", "PEPTIDE"]})

embedder = create_embedding(
    model_name="facebook/esm2_t6_8M_UR50D",
    dataset=df,
    column_seq="sequence",
    name_device="cuda",
    precision="fp16",  # fp32, fp16, or bf16
)

embedder.run_process(batch_size=8, pool="mean")  # mean, cls, or eos
embeddings = embedder.coded_dataset
embedder.export_encoder("embeddings.parquet")
```

Dimensionality reduction:

```python
from sylphy.reductions import reduce_dimensionality

model, reduced = reduce_dimensionality(
    method="pca",  # pca, truncated_svd, umap, tsne, isomap, etc.
    dataset=embeddings,
    n_components=2,
    random_state=42,
)
```

## CLI

```bash
sylphy --help

# Extract embeddings (ESM2, GPU, fp16)
sylphy embed \
  -i sequences.csv -o embeddings.parquet \
  -m facebook/esm2_t6_8M_UR50D -d cuda -p fp16 -b 16

# Extract embeddings (ProtT5, last 4 layers averaged)
sylphy embed \
  -i sequences.csv -o embeddings.npy \
  -m Rostlab/prot_t5_xl_uniref50 -d cuda -p bf16 --layers last4 --layer-agg mean

# Classical encoding — one-hot
sylphy encode --method one_hot -i sequences.csv -o encoded.parquet

# Classical encoding — physicochemical (AAIndex)
sylphy encode --method physicochemical \
  -i sequences.csv -o phys.parquet --name-property ARGP820101

# Classical encoding — k-mers TF-IDF
sylphy encode --method kmers -i sequences.csv -o kmers.csv -k 4

# Cache management
sylphy cache stats
sylphy cache ls --recursive
sylphy cache prune --max-size 10GB --apply
```

## Configuration

By default Sylphy stores cache data in the platform cache directory:

- Linux: `~/.cache/sylphy`
- macOS: `~/Library/Caches/sylphy`
- Windows: `%LOCALAPPDATA%\\sylphy\\Cache`

Useful environment variables:

- `SYLPHY_CACHE_ROOT` to override the cache location
- `SYLPHY_DEVICE` to force `cpu` or `cuda`
- `SYLPHY_MODEL_<NAME>` to override a registered model path

## Learn More

- [DEVELOPMENT.md](DEVELOPMENT.md) for local setup, tests, architecture, and contribution notes
- [examples/README.md](examples/README.md) for the examples index and runnable scripts/notebooks

## License

**GPL-3.0-only**. See [LICENSE](LICENSE).

## Acknowledgements

Built with the Hugging Face Transformers ecosystem, the Meta ESM-C SDK, and the broader scientific Python stack including scikit-learn, PyTorch, UMAP, and ClustPy.

Developed by **KREN AI Lab** at Universidad de Magallanes, Chile.
