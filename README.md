# Sylphy 🧬
[![PyPI](https://img.shields.io/pypi/v/sylphy?style=flat-square)](https://pypi.org/project/sylphy/)
[![PyVersions](https://img.shields.io/pypi/pyversions/sylphy?style=flat-square)](https://github.com/kren-ai-lab/sylphy)
[![Tests](https://img.shields.io/github/actions/workflow/status/kren-ai-lab/sylphy/tests.yml?style=flat-square)](https://github.com/kren-ai-lab/sylphy/actions/workflows/tests.yml)
![License](https://img.shields.io/github/license/kren-ai-lab/sylphy?style=flat-square)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21819735-blue?style=flat-square)](https://doi.org/10.5281/zenodo.21819735)

Sylphy is a Python toolkit for turning protein sequences into machine-learning-ready representations.

It covers three main workflows:

- Classical sequence encoders: one-hot, ordinal, frequency, k-mers, physicochemical, FFT
- Embedding extraction from pretrained protein models: ESM2, ProtT5, ProtBERT, Ankh2, Mistral-Prot, ESM-C
- Dimensionality reduction for downstream analysis and visualization

## Installation

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

df = pl.DataFrame({"id": ["p1", "p2", "p3"], "sequence": ["MKTAYIAKQR", "GAVLIMPFWK", "PEPTIDE"]})

encoder = create_encoder(
    "one_hot",  # or: ordinal, kmers, frequency, physicochemical, fft
    dataset=df,
    sequence_column="sequence",
)
encoder.run_process()
encoded = encoder.coded_dataset
encoder.export_encoder("encoded.parquet")  # same format as the CLI
```

Embedding extraction:

```python
import polars as pl
from sylphy.embedding_extractor import create_embedding

df = pl.DataFrame({"id": ["p1", "p2", "p3"], "sequence": ["MKTAYIAKQR", "GAVLIMPFWK", "PEPTIDE"]})

embedder = create_embedding(
    model_name="facebook/esm2_t6_8M_UR50D",
    dataset=df,
    column_seq="sequence",
    name_device="cuda",
    precision="fp16",  # fp32, fp16, or bf16
)

embedder.run_process(batch_size=8, pool="mean")  # mean, cls, or eos
embeddings = embedder.coded_dataset
embedder.export_encoder("embeddings.parquet")  # id_col="id" by default
```

`export_encoder(path, file_format, *, id_col="id")` writes the same files as the CLI (see [Inputs and outputs](#inputs-and-outputs)), keyed by the `id_col` column of the input dataset.
A missing, empty or duplicated id raises `sylphy.exceptions.DataError`.
For `fft`, pass the id along with the physicochemical matrix: `phys.coded_dataset.with_columns(phys.dataset["id"])`.

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

# Options from a YAML/JSON file (keys are option names, e.g. max_length); flags override the file
sylphy encode -i sequences.csv -o encoded.parquet --config encode.yaml
sylphy validate-config encode encode.yaml --json

# Cache management
sylphy cache stats
sylphy cache ls --recursive
sylphy cache prune --max-size 10GB --apply
```

### Validate a config from Python

```python
from sylphy import ConfigIssue, validate_config

issues = validate_config("encode.yaml", "encode")  # or a mapping; target is "encode" or "embed"
for issue in issues:
    print(issue.path or "<root>", issue.message)
```

An empty list means the config is valid; a missing or unparsable file is one issue with path `""`.

### Inputs and outputs

- Input: FASTA (`.fa`, `.fasta`, `.faa`; the id is the first token of the header), or CSV/TSV/Parquet with an id column (`--id-col`, default `id`) and a sequence column (`--seq-col`). Ids must be present, non-empty and unique.
- Output tables (`.csv`, `.parquet`): a string `id` column followed by the feature columns; the sequence is not written.
- `.npz`: `values`, `columns` and `ids`. `.npy`: the numeric matrix only, plus `<stem>.ids.txt` with one id per line.
- Each run also writes `<output>.provenance.json` (parameters, input/output hashes, package versions; for `embed`, the model revision, device and batch size actually used).
- `sylphy embed --revision` pins the model revision (branch, tag or commit).

### Exit codes

- `0` success; `2` usage or config error; `3` runtime or data error (sylphy errors, I/O, model loading, unparsable input); `4` unexpected error (traceback on stderr).

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

## Citation

If you use Sylphy in your research, please cite:

> Medina-Ortiz, D.; Álvarez Saravia, D.; García-Vinuesa, J.
> *Sylphy: Protein Sequence Representation: Encoders, Embeddings, and Reductions*.
> Version 1.0.0, 2026. Zenodo.
> https://doi.org/10.5281/zenodo.21819735

```bibtex
@software{sylphy2026,
  author    = {Medina-Ortiz, David and Álvarez Saravia, Diego and García-Vinuesa, Julián},
  title     = {{Sylphy}: Protein Sequence Representation: Encoders, Embeddings, and Reductions},
  year      = {2026},
  version   = {1.0.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21819735},
  url       = {https://doi.org/10.5281/zenodo.21819735}
}
```

To cite this exact release, use the version DOI [10.5281/zenodo.21819736](https://doi.org/10.5281/zenodo.21819736).
Machine-readable metadata lives in [CITATION.cff](CITATION.cff).

## License

**MIT**. See [LICENSE](LICENSE).

## Acknowledgements

Built with the Hugging Face Transformers ecosystem, the Meta ESM-C SDK, and the broader scientific Python stack including scikit-learn, PyTorch, UMAP, and ClustPy.

Developed by **KREN AI Lab** at Universidad de Magallanes, Chile.
