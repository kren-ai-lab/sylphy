
# Sylphy 🧬 — Protein Representation Toolkit

**Sylphy** is a lightweight Python toolkit that transforms **protein sequences** into numerical representations for machine learning and bioinformatics workflows.

It unifies three core components:

* **Classical sequence encoders** — one-hot, ordinal, frequency, k-mers (TF-IDF), physicochemical, and FFT.
* **Embedding extraction** — from state-of-the-art pretrained models (ESM2, ProtT5, ProtBERT, Ankh2, Mistral-Prot, ESM-C).
* **Dimensionality reduction** — linear and non-linear methods for downstream visualization and clustering.

> ✳️ **Design philosophy:** minimal public API, lazy loading of heavy dependencies, no side effects on import, fully testable and configurable.

---

## Table of Contents

* [Installation](#installation)
* [Quickstart](#quickstart)

  * [Logging](#logging)
  * [Sequence Encoders](#sequence-encoders)
  * [Embedding Extraction](#embedding-extraction)
  * [Dimensionality Reduction](#dimensionality-reduction)
* [Command-Line Interface (CLI)](#command-line-interface-cli)
* [Configuration and Cache](#configuration-and-cache)
* [Model Registry](#model-registry)
* [Public API](#public-api)
* [Development and Testing](#development-and-testing)
* [License](#license)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)

---

## Installation

### Requirements

* Python **3.10+**
* Optional GPU acceleration (PyTorch + CUDA)

### From Source

```bash
git clone https://github.com/KREN-AI-Lab/sylphy.git
cd sylphy
pip install -e .
```
---

## Quickstart

### Logging

All modules share a unified, hierarchical logger. Configure once at startup:

```python
from sylphy import setup_logger

setup_logger(name="sylphy", level="INFO")
```

Optional environment variable:

```bash
export SYLPHY_LOG_FILE=/tmp/sylphy.log
```

---

### Sequence Encoders

```python
import pandas as pd
from sylphy.sequence_encoder import create_encoder

df = pd.DataFrame({"sequence": ["ACD", "WYYVV", "KLMNPQ", "GGG"]})

encoder = create_encoder(
    "one_hot",                  # "ordinal", "kmers", "frequency", "physicochemical", "fft"
    dataset=df,
    sequence_column="sequence",
    max_length=1024,
    debug=True,
)

encoder.run_process()
X = encoder.coded_dataset
encoder.export_encoder("onehot.csv")
```

**FFT** encoders expect a numeric input matrix. A common workflow is:

```python
from sylphy.sequence_encoder import create_encoder

phys = create_encoder("physicochemical", dataset=df, name_property="ANDN920101")
phys.run_process()

fft = create_encoder("fft", dataset=phys.coded_dataset, sequence_column="sequence")
fft.run_process()
fft.coded_dataset.head()
```

---

### Embedding Extraction

Heavy dependencies (PyTorch, Transformers) are loaded lazily.

```python
import pandas as pd
from sylphy.embedding_extraction import create_embedding

df = pd.DataFrame({"sequence": ["MKT...", "GAVL...", "PPPP..."]})

embedder = create_embedding(
    model_name="facebook/esm2_t6_8M_UR50D",  # also: ProtT5, ProtBERT, Mistral-Prot, Ankh2, ESM-C
    dataset=df,
    column_seq="sequence",
    name_device="cuda",
    precision="fp16",
    oom_backoff=True,
    debug=True,
)

embedder.run_process(max_length=1024, batch_size=8, pool="mean")  # "mean" | "cls" | "eos"
embeddings = embedder.coded_dataset
embedder.export_encoder("embeddings.parquet")
```

**Supported model families:**
`("esm2", "ankh2", "prot_t5", "prot_bert", "mistral_prot", "esmc")`

---

### Dimensionality Reduction

```python
import numpy as np
from sylphy.reductions import reduce_dimensionality

# Suppose X is an (N, D) matrix
model, X_reduced = reduce_dimensionality(
    method="pca",            # e.g. "truncated_svd", "umap", "isomap"
    dataset=X,
    return_type="numpy",
    n_components=2,
    random_state=0,
)

print(X_reduced.shape)
```

---

## Command-Line Interface (CLI)

After installation, the CLI provides a single entrypoint:

```bash
sylphy --help
```

### Embedding Extraction

```bash
sylphy get-embedding run \
  --model facebook/esm2_t6_8M_UR50D \
  --input-data data/sequences.csv \
  --sequence-identifier sequence \
  --output out/emb_esm2.parquet \
  --device cuda --precision fp16 --batch-size 16 --pool mean
```

### Sequence Encoding

```bash
# One-hot encoding
sylphy encode-sequences run \
  --encoder one_hot \
  --input-data data/sequences.csv \
  --sequence-identifier sequence \
  --output out/onehot.csv

# Physicochemical + FFT (two-stage)
sylphy encode-sequences run \
  --encoder fft \
  --input-data data/sequences.csv \
  --sequence-identifier sequence \
  --name-property ANDN920101 \
  --output out/physchem_fft.csv
```

---

## Configuration and Cache

Model weights and intermediate files are cached locally (following OS-specific appdirs).

```python
from sylphy import get_config, set_cache_root, temporary_cache_root

cfg = get_config()
print(cfg.cache_paths.cache_root)

set_cache_root("/data/sylphy_cache")

with temporary_cache_root("/tmp/sylphy_cache"):
    ...
```

---

## Model Registry

Sylphy includes a minimal registry to manage model aliases, local overrides, and resolution.

```python
from sylphy import (
    ModelSpec, register_model, register_alias, resolve_model, list_registered_models
)

register_model(ModelSpec(name="esm2_t6", provider="huggingface", ref="facebook/esm2_t6_8M_UR50D"))
register_alias("esm2_small", "esm2_t6")

path = resolve_model("esm2_small")
print(path)
print(list_registered_models(include_aliases=True))
```

> To override a model path:
> set `SYLPHY_MODEL_<UPPERCASE_NAME>` as an environment variable
> (e.g. `SYLPHY_MODEL_ESM2_SMALL=/models/esm2_t6`).

---

## Public API

```python
from sylphy import (
    # Logging
    setup_logger, get_logger,

    # Config
    get_config, set_cache_root, temporary_cache_root,
    ModelSpec, register_model, register_alias, resolve_model,

    # Sequence encoders
    create_encoder, OneHotEncoder, OrdinalEncoder, KMersEncoder,
    PhysicochemicalEncoder, FFTEncoder, FrequencyEncoder,

    # Embeddings
    create_embedding, EmbeddingBased, SUPPORTED_FAMILIES,

    # Reductions
    reduce_dimensionality, LinearReduction, NonLinearReductions,
)
```

Subpackages also expose curated surfaces:

* `sylphy.sequence_encoder`
* `sylphy.embedding_extraction`
* `sylphy.reductions`

---

## Development and Testing

```bash
pip install -e .
pytest -q
```

The test suite runs **offline** (mocked HF and torch).
Coding style: fully type-annotated, NumPy-style docstrings, no side effects on import.

For reproducibility, set `random_state=0` in all stochastic reducers.

---

## License

Licensed under **GPL-3.0-only**.
See the `LICENSE` file for details.

---

## Acknowledgements

* Protein language models rely on the **Hugging Face ecosystem** and, optionally, **Meta’s ESM-C SDK**.
* Non-linear reducers (UMAP, Isomap, t-SNE) use **scikit-learn** and **ClustPy**.
* Developed by the **KREN AI Lab** (University of Magallanes, Chile).

---

## Contact

Maintained by **KREN AI Lab**
📧 [krenai@umag.cl](mailto:krenai@umag.cl)
