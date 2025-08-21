# protein-representation

A lightweight toolkit to turn **protein sequences** into numerical representations:

* Classical **sequence encoders** (one-hot, ordinal, k-mers TF-IDF, physicochemical, FFT)
* **Embedding extraction** from pretrained PLMs (ESM2, ProtT5, ProtBERT, Mistral-Prot, Ankh2, ESM-C)
* **Dimensionality reductions** (linear & non-linear) for downstream analysis
* A small, consistent **logging** facade and **model registry/cache** layer

> Design goals: thin public API, **lazy loading** of heavy deps, no side-effects on import, testable & configurable.

---

## Table of contents

* [Installation](#installation)
* [Quickstart](#quickstart)

  * [Logging (one-liner)](#logging-one-liner)
  * [Sequence encoders](#sequence-encoders)
  * [Embedding extraction](#embedding-extraction)
  * [Dimensionality reduction](#dimensionality-reduction)
* [Command-line interface (CLI)](#command-line-interface-cli)
* [Configuration & cache](#configuration--cache)
* [Model registry](#model-registry)
* [Public API](#public-api)
* [Development & testing](#development--testing)
* [License](#license)

---

## Installation

### Python

* Python **3.10+**

### From PyPI (recommended)

Core library:

```bash
pip install protein-representation
```

### From source

```bash
git clone https://github.com/ProteinEngineering-PESB2/protein_representations.git
cd protein-representation
pip install -e ".[embeddings,reductions,cli]"
```

> The package name on PyPI is `protein-representation` (import is `protein_representation`).

---

## Quickstart

### Logging (one-liner)

Configure once at program start; all components use hierarchical child loggers.

```python
from protein_representation import setup_logger

# Console INFO, file DEBUG (default path uses OS appdirs)
setup_logger(name="protein_representation", level="INFO")
```

You can also direct logs to a file via environment variable (optional):

```bash
export PR_LOG_FILE=/tmp/protein-repr.log
```

---

### Sequence encoders

```python
import pandas as pd
from protein_representation.sequence_encoder import create_encoder

df = pd.DataFrame({"sequence": ["ACD", "WYYVV", "KLMNPQ", "GGG"]})

enc = create_encoder(
    "one_hot",                # "ordinal", "kmers", "physicochemical", "frequency", "fft"
    dataset=df,
    sequence_column="sequence",
    max_length=1024,
    debug=True,
)

enc.run_process()
X = enc.coded_dataset              # pandas.DataFrame (features + 'sequence')
enc.export_encoder("onehot.csv")   # or .npy via file_format="npy"
```

**FFT** encoder expects a numeric matrix; a common pattern is physicochemical → FFT:

```python
from protein_representation.sequence_encoder import FFTEncoder, create_encoder

phys = create_encoder("physicochemical", dataset=df, name_property="ANDN920101")
phys.run_process()

fft = FFTEncoder(dataset=phys.coded_dataset, sequence_column="sequence", debug=True)
fft.run_process()
fft.coded_dataset.head()
```

---

### Embedding extraction

All heavy deps are lazily loaded. Use the factory to select the backend automatically.

```python
import pandas as pd
from protein_representation.embedding_extraction import create_embedding

df = pd.DataFrame({"sequence": ["MKT...", "GAVL...", "PPPP..."]})

embedder = create_embedding(
    model_name="facebook/esm2_t6_8M_UR50D",  # also supports ProtT5, ProtBERT, Mistral-Prot, Ankh2, ESM-C
    dataset=df,
    column_seq="sequence",
    name_device="cuda",          # or "cpu"
    precision="fp16",            # "fp32"|"fp16"|"bf16" (CUDA only)
    oom_backoff=True,            # halve batch size on CUDA OOM
    debug=True,
)

embedder.load_hf_tokenizer_and_model()
embedder.run_process(max_length=1024, batch_size=8, pool="mean")  # "mean"|"cls"|"eos"
emb = embedder.coded_dataset
embedder.export_encoder("embeddings.csv")
```

**Supported families** (factory keys/aliases): `("esm2", "ankh2", "prot_t5", "prot_bert", "mistral_prot", "esmc")`.

---

### Dimensionality reduction

```python
import numpy as np
from protein_representation.reductions import reduce_dimensionality

# Suppose `X` is a (N, D) numpy array
model, Xp = reduce_dimensionality(
    method="pca",         # e.g., "truncated_svd", "nmf", "isomap", "umap"
    dataset=X,
    return_type="numpy",  # or "pandas"
    n_components=2,
    random_state=0,
)

print(Xp.shape)  # (N, 2)
```

---

## Command-line interface (CLI)

After installing the `cli` extra, you’ll have a single entrypoint:

```bash
protein-representation --help
```

### Extract embeddings

```bash
protein-representation get-embedding run \
  --model facebook/esm2_t6_8M_UR50D \
  --input-data data/sequences.csv \
  --sequence-identifier sequence \
  --output out/emb_esm2.csv \
  --device cuda --precision fp16 --batch-size 16 --pool mean
```

### Encode sequences

```bash
# One-hot
protein-representation encode-sequences run \
  --encoder onehot \
  --input-data data/sequences.csv \
  --sequence-identifier sequence \
  --output out/onehot.csv

# Physicochemical + FFT (two-stage)
protein-representation encode-sequences run \
  --encoder fft \
  --input-data data/sequences.csv \
  --sequence-identifier sequence \
  --name-property ANDN920101 \
  --output out/physchem_fft.csv
```

### Reduce embeddings

```bash
protein-representation reduce run \
  --input out/emb_esm2.csv \
  --method pca \
  --n-components 2 \
  --return-type pandas \
  --out out/emb_esm2_pca.csv
```

> Use `protein-representation reduce run --list-methods` to see all available methods (linear & non-linear).

---

## Configuration & cache

The library keeps a cache for downloaded models / aux files.

```python
from protein_representation import get_config, set_cache_root, temporary_cache_root

cfg = get_config()
print(cfg.cache_paths.cache_root)   # OS-specific appdir

# Change cache root permanently:
set_cache_root("/data/protein_cache")

# Or temporarily within a context:
with temporary_cache_root("/tmp/pr_cache"):
    ...
```

---

## Model registry

A tiny registry sits in front of providers (e.g., Hugging Face Hub) so you can pin names, create aliases, and resolve to local paths.

```python
from protein_representation import (
  ModelSpec, register_model, register_alias, resolve_model, list_registered_models
)

register_model(ModelSpec(name="esm2_t6", provider="huggingface", ref="facebook/esm2_t6_8M_UR50D"))
register_alias("esm2_small", "esm2_t6")

local_dir = resolve_model("esm2_small")
print(local_dir)
print(list_registered_models(include_aliases=True))
```

**Environment override** (per-model): set `PR_MODEL_<UPPERCASE_NAME>` to point to a local directory, and the resolver will use that path instead of downloading (e.g., `PR_MODEL_ESM2_SMALL=/models/esm2_t6`).

---

## Public API

Top-level (lazy-loaded) imports for convenience:

```python
from protein_representation import (
  # logging
  setup_logger, get_logger, add_context,

  # core
  get_config, set_cache_root, temporary_cache_root,
  ModelSpec, register_model, register_alias, resolve_model,

  # sequence encoders
  create_encoder, OrdinalEncoder, OneHotEncoder, KMersEncoders,
  PhysicochemicalEncoder, FFTEncoder, FrequencyEncoder,

  # embeddings
  create_embedding, EmbeddingBased, SUPPORTED_FAMILIES,

  # reductions
  reduce_dimensionality, LinearReduction, NonLinearReductions,
)
```

The subpackages also expose their own curated surfaces:

* `protein_representation.sequence_encoder`
* `protein_representation.embedding_extraction`
* `protein_representation.reductions`

---

## Development & testing

We use `pytest`. The test suite is offline by default (fakes/mocks for heavy deps and network).

```bash
pip install -e .
pytest -q
```

Coding style: type-annotated Python, NumPy-style docstrings, no side-effects on import.
For reproducible reductions, prefer to pass `random_state=0` where applicable.

---

## License

This project is licensed under **GPL-3.0-only**.
See `LICENSE` for full text.

---

## Acknowledgements

* Protein language model backends rely on the Hugging Face ecosystem and, optionally, Meta’s ESM-C SDK.
* Some non-linear reductions use UMAP and ClustPy.

---

### Contact

Maintained by **Kren AI Lab** — contributions and PRs are welcome! [Concat us!](mailto:krenai@umag.cl)

