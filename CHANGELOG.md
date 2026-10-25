# Changelog

All notable changes to Sylphy are recorded here. Versions follow
[Semantic Versioning](https://semver.org/).

## 1.2.0

Adopts the shared ecosystem CLI contract for `sylphy encode` and `sylphy embed`.

- Input may be FASTA (`.fa`, `.fasta`, `.faa`); the id is the first token of the header.
- New `--id-col` (default `id`) for tabular input. A missing, empty or duplicated id is an error (exit code 3; unsupported format or a missing `--id-col` column is 2).
- New `--config PATH` (YAML/JSON, keys are option names; defaults < file < flags) and `sylphy validate-config {encode|embed} CONFIG [--json]`.
- New `sylphy embed --revision` to pin the model revision.
- Every run writes `<output>.provenance.json`; `embed` records the requested and resolved revision, the device actually used and the effective batch size.
- Exit codes: `2` usage/config error, `3` runtime or data error (sylphy's own exceptions, `OSError`, model loading/download, unparsable input), `4` unexpected error with traceback. Previously runtime failures exited 1.
- New `sylphy.exceptions` (`SylphyError`, `DataError`, `SylphyRuntimeError`); the model registry errors now derive from `SylphyError`.
- The Python API writes the same files as the CLI through one shared writer (`sylphy.feature_io.write_features`).
- New public `sylphy.validate_config(config, target)` (and `ConfigIssue`): the check behind `validate-config`, for Python; returns a list of issues, never raises for config problems.
- A missing input data file now exits `3` (it was `2`).
- `pyyaml` is now a runtime dependency; `jsonschema` a dev dependency.

### Breaking changes

- Ids are required: tabular input without an `id` column (or `--id-col`) now fails.
- Output tables no longer contain the sequence column: `id` first, then feature columns only.
- `.npz` output gains an `ids` array.
- `.npy` output is a numeric matrix (no object dtype, no pickle) with a sibling `<stem>.ids.txt`.
- API: `export_encoder` (encoders and embeddings) now writes the CLI format (`id` first, no sequence column, `ids` in `.npz`, `<stem>.ids.txt` for `.npy`) and requires an id column in the input dataset, selected with the new keyword `id_col="id"`; a missing, empty or duplicated id raises `DataError`. `FFTEncoder` takes `id_col` in its constructor and needs the id column in its input dataset to export.
- API: `export_encoder` no longer pickles object arrays in `.npy`.
- API: `get_config().default_device` is `None` unless set, so importing sylphy no longer probes CUDA (which imported torch).

## 1.1.0

- The `embeddings` extra now is supported on Python 3.13 and 3.14
- Bump requirement of `clustpy>=0.0.3` in the `reductions` extra.
- Use only pyrefly for typechecking.

## 1.0.0

This is a breaking release: the CLI, the public class names, and the DataFrame
backend all changed.

- **Polars replaces pandas** as the DataFrame backend across encoders,
  embeddings, reductions and I/O.
- **Redesigned CLI**: `sylphy encode` and `sylphy embed` replace
  `encode-sequences` and `get-embeddings`.
- **MIT license** (previously GPL-3.0-only).
- **Python 3.13 and 3.14 support** (embeddings still require 3.11 or 3.12).

### Breaking changes

CLI:

| Before | Now |
| --- | --- |
| `sylphy encode-sequences ...` | `sylphy encode ...` |
| `sylphy get-embeddings ...` | `sylphy embed ...` |

Embedding backends dropped the `Based` infix, and modules were renamed to match:

| Before | Now |
| --- | --- |
| `EmbeddingBased` (`embedding_based`) | `EmbeddingBase` (`embedding_base`) |
| `ESMBasedEmbedding` (`esm_based`) | `ESMEmbedding` (`esm_embedding`) |
| `ESMCBasedEmbedding` (`esmc_based`) | `ESMCEmbedding` (`esmc_embedding`) |
| `Prot5Based` (`prot5_based`) | `ProtT5Embedding` (`prot_t5_embedding`) |
| `BertBasedEmbedding` (`bert_based`) | `ProtBertEmbedding` (`prot_bert_embedding`) |
| `MistralBasedEmbedding` (`mistral_based`) | `MistralEmbedding` (`mistral_embedding`) |
| `Ankh2BasedEmbedding` (`ankh2_based`) | `Ankh2Embedding` (`ankh2_embedding`) |
| `EmbeddingFactory` | `create_embedding` (function) |

Sequence encoders:

| Before | Now |
| --- | --- |
| `Encoders` (`base_encoder`) | `EncoderBase` (`encoder_base`) |
| `KMersEncoders` (`kmers_encoder`) | `KMerEncoder` (`kmer_encoder`) |

- Failures now raise exceptions instead of returning status messages. The model
  registry raises `ModelRegistryError`, `ModelNotFoundError` and
  `ModelDownloadError`.
- `pandas` is no longer a dependency.
- The `parquet` extra was removed; Parquet is supported out of the box. Output
  formats: `csv`, `parquet`, `npy`, `npz`.

### Improvements

- Faster sequence encoders (`one_hot`, `ordinal`, `frequency`, `kmers`,
  `physicochemical`, `fft`) and faster ESM-C embedding extraction.
- Model weights are stored in the standard HuggingFace cache, shared with other
  tools.
- Logging configuration reworked and exported from `sylphy.logging`.
- Docstring examples added across the public API.
- Examples converted to plain runnable scripts, executed in CI.

### Migration checklist

1. Replace `encode-sequences` / `get-embeddings` invocations with `encode` /
   `embed`.
2. Rename imported embedding and encoder classes per the tables above; swap
   `EmbeddingFactory(...)` for `create_embedding(...)`.
3. Replace pandas DataFrames passed into or read out of Sylphy with Polars
   DataFrames.
4. Drop `sylphy[parquet]` from install specs.
5. Replace status-message checks with `try/except` on the registry exceptions.

## 0.2.0

- **Breaking:** removed the `run` namespace from the CLI, i.e. from
  `sylphy get-embedding run` to `sylphy get-embedding`.
- Improved handling and error messages for optional dependencies.
- Stricter linting.

## 0.1.3

- Updated dependencies.
- Published the package on PyPI.

## 0.1.2

- Support for Ankh3.
- Updated dependencies.

## 0.1.1

- Allow `Path` on outputs.
- Fixed type checking errors.
- Updated dependencies (scipy, pyarrow).

## 0.1.0

Initial release.
