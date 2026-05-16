# Examples

This directory contains runnable scripts for the main Sylphy workflows.

## Index

- [`1_quick_start_encoders.py`](1_quick_start_encoders.py): walkthrough of all classical sequence encoders and the encoder factory.
- [`2_simple_demo_embedding_extractor.py`](2_simple_demo_embedding_extractor.py): embedding extraction across the supported model families.
- [`3_quick_start_reduction_process.py`](3_quick_start_reduction_process.py): dimensionality reduction workflows using linear (PCA, TruncatedSVD) and non-linear (t-SNE, Isomap) methods.
- [`4_demo_embedding_different_layers.py`](4_demo_embedding_different_layers.py): layer selection, aggregation, and pooling options for ESM2 embeddings.
- [`encoder_sequences_using_sylphy.py`](encoder_sequences_using_sylphy.py): batch-oriented encoder example for training and test datasets.
- [`extract_embedding_using_sylphy.py`](extract_embedding_using_sylphy.py): batch-oriented embedding extraction example for train/test splits.

## Running Examples

```bash
uv run python examples/1_quick_start_encoders.py
uv run python examples/2_simple_demo_embedding_extractor.py
```

Embedding examples require optional dependencies and benefit from CUDA-enabled PyTorch:

```bash
pip install -e ".[embeddings]"
```
