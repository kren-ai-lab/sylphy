# Examples

This directory contains runnable examples and notebooks for the main Sylphy workflows.

## Index

- [`1_quick_start_encoders.ipynb`](1_quick_start_encoders.ipynb): notebook walkthrough of the classical sequence encoders.
- [`2_simple_demo_embedding_extractor.py`](2_simple_demo_embedding_extractor.py): script demonstrating embedding extraction across the supported model families.
- [`3_quick_start_reduction_process.ipynb`](3_quick_start_reduction_process.ipynb): notebook focused on dimensionality reduction workflows.
- [`4_demo_embedding_different_layers.py`](4_demo_embedding_different_layers.py): script showing layer selection, aggregation, and pooling options for ESM2 embeddings.
- [`encoder_sequences_using_sylphy.py`](encoder_sequences_using_sylphy.py): batch-oriented encoder example for training and test datasets.
- [`extract_embedding_using_sylphy.py`](extract_embedding_using_sylphy.py): batch-oriented embedding extraction example for train/test splits.

## Running Examples

Python scripts:

```bash
python examples/2_simple_demo_embedding_extractor.py
```

Jupyter notebooks:

```bash
jupyter notebook examples/1_quick_start_encoders.ipynb
```

Some embedding examples require optional dependencies and, in practice, benefit from CUDA-enabled PyTorch.
