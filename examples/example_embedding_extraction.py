#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example: Embedding extraction from protein sequences or a FASTA file.

Usage:
  python examples/example_embedding_extraction.py \
      --model esm2 \
      --out ./artifacts/embeddings.npy \
      --cache-dir ./cache \
      --fasta ./data/demo.fasta

If no --fasta and no --seq are provided, uses two demo sequences.
"""

from __future__ import annotations
import argparse
import os
import sys
import numpy as np

def _load_sequences(args: argparse.Namespace) -> list[str]:
    seqs = []
    if args.fasta:
        # Minimal FASTA reader
        with open(args.fasta, "r") as fh:
            acc = []
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if acc:
                        seqs.append("".join(acc))
                        acc = []
                else:
                    acc.append(line)
            if acc:
                seqs.append("".join(acc))
    if args.seq:
        seqs.extend(args.seq)
    if not seqs:
        # Two short demo sequences (Calmodulin fragments)
        seqs = [
            "MADQLTEEQIAEFKEAFSLFDKDGDGTITTKELGTVMRSLGQNPTEAELQDMINEVDADGNGTIDFPEFLTMMARK",
            "MKTLLLTLVVVTIVCLDLGYT"
        ]
    return seqs

def _resolve_model_cls(model_name: str):
    """
    Map a friendly --model name to a Sylphy embedding class.
    """
    name = model_name.lower()
    if name == "esm2":
        from sylphy.embedding_extractor import ESMBasedEmbedding
        return ESMBasedEmbedding
    if name in {"protT5", "prott5", "prott5-xl"}:
        from sylphy.embedding_extractor import Prot5Based
        return Prot5Based
    # Fallback / extend here with more models if needed
    raise ValueError(f"Unknown model '{model_name}'. Try 'esm2' or 'protT5'.")

def main():
    parser = argparse.ArgumentParser(description="Extract protein embeddings with Sylphy.")
    parser.add_argument("--model", type=str, default="esm2",
                        help="Embedding model to use (e.g., 'esm2', 'protT5').")
    parser.add_argument("--fasta", type=str, default=None, help="Path to FASTA file with sequences.")
    parser.add_argument("--seq", type=str, nargs="*", help="Raw sequences provided inline.")
    parser.add_argument("--out", type=str, default="./artifacts/embeddings.npy", help="Output .npy file.")
    parser.add_argument("--out-csv", type=str, default=None, help="Optional CSV path to save embeddings.")
    parser.add_argument("--cache-dir", type=str, default="./cache", help="Cache directory for models.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for encoding.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    seqs = _load_sequences(args)
    ModelCls = _resolve_model_cls(args.model)
    try:
        model = ModelCls(cache_dir=args.cache_dir, batch_size=args.batch_size)
    except TypeError:
        # If your class signature differs, fallback to minimal
        model = ModelCls()

    print(f"[info] Encoding {len(seqs)} sequences with model '{args.model}'...")
    X = model.encode(seqs)
    print(f"[info] Embeddings shape: {getattr(X, 'shape', None)}")

    np.save(args.out, X)
    print(f"[ok] Saved embeddings to: {args.out}")

    if args.out_csv:
        # Flatten per-sequence embeddings if necessary
        arr = X
        if arr.ndim == 3:
            # e.g., (N, L, D) -> mean-pool -> (N, D)
            arr = arr.mean(axis=1)
        np.savetxt(args.out_csv, arr, delimiter=",")
        print(f"[ok] Saved CSV to: {args.out_csv}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)
