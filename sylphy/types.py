from __future__ import annotations

from typing import Literal

PrecisionType = Literal["fp32", "fp16", "bf16"]
PoolType = Literal["mean", "cls", "eos"]
LayerAggType = Literal["mean", "sum", "concat"]
FileFormat = Literal["csv", "npy", "npz", "parquet"]
