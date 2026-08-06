"""CLI-layer enum choices for Typer option validation and help display."""

from __future__ import annotations

from enum import StrEnum


class EncodeMethod(StrEnum):
    one_hot = "one_hot"
    ordinal = "ordinal"
    frequency = "frequency"
    kmers = "kmers"
    physicochemical = "physicochemical"
    fft = "fft"


class DescriptorType(StrEnum):
    aaindex = "aaindex"
    group_based = "group_based"


class DeviceChoice(StrEnum):
    cuda = "cuda"
    cpu = "cpu"


class PrecisionChoice(StrEnum):
    fp32 = "fp32"
    fp16 = "fp16"
    bf16 = "bf16"


class PoolChoice(StrEnum):
    mean = "mean"
    cls = "cls"
    eos = "eos"


class LayerAggChoice(StrEnum):
    mean = "mean"
    sum = "sum"
    concat = "concat"
