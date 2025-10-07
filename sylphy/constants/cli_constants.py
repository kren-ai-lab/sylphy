# cli_constants.py
from enum import Enum


class EncoderType(str, Enum):
    """Sequence encoders available in the CLI."""
    onehot = "onehot"
    ordinal = "ordinal"
    fft = "fft"
    physicochemical = "physicochemical"
    frequency = "frequency"
    kmers = "kmers"


class ExportOption(str, Enum):
    """Export formats for CLI outputs."""
    csv = "csv"
    npy = "npy"
    parquet = "parquet"


class PhysicochemicalOption(str, Enum):
    """Feature sources for physicochemical encoders."""
    aaindex = "aaindex"
    group_based = "group_based"


class DebugMode(str, Enum):
    """Textual logging levels accepted by the CLI."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Device(str, Enum):
    """Compute device selection for embedding extraction."""
    cuda = "cuda"
    cpu = "cpu"


class Precision(str, Enum):
    """Floating-point precision for model execution."""
    fp32 = "fp32"
    fp16 = "fp16"
    bf16 = "bf16"


class PoolOption(str, Enum):
    """Pooling strategies for sequence/embedding features."""
    mean = "mean"
    cls = "cls"
    eos = "eos"
