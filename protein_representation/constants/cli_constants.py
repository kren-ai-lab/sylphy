from enum import Enum

class EncoderType(str, Enum):
    onehot = "onehot"
    ordinal = "ordinal"
    fft = "fft"
    physicochemical = "physicochemical"
    frequency = "frequency"
    kmers = "kmers"

class ExportOption(str, Enum):
    csv = "csv"
    npy = "npy"

class PhysicochemicalOption(str, Enum):
    aaindex = "aaindex"
    group_based = "group_based"

class DebugMode(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class Device(str, Enum):
    cuda = "cuda"
    cpu = "cpu"

class Precision(str, Enum):
    fp32 = "fp32"
    fp16 = "fp16"
    bf16 = "bf16"

class PoolOption(str, Enum):
    mean = "mean"
    cls = "cls"
    eos = "eos"