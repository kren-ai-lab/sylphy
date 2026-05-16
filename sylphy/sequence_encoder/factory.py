"""Create sequence encoders from canonical names or aliases."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from sylphy.core.model_registry import normalize_name
from sylphy.logging import get_child_logger

from .base_encoder import EncoderBase
from .fft_encoder import FFTEncoder
from .frequency_encoder import FrequencyEncoder
from .kmers_encoder import KMerEncoder
from .one_hot_encoder import OneHotEncoder
from .ordinal_encoder import OrdinalEncoder
from .physicochemical_encoder import PhysicochemicalEncoder

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "EncoderBase",
    "FFTEncoder",
    "FrequencyEncoder",
    "KMerEncoder",
    "OneHotEncoder",
    "OrdinalEncoder",
    "PhysicochemicalEncoder",
    "create_encoder",
]

_logger = get_child_logger("sequence_encoder.factory", component="sequence_encoder", facility="factory")

# Canonical names for encoders
_ALIASES: dict[str, str] = {
    # one-hot
    "onehot": "one_hot",
    "one_hot": "one_hot",
    # ordinal
    "ordinal": "ordinal",
    # frequency
    "frequency": "frequency",
    "freq": "frequency",
    # k-mers
    "kmers": "kmers",
    "kmer": "kmers",
    "tfidf": "kmers",
    # physicochemical
    "physicochemical": "physicochemical",
    "physchem": "physicochemical",
    "aaindex": "physicochemical",
    # fft
    "fft": "fft",
}

# Class map
EncoderInstance = EncoderBase | FFTEncoder

_CLASSES: dict[str, type[EncoderBase | FFTEncoder]] = {
    "one_hot": OneHotEncoder,
    "ordinal": OrdinalEncoder,
    "frequency": FrequencyEncoder,
    "kmers": KMerEncoder,
    "physicochemical": PhysicochemicalEncoder,
    "fft": FFTEncoder,
}

# Whitelisted kwargs per encoder (only these are forwarded)
_ALLOWED: dict[str, set[str]] = {
    "one_hot": {
        "dataset",
        "sequence_column",
        "max_length",
        "allow_extended",
        "allow_unknown",
        "debug",
        "debug_mode",
    },
    "ordinal": {
        "dataset",
        "sequence_column",
        "max_length",
        "allow_extended",
        "allow_unknown",
        "debug",
        "debug_mode",
    },
    "frequency": {"dataset", "sequence_column", "allow_extended", "allow_unknown", "debug", "debug_mode"},
    "kmers": {
        "dataset",
        "sequence_column",
        "size_kmer",
        "allow_extended",
        "allow_unknown",
        "debug",
        "debug_mode",
    },
    "physicochemical": {
        "dataset",
        "sequence_column",
        "max_length",
        "type_descriptor",
        "name_property",
        "allow_extended",
        "allow_unknown",
        "debug",
        "debug_mode",
    },
    "fft": {"dataset", "sequence_column", "debug", "debug_mode"},
}


def _canonical(name: str) -> str:
    """Resolve an encoder name or alias to its canonical key."""
    key = normalize_name(name)
    if key in _ALIASES:
        return _ALIASES[key]
    msg = (
        f"Unknown encoder '{name}'. "
        f"Available: {sorted(set(_ALIASES.values()))} (aliases supported: {sorted(_ALIASES.keys())})"
    )
    raise ValueError(
        msg,
    )


def _filter_kwargs(kind: str, kwargs: dict[str, object]) -> dict[str, object]:
    """Filter constructor kwargs to those supported by a specific encoder."""
    allowed = _ALLOWED[kind]
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    ignored = sorted(set(kwargs.keys()) - allowed)
    if ignored:
        _logger.debug("Ignoring unsupported arguments for %s: %s", kind, ignored)
    return filtered


def create_encoder(name: str, **kwargs: object) -> EncoderInstance:
    """Create a sequence encoder with backend-specific argument filtering.

    Args:
        name: Encoder name or alias.
        **kwargs: Backend-specific constructor parameters.

    Returns:
        An initialized encoder instance.

    """
    kind = _canonical(name)
    cls = _CLASSES[kind]
    params = _filter_kwargs(kind, kwargs)
    _logger.info(
        "Creating encoder kind=%s class=%s kwargs=%s",
        kind,
        cls.__name__,
        params,
        extra={"encoder": cls.__name__},
    )
    constructor = cast("Callable[..., EncoderInstance]", cls)
    return constructor(**params)
