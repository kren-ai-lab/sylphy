"""Sylphy's own exception types (mapped to exit code 3 by the CLI)."""

from __future__ import annotations


class SylphyError(Exception):
    """Base class for runtime and data errors raised by sylphy."""


class DataError(SylphyError, ValueError):
    """Invalid input data, e.g. a missing, empty or duplicated record id."""


class SylphyRuntimeError(SylphyError, RuntimeError):
    """A model could not be loaded or an extraction run failed."""
