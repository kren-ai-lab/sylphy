"""
Miscellaneous Utilities for Sequence/Embedding Workflows
========================================================

This package contains utility abstractions shared across the project such as:
random/stratified sampling, distance computations, filesystem helpers, and export
functions.

Modules
-------
- utils_lib : High-level utilities for sampling, distances, job IDs, safe deletion, and export.

Author: KREN AI LAB
License: GNU GENERAL PUBLIC LICENSE
"""

from .utils_lib import UtilsLib

__all__ = ["UtilsLib"]
__version__ = "1.0.0"
__author__ = "KREN AI LAB"
__email__ = "krenai@umag.cl"
__license__ = "GNU GENERAL PUBLIC LICENSE"
