# __init__.py
"""Public constants API for sylphy.constants.

This module re-exports selected names to provide a clean and stable surface.
"""

# tool_constants
# config_constants
from .config_constants import CachePaths

# logging_constants
from .logging_constants import (
    LOG_DEFAULT_BACKUPS,
    LOG_DEFAULT_JSON,
    LOG_DEFAULT_LEVEL,
    LOG_DEFAULT_MAX_BYTES,
    LOG_DEFAULT_NAME,
    LOG_DEFAULT_STDERR,
    LOG_ENV_PREFIX,
    LOG_LEVEL_MAP,
    env_log_json,
    env_log_level,  # exported for CLI/apps
    env_log_stderr,
)

# tool_configs
from .tool_configs import ToolConfig, get_config, set_config
from .tool_constants import (
    _ENV_PREFIX as SYLPHY_ENV_PREFIX,
)
from .tool_constants import (
    BASE_URL_AAINDEX,
    BASE_URL_CLUSTERS_DESCRIPTORS,
    LIST_DESCRIPTORS_SEQUENCE,
    LIST_DESCRIPTORS_SEQUENCE_NON_NUMERIC,
    LIST_RESIDUES,
    LIST_RESIDUES_EXTENDED,
    get_index,
    get_residue,
    position_residues,
    residues,
)

__all__ = [
    "BASE_URL_AAINDEX",
    "BASE_URL_CLUSTERS_DESCRIPTORS",
    "LIST_DESCRIPTORS_SEQUENCE",
    "LIST_DESCRIPTORS_SEQUENCE_NON_NUMERIC",
    "LIST_RESIDUES",
    "LIST_RESIDUES_EXTENDED",
    "LOG_DEFAULT_BACKUPS",
    "LOG_DEFAULT_JSON",
    "LOG_DEFAULT_LEVEL",
    "LOG_DEFAULT_MAX_BYTES",
    "LOG_DEFAULT_NAME",
    "LOG_DEFAULT_STDERR",
    "LOG_ENV_PREFIX",
    "LOG_LEVEL_MAP",
    "SYLPHY_ENV_PREFIX",
    "CachePaths",
    "ToolConfig",
    "env_log_json",
    "env_log_level",
    "env_log_stderr",
    "get_config",
    "get_index",
    "get_residue",
    "position_residues",
    "residues",
    "set_config",
]
