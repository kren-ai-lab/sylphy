"""
Public constants API for sylphy.

"""

# Re-export selected names from submodules to keep a clean, stable public API.

# tool_constants
from .tool_constants import (
    _ENV_PREFIX as SYLPHY_ENV_PREFIX,
    LIST_RESIDUES, POSITION_RESIDUES,
    LIST_DESCRIPTORS_SEQUENCE, LIST_DESCRIPTORS_SEQUENCE_NON_NUMERIC,
    BASE_URL_AAINDEX, BASE_URL_CLUSTERS_DESCRIPTORS,
    get_index, get_residue,
)

# config_constants
from .config_constants import CachePaths

# tool_configs
from .tool_configs import ToolConfig, get_config, set_config

# logging_constants
from .logging_constants import (
    LOG_ENV_PREFIX,
    LOG_DEFAULT_NAME, LOG_DEFAULT_LEVEL, LOG_DEFAULT_JSON, LOG_DEFAULT_STDERR,
    LOG_DEFAULT_MAX_BYTES, LOG_DEFAULT_BACKUPS,
    LOG_LEVEL_MAP, env_log_json, env_log_stderr
)

# cli_constants
from .cli_constants import (ExportOption, EncoderType, Device, 
                            DebugMode, PhysicochemicalOption, PoolOption,
                            Precision)

__all__ = [
    # tool_constants
    "SYLPHY_ENV_PREFIX",
    "LIST_RESIDUES", "POSITION_RESIDUES",
    "LIST_DESCRIPTORS_SEQUENCE", "LIST_DESCRIPTORS_SEQUENCE_NON_NUMERIC",
    "BASE_URL_AAINDEX", "BASE_URL_CLUSTERS_DESCRIPTORS",
    "get_index", "get_residue",

    # config_constants
    "CachePaths",

    # tool_configs
    "ToolConfig", "get_config", "set_config",

    # logging_constants
    "LOG_ENV_PREFIX",
    "LOG_DEFAULT_NAME", "LOG_DEFAULT_LEVEL", "LOG_DEFAULT_JSON", "LOG_DEFAULT_STDERR",
    "LOG_DEFAULT_MAX_BYTES", "LOG_DEFAULT_BACKUPS",
    "LOG_LEVEL_MAP", "env_log_json", "env_log_stderr"

    # cli_constants
    "ExportOption", "EncoderType", "DebugMode", "Device", 
    "DebugMode", "PhysicochemicalOption", "PoolOption", "Precision"
]
