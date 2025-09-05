from .tool_configs import ToolConfig
import logging

# ----------------------------
# Defaults & Env Overrides
# ----------------------------

_ENV_PREFIX = "PR_LOG_"  # e.g., PR_LOG_LEVEL, PR_LOG_JSON

_DEF_NAME = "protein_representation"
_DEF_LEVEL = ToolConfig.log_level
_DEF_JSON = False
_DEF_STDERR = False  # default to stdout unless overridden
_DEF_MAX_BYTES = 10 * 1024 * 1024  # 10 MiB per file
_DEF_BACKUPS = 3

# Accept common textual levels
_LEVEL_MAP = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}
