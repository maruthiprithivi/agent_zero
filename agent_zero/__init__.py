"""Agent Zero package for ClickHouse database management."""

import warnings

# Dynamic version from setuptools-scm
try:
    from agent_zero._version import __version__
except ImportError:
    # Fallback version when not installed with setuptools-scm
    __version__ = "0.2.0"

from agent_zero.config import UnifiedConfig
from agent_zero.server import (
    create_clickhouse_client,
    run,
)

# For backward compatibility - deprecated
class _DeprecatedServerConfig:
    """Deprecated ServerConfig alias that issues warnings."""

    def __new__(cls, *args, **kwargs):
        warnings.warn(
            "ServerConfig is deprecated. Use UnifiedConfig instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return UnifiedConfig(*args, **kwargs)

ServerConfig = _DeprecatedServerConfig

__all__ = [
    "ServerConfig",  # Deprecated alias
    "UnifiedConfig",
    "create_clickhouse_client",
    "run",
]
