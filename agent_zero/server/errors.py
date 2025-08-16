"""Custom error types for MCP tools.

These errors provide structured context for downstream tooling while
remaining optional (gated by configuration to preserve legacy behavior).
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class MCPToolError(Exception):
    code: str
    message: str
    context: dict[str, Any] | None = None
    retriable: bool = False

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"
