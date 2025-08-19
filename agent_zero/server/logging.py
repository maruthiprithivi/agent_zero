"""Production-grade structured logging for Agent Zero MCP Server.

This module implements comprehensive structured logging following 2025 best practices
with correlation IDs, contextual information, and proper log aggregation support.
"""

import logging
import os
import sys
import uuid
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

# Optional imports for structured logging
try:
    import structlog

    structlog_available = True
except ImportError:
    structlog_available = False
    structlog = None

# Context variable for correlation IDs
correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)
request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
user_id: ContextVar[str | None] = ContextVar("user_id", default=None)


class LogLevel(Enum):
    """Log level enumeration."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class LogContext:
    """Structured log context information."""

    correlation_id: str | None = None
    request_id: str | None = None
    user_id: str | None = None
    component: str | None = None
    operation: str | None = None
    tool_name: str | None = None
    query_id: str | None = None
    client_ip: str | None = None
    user_agent: str | None = None
    session_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class LogEntry:
    """Structured log entry."""

    timestamp: str
    level: str
    message: str
    logger_name: str
    context: LogContext
    exception: dict[str, Any] | None = None
    performance: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        entry = {
            "timestamp": self.timestamp,
            "level": self.level,
            "message": self.message,
            "logger": self.logger_name,
            **self.context.to_dict(),
        }

        if self.exception:
            entry["exception"] = self.exception

        if self.performance:
            entry["performance"] = self.performance

        return entry


class StructuredLogger:
    """Enhanced structured logger with correlation ID support."""

    def __init__(self, name: str, config: dict[str, Any] | None = None):
        """Initialize structured logger.

        Args:
            name: Logger name
            config: Logger configuration
        """
        self.name = name
        self.config = config or {}
        self._setup_logger()

    def _setup_logger(self) -> None:
        """Setup the underlying logger."""
        self.logger = logging.getLogger(self.name)

        # Set log level
        level_str = self.config.get("level", "INFO").upper()
        level = getattr(logging, level_str, logging.INFO)
        self.logger.setLevel(level)

    def _get_current_context(self) -> LogContext:
        """Get current log context from context variables."""
        return LogContext(
            correlation_id=correlation_id.get(),
            request_id=request_id.get(),
            user_id=user_id.get(),
            component=self.config.get("component"),
        )

    def _create_log_entry(
        self,
        level: str,
        message: str,
        context: LogContext | None = None,
        exception: Exception | None = None,
        extra: dict[str, Any] | None = None,
    ) -> LogEntry:
        """Create a structured log entry.

        Args:
            level: Log level
            message: Log message
            context: Additional context
            exception: Exception information
            extra: Extra fields

        Returns:
            Structured log entry
        """
        # Merge contexts
        current_context = self._get_current_context()
        if context:
            # Update with provided context
            for field_name, field_value in context.to_dict().items():
                if field_value is not None:
                    setattr(current_context, field_name, field_value)

        # Add extra fields
        if extra:
            current_context.extra.update(extra)

        # Format exception information
        exception_info = None
        if exception:
            exception_info = {
                "type": type(exception).__name__,
                "message": str(exception),
                "module": getattr(exception, "__module__", None),
            }

            # Add traceback in development
            if self.config.get("include_traceback", False):
                import traceback

                exception_info["traceback"] = traceback.format_exc()

        return LogEntry(
            timestamp=datetime.now(UTC).isoformat() + "Z",
            level=level.upper(),
            message=message,
            logger_name=self.name,
            context=current_context,
            exception=exception_info,
        )

    def _emit_log(self, entry: LogEntry) -> None:
        """Emit log entry to configured handlers.

        Args:
            entry: Log entry to emit
        """
        # Convert to dictionary for structured output
        log_dict = entry.to_dict()

        # Use structured logging if available
        if structlog_available and self.config.get("use_structlog", True):
            # Create structlog logger
            struct_logger = structlog.get_logger(self.name)

            # Extract message and extra fields
            message = log_dict.pop("message")
            level = log_dict.pop("level").lower()

            # Log with appropriate level
            getattr(struct_logger, level)(message, **log_dict)
        else:
            # Fall back to standard logging with JSON formatting
            import json

            # Format as JSON for log aggregation
            json_msg = json.dumps(log_dict, default=str)

            # Map level to logging level
            level_mapping = {
                "DEBUG": logging.DEBUG,
                "INFO": logging.INFO,
                "WARNING": logging.WARNING,
                "ERROR": logging.ERROR,
                "CRITICAL": logging.CRITICAL,
            }

            log_level = level_mapping.get(entry.level, logging.INFO)
            self.logger.log(log_level, json_msg)

    def debug(self, message: str, context: LogContext | None = None, **kwargs) -> None:
        """Log debug message."""
        entry = self._create_log_entry("DEBUG", message, context, extra=kwargs)
        self._emit_log(entry)

    def info(self, message: str, context: LogContext | None = None, **kwargs) -> None:
        """Log info message."""
        entry = self._create_log_entry("INFO", message, context, extra=kwargs)
        self._emit_log(entry)

    def warning(self, message: str, context: LogContext | None = None, **kwargs) -> None:
        """Log warning message."""
        entry = self._create_log_entry("WARNING", message, context, extra=kwargs)
        self._emit_log(entry)

    def error(
        self,
        message: str,
        context: LogContext | None = None,
        exception: Exception | None = None,
        **kwargs,
    ) -> None:
        """Log error message."""
        entry = self._create_log_entry("ERROR", message, context, exception, extra=kwargs)
        self._emit_log(entry)

    def critical(
        self,
        message: str,
        context: LogContext | None = None,
        exception: Exception | None = None,
        **kwargs,
    ) -> None:
        """Log critical message."""
        entry = self._create_log_entry("CRITICAL", message, context, exception, extra=kwargs)
        self._emit_log(entry)

    def log_performance(
        self, operation: str, duration_ms: float, context: LogContext | None = None, **kwargs
    ) -> None:
        """Log performance metrics.

        Args:
            operation: Operation name
            duration_ms: Duration in milliseconds
            context: Additional context
            **kwargs: Extra performance metrics
        """
        perf_context = context or LogContext()
        perf_context.operation = operation

        performance_data = {"duration_ms": duration_ms, **kwargs}

        entry = self._create_log_entry(
            "INFO", f"Performance: {operation} completed in {duration_ms:.2f}ms", perf_context
        )
        entry.performance = performance_data
        self._emit_log(entry)

    def log_audit(
        self, action: str, resource: str, outcome: str, context: LogContext | None = None, **kwargs
    ) -> None:
        """Log audit events.

        Args:
            action: Action performed
            resource: Resource affected
            outcome: Outcome of the action
            context: Additional context
            **kwargs: Extra audit fields
        """
        audit_context = context or LogContext()
        audit_data = {
            "audit": True,
            "action": action,
            "resource": resource,
            "outcome": outcome,
            **kwargs,
        }

        entry = self._create_log_entry(
            "INFO", f"Audit: {action} on {resource} - {outcome}", audit_context, extra=audit_data
        )
        self._emit_log(entry)


class LoggingManager:
    """Manages logging configuration and loggers."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize logging manager.

        Args:
            config: Logging configuration
        """
        self.config = config or {}
        self.loggers: dict[str, StructuredLogger] = {}
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup global logging configuration."""
        # Configure structlog if available
        if structlog_available and self.config.get("use_structlog", True):
            self._setup_structlog()
        else:
            self._setup_standard_logging()

    def _setup_structlog(self) -> None:
        """Setup structlog configuration."""
        processors = [
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
        ]

        # Add development vs production processors
        if self.config.get("development", False):
            processors.extend([structlog.dev.ConsoleRenderer(colors=True)])
        else:
            processors.extend(
                [structlog.processors.dict_tracebacks, structlog.processors.JSONRenderer()]
            )

        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(
                getattr(logging, self.config.get("level", "INFO").upper())
            ),
            logger_factory=structlog.WriteLoggerFactory(),
            cache_logger_on_first_use=True,
        )

    def _setup_standard_logging(self) -> None:
        """Setup standard logging configuration."""
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.get("level", "INFO").upper()))

        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Add console handler
        console_handler = logging.StreamHandler(sys.stdout)

        # Use JSON formatter for production
        if self.config.get("json_format", True):
            formatter = JsonFormatter()
        else:
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # Add file handler if configured
        log_file = self.config.get("log_file")
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

    def get_logger(self, name: str) -> StructuredLogger:
        """Get or create a structured logger.

        Args:
            name: Logger name

        Returns:
            Structured logger instance
        """
        if name not in self.loggers:
            self.loggers[name] = StructuredLogger(name, self.config)
        return self.loggers[name]

    def set_correlation_id(self, correlation_id_value: str | None = None) -> str:
        """Set correlation ID for current context.

        Args:
            correlation_id_value: Correlation ID value, generates one if None

        Returns:
            The correlation ID that was set
        """
        if correlation_id_value is None:
            correlation_id_value = str(uuid.uuid4())

        correlation_id.set(correlation_id_value)
        return correlation_id_value

    def set_request_id(self, request_id_value: str | None = None) -> str:
        """Set request ID for current context.

        Args:
            request_id_value: Request ID value, generates one if None

        Returns:
            The request ID that was set
        """
        if request_id_value is None:
            request_id_value = str(uuid.uuid4())

        request_id.set(request_id_value)
        return request_id_value

    def set_user_id(self, user_id_value: str) -> None:
        """Set user ID for current context.

        Args:
            user_id_value: User ID value
        """
        user_id.set(user_id_value)

    def clear_context(self) -> None:
        """Clear all context variables."""
        correlation_id.set(None)
        request_id.set(None)
        user_id.set(None)


class JsonFormatter(logging.Formatter):
    """JSON formatter for standard logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.

        Args:
            record: Log record

        Returns:
            JSON formatted log message
        """
        import json

        log_entry = {
            "timestamp": datetime.now(UTC).isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add context variables
        if correlation_id.get():
            log_entry["correlation_id"] = correlation_id.get()
        if request_id.get():
            log_entry["request_id"] = request_id.get()
        if user_id.get():
            log_entry["user_id"] = user_id.get()

        # Add exception information
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }

        return json.dumps(log_entry, default=str)


def configure_logging(config: dict[str, Any] | None = None) -> LoggingManager:
    """Configure global logging.

    Args:
        config: Logging configuration

    Returns:
        Logging manager instance
    """
    # Default configuration
    default_config = {
        "level": os.getenv("LOG_LEVEL", "INFO"),
        "use_structlog": structlog_available,
        "json_format": os.getenv("LOG_FORMAT", "json").lower() == "json",
        "development": os.getenv("ENVIRONMENT", "production").lower() == "development",
        "include_traceback": os.getenv("LOG_INCLUDE_TRACEBACK", "false").lower() == "true",
        "log_file": os.getenv("LOG_FILE"),
    }

    # Merge with provided config
    if config:
        default_config.update(config)

    return LoggingManager(default_config)


# Global logging manager
logging_manager: LoggingManager | None = None


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name

    Returns:
        Structured logger instance
    """
    global logging_manager
    if logging_manager is None:
        logging_manager = configure_logging()

    return logging_manager.get_logger(name)


def set_correlation_id(correlation_id_value: str | None = None) -> str:
    """Set correlation ID for current context.

    Args:
        correlation_id_value: Correlation ID value

    Returns:
        The correlation ID that was set
    """
    global logging_manager
    if logging_manager is None:
        logging_manager = configure_logging()

    return logging_manager.set_correlation_id(correlation_id_value)


def get_correlation_id() -> str | None:
    """Get current correlation ID.

    Returns:
        Current correlation ID or None
    """
    return correlation_id.get()


def reset_logging_manager() -> None:
    """Reset the global logging manager (useful for testing)."""
    global logging_manager
    logging_manager = None
