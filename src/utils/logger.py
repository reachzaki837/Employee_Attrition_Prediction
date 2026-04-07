"""Structured logging utility for PulseML."""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Create logs directory if it doesn't exist
LOGS_DIR = Path(__file__).parent.parent.parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS_DIR / "app.log"


class JsonFormatter(logging.Formatter):
    """Format logs as JSON for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            log_obj.update(record.extra_fields)

        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj)


class PulseMLLogger(logging.Logger):
    """Custom logger with extra fields support."""

    def _log(
        self,
        level: int,
        msg: str,
        args: tuple,
        exc_info: Any = None,
        extra: Optional[Dict[str, Any]] = None,
        stack_info: bool = False,
    ):
        """Log with extra fields."""
        if extra:
            # Create a record and attach extra fields
            record = self.makeRecord(
                self.name,
                level,
                "(unknown file)",
                0,
                msg,
                args,
                exc_info,
                func=None,
                extra=extra,
                sinfo=None,
            )
            record.extra_fields = extra
        else:
            record = None

        # Call parent logger
        super()._log(level, msg, args, exc_info, extra, stack_info)


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger for the application.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger instance
    """
    logging.setLoggerClass(PulseMLLogger)
    logger = logging.getLogger(name)

    # Only configure once
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Console handler (human-readable)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "[%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)

    # File handler (JSON format)
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    json_formatter = JsonFormatter()
    file_handler.setFormatter(json_formatter)

    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
