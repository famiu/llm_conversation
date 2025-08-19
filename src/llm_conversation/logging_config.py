"""Logging configuration module for LLM Conversation package."""

import logging
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler


def setup_logging() -> None:
    """Set up logging configuration based on environment variables.

    Env vars:
        LLM_CONVERSATION_LOG_LEVEL: Log level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL) (default: disabled)
        LLM_CONVERSATION_LOG_FILE: Log file path (default: stderr only)
    """
    log_level_str = os.getenv("LLM_CONVERSATION_LOG_LEVEL")

    if not log_level_str:
        return  # logging disabled unless explicitly set

    try:
        log_level = getattr(logging, log_level_str.upper())
    except AttributeError:
        print(f"Invalid log level: {log_level_str}. Logging will be disabled.", file=sys.stderr)
        return

    log_file_str = os.getenv("LLM_CONVERSATION_LOG_FILE")
    log_file = Path(log_file_str) if log_file_str else None

    logger = logging.getLogger("llm_conversation")
    logger.setLevel(log_level)
    logger.handlers.clear()
    logger.propagate = False  # don’t bubble up to root

    # Console output handler using Rich
    rich_handler = RichHandler(
        console=Console(stderr=True),
        show_time=True,
        show_level=True,
        show_path=True,
        markup=True,
        rich_tracebacks=True,
    )
    logger.addHandler(rich_handler)

    # File handler (if log file is specified)
    if log_file:
        if log_file.is_dir():
            raise ValueError(f"Log file path {log_file} is a directory, not a file.")

        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter(fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.

    Args:
        name: Name for the logger, typically __name__

    Returns:
        Logger instance
    """
    # Prevent a "No handlers could be found" warning and ensure the program doesn't output logs unless configured.
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.addHandler(logging.NullHandler())

    return logger
