"""Consistent, quiet application logging."""

import logging

def setup_logging(level: str = "INFO") -> None:
    """Configure the application console handler once."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

def get_logger(name: str) -> logging.Logger:
    """Return a standard module logger."""
    return logging.getLogger(name)
