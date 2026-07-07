"""
pipeline/logging_utils.py

Configuración centralizada de logging para todo el pipeline. Registra en
consola y en results/pipeline.log simultáneamente.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOGGER_NAME = "forecast_pipeline"


def setup_logger(log_path: Path, level: int = logging.INFO) -> logging.Logger:
    """
    Configura (una sola vez) el logger del pipeline con dos handlers:
    consola (stdout) y archivo (log_path). Llamadas subsecuentes a
    get_logger() reutilizan la misma configuración.
    """
    logger = logging.getLogger(_LOGGER_NAME)

    if logger.handlers:
        # Ya configurado (ej. si setup_logger se llama más de una vez en
        # el mismo proceso) — no duplicar handlers.
        return logger

    logger.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    return logger


def get_logger() -> logging.Logger:
    """Devuelve el logger del pipeline (debe llamarse setup_logger antes)."""
    return logging.getLogger(_LOGGER_NAME)
