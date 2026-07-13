"""Настройка логирования.

Единая точка конфигурации логов для всех модулей конвейера: вывод в консоль
и в файл ``logs/stocker.log`` с ротацией. Вызывается один раз на старте
(идемпотентно — повторный вызов не плодит обработчики).
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(logs_dir: Path, level: str = "INFO") -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    # Идемпотентность: не добавляем обработчики повторно.
    if getattr(root, "_stocker_configured", False):
        return

    formatter = logging.Formatter(_LOG_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        logs_dir / "stocker.log",
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    root._stocker_configured = True  # type: ignore[attr-defined]
