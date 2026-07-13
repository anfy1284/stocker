"""Командная строка Stocker.

Единая точка запуска модулей конвейера. На Шаге 1 доступны:
  * ``init``   — создать каталоги и пустую БД (действие по умолчанию);
  * ``config`` — показать текущую конфигурацию (без утечки ключа).
"""

from __future__ import annotations

import argparse
import logging
import sys

from .config import Config, load_config
from .db import init_db
from .logging_setup import setup_logging

log = logging.getLogger("stocker")


def _force_utf8_streams() -> None:
    """Выводит кириллицу без искажений в консоли Windows (cp1251/cp866)."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


def cmd_init(cfg: Config) -> int:
    cfg.ensure_dirs()
    init_db(cfg.db_path)
    log.info("Инициализация завершена. Входящие: %s", cfg.inbox_dir)
    if not cfg.has_api_key:
        log.warning(
            "ANTHROPIC_API_KEY не задан — классификация (Шаг 3) работать не будет."
        )
    return 0


def cmd_config(cfg: Config) -> int:
    for key, value in cfg.as_display_dict().items():
        print(f"{key:16} {value}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stocker",
        description="Отбор и загрузка фотографий на микростоки.",
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("init", help="Создать каталоги и пустую БД (по умолчанию).")
    sub.add_parser("config", help="Показать текущую конфигурацию.")
    return parser


def main(argv: list[str] | None = None) -> int:
    _force_utf8_streams()
    args = build_parser().parse_args(argv)

    cfg = load_config()
    setup_logging(cfg.logs_dir, cfg.log_level)

    command = args.command or "init"
    if command == "config":
        return cmd_config(cfg)
    return cmd_init(cfg)
