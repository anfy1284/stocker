"""Командная строка Stocker.

Единая точка запуска модулей конвейера. На Шаге 1 доступны:
  * ``init``   — создать каталоги и пустую БД (действие по умолчанию);
  * ``config`` — показать текущую конфигурацию (без утечки ключа).
"""

from __future__ import annotations

import argparse
import logging
import sys

from .classifier import run_classification
from .config import Config, load_config
from .db import init_db
from .intake import run_intake
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


def cmd_intake(cfg: Config) -> int:
    cfg.ensure_dirs()
    init_db(cfg.db_path)
    run_intake(cfg)
    return 0


def cmd_classify(cfg: Config) -> int:
    cfg.ensure_dirs()
    init_db(cfg.db_path)
    try:
        run_classification(cfg)
    except RuntimeError as exc:
        log.error("%s", exc)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stocker",
        description="Отбор и загрузка фотографий на микростоки.",
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("init", help="Создать каталоги и пустую БД (по умолчанию).")
    sub.add_parser("config", help="Показать текущую конфигурацию.")
    sub.add_parser("intake", help="Принять новые файлы из папки-входящих.")
    sub.add_parser("classify", help="Классифицировать новые снимки (сток/не-сток).")
    return parser


def main(argv: list[str] | None = None) -> int:
    _force_utf8_streams()
    args = build_parser().parse_args(argv)

    cfg = load_config()
    setup_logging(cfg.logs_dir, cfg.log_level)

    command = args.command or "init"
    if command == "config":
        return cmd_config(cfg)
    if command == "intake":
        return cmd_intake(cfg)
    if command == "classify":
        return cmd_classify(cfg)
    return cmd_init(cfg)
