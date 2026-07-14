"""Командная строка Stocker.

Единая точка запуска модулей конвейера. На Шаге 1 доступны:
  * ``init``   — создать каталоги и пустую БД (действие по умолчанию);
  * ``config`` — показать текущую конфигурацию (без утечки ключа).
"""

from __future__ import annotations

import argparse
import logging
import sys

import anthropic

from .classifier import run_classification
from .config import Config, load_config
from .db import get_connection, init_db
from .improver import improve_prompt
from .intake import run_intake
from .logging_setup import setup_logging
from .metadata import run_metadata
from .upload import run_upload

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


def cmd_improve_prompt(cfg: Config) -> int:
    cfg.ensure_dirs()
    init_db(cfg.db_path)
    if not cfg.has_api_key:
        log.error("ANTHROPIC_API_KEY не задан — доработка промпта невозможна.")
        return 1
    conn = get_connection(cfg.db_path)
    try:
        version = improve_prompt(conn, anthropic.Anthropic(api_key=cfg.anthropic_api_key))
    finally:
        conn.close()
    if version is None:
        log.info("Нет необработанных правок — промпт не менялся.")
    else:
        log.info("Промпт доработан → активна версия %d.", version)
    return 0


def cmd_metadata(cfg: Config) -> int:
    cfg.ensure_dirs()
    init_db(cfg.db_path)
    try:
        run_metadata(cfg)
    except RuntimeError as exc:
        log.error("%s", exc)
        return 1
    return 0


def cmd_upload(cfg: Config, dry_run: bool) -> int:
    cfg.ensure_dirs()
    init_db(cfg.db_path)
    stats = run_upload(cfg, dry_run=dry_run)
    if stats["total"] == 0:
        log.info("Нет описанных снимков — выгружать нечего.")
        return 0
    if stats["error"]:
        return 1  # причина уже залогирована в run_upload
    if stats["sent"]:
        log.info(
            "Изображения залиты по FTPS. Теперь применить метаданные: открой "
            "submit.shutterstock.com → кнопка «CSV» вверху → выбери %s",
            stats["csv_path"],
        )
    elif not dry_run and not cfg.has_ftps_creds:
        log.error(
            "Заливка не выполнена: задай STOCKER_SHUTTERSTOCK_USER и "
            "STOCKER_SHUTTERSTOCK_PASSWORD в .env. CSV готов: %s",
            stats["csv_path"],
        )
        return 1
    return 0


def cmd_web(cfg: Config) -> int:
    from .webapp import run_server

    run_server(cfg)
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
    sub.add_parser("improve-prompt", help="Доработать промпт по накопленным правкам.")
    sub.add_parser("metadata", help="Сгенерировать метаданные для одобренных снимков.")
    upload = sub.add_parser(
        "upload", help="Выгрузить описанные снимки на Shutterstock (FTPS + CSV)."
    )
    upload.add_argument(
        "--dry-run",
        action="store_true",
        help="Только собрать CSV, без соединения и заливки (проверка формата).",
    )
    sub.add_parser("web", help="Запустить веб-интерфейс ревью (локальный сервер).")
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
    if command == "improve-prompt":
        return cmd_improve_prompt(cfg)
    if command == "metadata":
        return cmd_metadata(cfg)
    if command == "upload":
        return cmd_upload(cfg, dry_run=args.dry_run)
    if command == "web":
        return cmd_web(cfg)
    return cmd_init(cfg)
