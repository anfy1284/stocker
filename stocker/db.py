"""Доступ к SQLite-БД и её инициализация.

На Шаге 1 БД создаётся пустой (без таблиц) — схема ``assets`` появится на
Шаге 2. Версия схемы хранится в ``PRAGMA user_version``: это задел под
пошаговые миграции, чтобы последующие модули добавляли таблицы предсказуемо.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

log = logging.getLogger(__name__)

# Текущая версия схемы. На Шаге 1 таблиц ещё нет → 0.
# Каждый шаг, добавляющий таблицы/поля, поднимает это число и свою миграцию.
SCHEMA_VERSION = 0


def get_connection(db_path: Path) -> sqlite3.Connection:
    """Открывает соединение с включёнными внешними ключами и доступом по имени."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Path) -> bool:
    """Создаёт файл БД при первом запуске и фиксирует версию схемы.

    Возвращает ``True``, если файл был создан этим вызовом, иначе ``False``.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    created = not db_path.exists()

    conn = get_connection(db_path)
    try:
        current = conn.execute("PRAGMA user_version").fetchone()[0]
        if current < SCHEMA_VERSION:
            # Место для будущих миграций (Шаг 2+).
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            conn.commit()
    finally:
        conn.close()

    if created:
        log.info("Создан файл БД: %s (версия схемы %d)", db_path, SCHEMA_VERSION)
    else:
        log.info("БД уже существует: %s (версия схемы %d)", db_path, SCHEMA_VERSION)
    return created
