"""Доступ к SQLite-БД, схема и её пошаговые миграции.

Версия схемы хранится в ``PRAGMA user_version``. ``_MIGRATIONS`` — список
функций «поднять схему на одну версию»; ``init_db`` применяет недостающие по
порядку. Каждый шаг проекта, добавляющий таблицы/поля, дописывает сюда свою
миграцию — так схема растёт предсказуемо и обновляется на месте.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from pathlib import Path

log = logging.getLogger(__name__)

# --- Значения статуса в конвейере (колонка assets.status) ------------------
STATUS_NEW = "new"  # принят, ещё не классифицирован
STATUS_STOCK_CANDIDATE = "stock_candidate"  # прошёл классификатор как сток
STATUS_NON_STOCK = "non_stock"  # отсеян классификатором (корзина, не удаление)


# --- Миграции --------------------------------------------------------------
def _migrate_v1_assets(conn: sqlite3.Connection) -> None:
    """v1 (Шаг 2, приём): таблица снимков ``assets``.

    Заведены только поля этапа приёма и статус. Поля последующих модулей
    (классификация, группировка, метаданные, загрузка) добавляют свои шаги
    отдельными миграциями.
    """
    conn.execute(
        """
        CREATE TABLE assets (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            original_path TEXT    NOT NULL,
            preview_path  TEXT,
            content_hash  TEXT    NOT NULL UNIQUE,
            file_type     TEXT    NOT NULL,
            file_size     INTEGER,
            width         INTEGER,
            height        INTEGER,
            captured_at   TEXT,
            camera_make   TEXT,
            camera_model  TEXT,
            orientation   INTEGER,
            status        TEXT    NOT NULL DEFAULT 'new',
            created_at    TEXT    NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX idx_assets_status ON assets(status)")


def _migrate_v2_classification(conn: sqlite3.Connection) -> None:
    """v2 (Шаг 3, классификация): поля вердикта «сток/не-сток» в ``assets``.

    Заполняются классификатором; до его прогона — NULL. Флаги комплаенса
    (логотип/бренд/текст) хранятся как 0/1.
    """
    for column, coltype in (
        ("stock_worthy", "INTEGER"),
        ("classification_reason", "TEXT"),
        ("category", "TEXT"),
        ("has_logo", "INTEGER"),
        ("has_brand", "INTEGER"),
        ("has_text", "INTEGER"),
        ("classification_notes", "TEXT"),
        ("classified_at", "TEXT"),
    ):
        conn.execute(f"ALTER TABLE assets ADD COLUMN {column} {coltype}")


# Порядковый список миграций; индекс+1 = целевая версия схемы.
_MIGRATIONS: list[Callable[[sqlite3.Connection], None]] = [
    _migrate_v1_assets,
    _migrate_v2_classification,
]

# Текущая версия схемы = число применённых миграций.
SCHEMA_VERSION = len(_MIGRATIONS)


def get_connection(db_path: Path) -> sqlite3.Connection:
    """Открывает соединение с включёнными внешними ключами и доступом по имени."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Path) -> bool:
    """Создаёт файл БД при первом запуске и применяет недостающие миграции.

    Возвращает ``True``, если файл был создан этим вызовом, иначе ``False``.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    created = not db_path.exists()

    conn = get_connection(db_path)
    try:
        current = conn.execute("PRAGMA user_version").fetchone()[0]
        for version in range(current, SCHEMA_VERSION):
            _MIGRATIONS[version](conn)
            conn.execute(f"PRAGMA user_version = {version + 1}")
            conn.commit()
            log.info("Применена миграция схемы v%d", version + 1)
    finally:
        conn.close()

    if created:
        log.info("Создан файл БД: %s (версия схемы %d)", db_path, SCHEMA_VERSION)
    else:
        log.info("БД готова: %s (версия схемы %d)", db_path, SCHEMA_VERSION)
    return created
