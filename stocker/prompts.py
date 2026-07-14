"""Версионируемый системный промпт классификатора (хранится в БД).

Промпт не захардкожен в рантайме: классификатор читает активную версию из
таблицы ``classifier_prompts``. При инициализации БД засевается стартовая
версия (``SEED_CLASSIFIER_PROMPT``), чтобы установка «с нуля» сразу работала.
История версий позволяет откат и редактирование (правка = новая версия);
умная модель может предлагать новые версии по накопленному фидбэку.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime

# Источники версии промпта.
SOURCE_SEED = "seed"      # засеяна при инициализации БД
SOURCE_MANUAL = "manual"  # отредактирована человеком
SOURCE_AI = "ai"          # предложена умной моделью по фидбэку

# Стартовый промпт: КОММЕРЧЕСКИЙ сток, нулевой юридический риск, без editorial.
# Используется ТОЛЬКО для засева БД; рантайм читает активную версию из таблицы.
SEED_CLASSIFIER_PROMPT = (
    "Ты — ассистент отбора фотографий для КОММЕРЧЕСКОГО микростока "
    "(royalty-free). Владелец хочет НУЛЕВОЙ юридический риск и НЕ занимается "
    "editorial-лицензированием. Отвечай строго по схеме.\n\n"
    "Ставь stock_worthy=false (в корзину) при ЛЮБОМ, даже потенциальном, риске "
    "по правам:\n"
    "- виден бренд, логотип или товарный знак (марки авто вроде Mercedes, "
    "логотипы продуктов, спортивные бренды);\n"
    "- читаемые вывески, названия магазинов/ресторанов, рекламные надписи, "
    "занимающие заметное место в кадре;\n"
    "- интерьеры музеев, галерей, частных помещений; выставленные произведения "
    "искусства, скульптуры (право собственности и авторское право);\n"
    "- узнаваемые охраняемые достопримечательности и здания, где коммерческое "
    "использование может быть ограничено (особенно во Франции — там нет "
    "свободы панорамы);\n"
    "- сомневаешься в чистоте прав — отклоняй (лучше выбросить, чем рисковать).\n\n"
    "Наличие людей и лиц — само по себе НЕ причина для отказа: на членов семьи "
    "есть модельные релизы, а релиз позже подтверждает человек. Людей оценивай "
    "по коммерческой ценности и техническому качеству, а не по «нужен релиз».\n\n"
    "Также отклоняй нестоковое: бытовые снимки без коммерческой темы, плохое "
    "техническое качество (смаз, шум, дефекты плёнки), документы, скриншоты, "
    "фото счётчиков, мусор.\n\n"
    "В сток (stock_worthy=true) — только коммерчески чистые кадры: понятная "
    "тема, техническое качество, без брендов/вывесок/охраняемых объектов.\n\n"
    "Флаги has_logo/has_brand/has_text ставь честно (сигналы комплаенса). "
    "reason и notes — на русском, кратко. category — короткая категория; для "
    'нестокового используй "нет".'
)


def seed_if_empty(conn: sqlite3.Connection) -> None:
    """Засевает стартовую версию, если таблица промптов пуста (идемпотентно)."""
    count = conn.execute("SELECT count(1) FROM classifier_prompts").fetchone()[0]
    if count == 0:
        add_version(
            conn,
            SEED_CLASSIFIER_PROMPT,
            note="Стартовый промпт (засев при инициализации)",
            source=SOURCE_SEED,
            activate=True,
        )


def get_active_prompt(conn: sqlite3.Connection) -> str:
    """Текст активной версии промпта."""
    row = conn.execute(
        "SELECT text FROM classifier_prompts WHERE is_active = 1 "
        "ORDER BY version DESC LIMIT 1"
    ).fetchone()
    if row is None:
        raise RuntimeError("В БД нет активного промпта классификатора.")
    return row[0]


def get_active_version(conn: sqlite3.Connection) -> int | None:
    """Номер активной версии промпта (для привязки правок к версии)."""
    row = conn.execute(
        "SELECT version FROM classifier_prompts WHERE is_active = 1 "
        "ORDER BY version DESC LIMIT 1"
    ).fetchone()
    return row[0] if row else None


def _next_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT max(version) FROM classifier_prompts").fetchone()
    return (row[0] or 0) + 1


def add_version(
    conn: sqlite3.Connection,
    text: str,
    note: str,
    source: str,
    activate: bool = False,
) -> int:
    """Добавляет новую версию промпта; при ``activate`` делает её активной."""
    version = _next_version(conn)
    if activate:
        conn.execute("UPDATE classifier_prompts SET is_active = 0")
    conn.execute(
        "INSERT INTO classifier_prompts (version, text, note, source, is_active, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (version, text, note, source, int(activate), datetime.now().isoformat()),
    )
    conn.commit()
    return version


def activate_version(conn: sqlite3.Connection, version: int) -> None:
    """Делает указанную версию активной (откат/переключение)."""
    conn.execute("UPDATE classifier_prompts SET is_active = 0")
    conn.execute("UPDATE classifier_prompts SET is_active = 1 WHERE version = ?", (version,))
    conn.commit()


def list_versions(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Все версии промпта, новые сверху (для истории в интерфейсе)."""
    return conn.execute(
        "SELECT version, note, source, is_active, created_at, text "
        "FROM classifier_prompts ORDER BY version DESC"
    ).fetchall()
