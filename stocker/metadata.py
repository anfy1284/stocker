"""Генерация метаданных для стока (трек загрузки, Шаг A).

Для каждого одобренного (``approved``) снимка — один vision-запрос к Haiku,
возвращающий структурированные метаданные в формате Shutterstock: английское
описание, ключевые слова и 1–2 категории строго из фиксированного списка.
Результат пишется в БД, статус → ``described`` (ждёт ревью метаданных, Шаг B).

Метаданные генерим ТОЛЬКО для одобренных на триаже кадров (не заранее для всех
— это расточительно): триаж → метаданные → ревью метаданных → очередь загрузки.
Промпт пока константа модуля (не версионируется, в отличие от классификатора).
"""

from __future__ import annotations

import base64
import json
import logging
from datetime import datetime
from pathlib import Path

import anthropic

from . import costs
from .config import Config
from .db import STATUS_APPROVED, STATUS_DESCRIBED, get_connection

log = logging.getLogger(__name__)

# Haiku 4.5 — дёшево и достаточно для описаний/ключевиков (тэггинг, см. ТЗ §4).
MODEL = "claude-haiku-4-5"
MAX_TOKENS = 1024

# Ограничения Shutterstock: описание до 200 символов, до 50 ключевых слов.
MAX_DESCRIPTION = 200
MAX_KEYWORDS = 50

# 26 фиксированных категорий Shutterstock для фото (значение колонки D в CSV).
# Список закрытый: любое отклонение → CSV отклоняется, поэтому передаём его в
# JSON-схему как enum, чтобы модель не могла выдумать категорию.
CATEGORIES = [
    "Abstract",
    "Animals/Wildlife",
    "Arts",
    "Backgrounds/Textures",
    "Beauty/Fashion",
    "Buildings/Landmarks",
    "Business/Finance",
    "Celebrities",
    "Education",
    "Food and Drink",
    "Healthcare/Medical",
    "Holidays",
    "Industrial",
    "Interiors",
    "Miscellaneous",
    "Nature",
    "Objects",
    "Parks/Outdoor",
    "People",
    "Religion",
    "Science",
    "Signs/Symbols",
    "Sports/Recreation",
    "Technology",
    "Transportation",
    "Vintage",
]

SYSTEM_PROMPT = (
    "You write metadata for photos submitted to Shutterstock as royalty-free "
    "COMMERCIAL stock. Everything you output is in ENGLISH (buyers search in "
    "English), regardless of the image origin. Answer strictly per the schema.\n\n"
    "description: one natural, buyer-facing sentence describing the subject and "
    "context, plain and specific, no marketing fluff, no camera settings, no "
    f"trailing period required. Hard limit {MAX_DESCRIPTION} characters.\n\n"
    "keywords: 25–50 single- or two-word English terms, ordered MOST RELEVANT "
    "FIRST (the first ~10 matter most for search). Cover the concrete subject, "
    "setting, actions, concepts, colors, mood and season. Use singular nouns, "
    "lowercase. No duplicates, no phrases longer than two words, no brand names, "
    "no punctuation inside a keyword.\n\n"
    "category1 (required) and category2 (optional): choose from the fixed "
    "Shutterstock list only. If a single category fits, leave category2 as an "
    "empty string.\n\n"
    "This is commercial (not editorial) content, so never reference brands, "
    "logos or trademarks in the text even if you think you see them."
)

USER_PROMPT = "Generate Shutterstock metadata for this photo."

# JSON-схема метаданных — гарантирует разбираемый структурированный ответ и
# не даёт модели выйти за закрытый список категорий.
_METADATA_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        # Границы количества задаём промптом и подчищаем в _clean_keywords:
        # structured output не поддерживает minItems/maxItems > 1.
        "keywords": {"type": "array", "items": {"type": "string"}},
        "category1": {"type": "string", "enum": CATEGORIES},
        "category2": {"type": "string", "enum": [*CATEGORIES, ""]},
    },
    "required": ["description", "keywords", "category1", "category2"],
    "additionalProperties": False,
}


def _encode_image(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode("ascii")


def _clean_keywords(raw: object) -> list[str]:
    """Нормализует ключевые слова: строки, обрезка, нижний регистр, дедуп, лимит."""
    seen: dict[str, None] = {}
    for item in raw if isinstance(raw, list) else []:
        kw = str(item).strip().lower()
        if kw and kw not in seen:
            seen[kw] = None
        if len(seen) >= MAX_KEYWORDS:
            break
    return list(seen)


def _generate(
    client: anthropic.Anthropic, preview_path: Path
) -> tuple[dict[str, object], object]:
    """Один vision-запрос к Haiku; возвращает (метаданные, usage для учёта)."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": _encode_image(preview_path),
                        },
                    },
                    {"type": "text", "text": USER_PROMPT},
                ],
            }
        ],
        output_config={"format": {"type": "json_schema", "schema": _METADATA_SCHEMA}},
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("модель отклонила запрос (refusal)")
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text), response.usage


def _apply_metadata(conn, asset_id: int, meta: dict[str, object]) -> None:
    """Пишет метаданные в БД и переводит снимок в статус ``described``."""
    description = str(meta["description"]).strip()[:MAX_DESCRIPTION]
    keywords = _clean_keywords(meta.get("keywords"))
    category2 = str(meta.get("category2") or "").strip()
    conn.execute(
        """
        UPDATE assets SET
            meta_description = ?,
            meta_keywords = ?,
            meta_category1 = ?,
            meta_category2 = ?,
            meta_generated_at = ?,
            status = ?
        WHERE id = ?
        """,
        (
            description,
            json.dumps(keywords, ensure_ascii=False),
            str(meta["category1"]).strip(),
            category2 or None,
            datetime.now().isoformat(),
            STATUS_DESCRIBED,
            asset_id,
        ),
    )


def run_metadata(cfg: Config) -> dict[str, int]:
    """Генерирует метаданные для всех одобренных снимков. Возвращает статистику."""
    if not cfg.has_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY не задан — генерация метаданных невозможна (см. .env)."
        )

    client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
    stats = {"total": 0, "done": 0, "errors": 0}

    conn = get_connection(cfg.db_path)
    try:
        rows = conn.execute(
            "SELECT id, preview_path FROM assets WHERE status = ?", (STATUS_APPROVED,)
        ).fetchall()
        stats["total"] = len(rows)

        for row in rows:
            try:
                meta, usage = _generate(client, Path(row["preview_path"]))
                _apply_metadata(conn, row["id"], meta)
                costs.record(conn, MODEL, "metadata", row["id"], usage)
                conn.commit()
                stats["done"] += 1
                log.info(
                    "Снимок %d: метаданные готовы — %s [%s]",
                    row["id"],
                    str(meta["description"])[:60],
                    meta["category1"],
                )
            except Exception:
                stats["errors"] += 1
                log.exception("Ошибка генерации метаданных снимка %d", row["id"])
    finally:
        conn.close()

    log.info(
        "Генерация метаданных завершена: всего %(total)d, готово %(done)d, "
        "ошибок %(errors)d",
        stats,
    )
    return stats
