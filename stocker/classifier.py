"""Классификация «сток / не-сток» (M2, смысловой фильтр).

Для каждого снимка со статусом «новый» — один запрос к Haiku-зрению,
возвращающий структурированный вердикт (стоковость, причина, флаги
логотип/бренд/текст, категория). Результат пишется в БД, статус меняется на
«сток-кандидат» или «не-сток» (корзина). Обработка по одному снимку (без батча
— это Шаг 12). Промпт калибруется на валидационной выборке (Шаг 4).
"""

from __future__ import annotations

import base64
import json
import logging
from datetime import datetime
from pathlib import Path

import anthropic

from .config import Config
from .db import (
    STATUS_NEW,
    STATUS_NON_STOCK,
    STATUS_STOCK_CANDIDATE,
    get_connection,
)

log = logging.getLogger(__name__)

# Haiku 4.5 — дёшево и достаточно для классификации/комплаенса (см. ТЗ §4).
MODEL = "claude-haiku-4-5"
MAX_TOKENS = 1024

# Стартовый промпт. Критерии «стоковости» уточняются на выборке (Шаг 4).
SYSTEM_PROMPT = (
    "Ты — ассистент фотостокового отбора. По фотографии реши, годится ли она "
    "как коммерческий сток, или это бытовой семейный снимок либо мусор "
    "(документ, скриншот, фото счётчика, фото для жалобы, случайный кадр).\n"
    "Критерии стоковости: чёткая тема, техническое качество, потенциальная "
    "коммерческая или редакционная ценность. Отвечай строго по схеме. "
    "Поля reason и notes — на русском, кратко. category — короткая стоковая "
    "категория (например «природа», «еда», «люди», «бизнес»); для нестокового "
    'используй "нет". Флаги has_logo/has_brand/has_text — виден ли на кадре '
    "логотип, узнаваемый бренд, читаемый текст (важно для комплаенса)."
)

USER_PROMPT = "Оцени этот снимок как кандидата на микросток."

# JSON-схема вердикта — гарантирует разбираемый структурированный ответ.
_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "stock_worthy": {"type": "boolean"},
        "reason": {"type": "string"},
        "category": {"type": "string"},
        "has_logo": {"type": "boolean"},
        "has_brand": {"type": "boolean"},
        "has_text": {"type": "boolean"},
        "notes": {"type": "string"},
    },
    "required": [
        "stock_worthy",
        "reason",
        "category",
        "has_logo",
        "has_brand",
        "has_text",
        "notes",
    ],
    "additionalProperties": False,
}


def _encode_image(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode("ascii")


def _classify_image(client: anthropic.Anthropic, preview_path: Path) -> dict[str, object]:
    """Один vision-запрос к Haiku; возвращает разобранный вердикт."""
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
        output_config={"format": {"type": "json_schema", "schema": _VERDICT_SCHEMA}},
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("модель отклонила запрос (refusal)")
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def _apply_verdict(conn, asset_id: int, verdict: dict[str, object]) -> None:
    """Пишет вердикт в БД и переводит снимок в сток-кандидаты или корзину."""
    stock_worthy = bool(verdict["stock_worthy"])
    status = STATUS_STOCK_CANDIDATE if stock_worthy else STATUS_NON_STOCK
    conn.execute(
        """
        UPDATE assets SET
            stock_worthy = ?,
            classification_reason = ?,
            category = ?,
            has_logo = ?,
            has_brand = ?,
            has_text = ?,
            classification_notes = ?,
            classified_at = ?,
            status = ?
        WHERE id = ?
        """,
        (
            int(stock_worthy),
            verdict["reason"],
            verdict["category"],
            int(bool(verdict["has_logo"])),
            int(bool(verdict["has_brand"])),
            int(bool(verdict["has_text"])),
            verdict["notes"],
            datetime.now().isoformat(),
            status,
            asset_id,
        ),
    )


def run_classification(cfg: Config) -> dict[str, int]:
    """Классифицирует все снимки со статусом «новый». Возвращает статистику."""
    if not cfg.has_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY не задан — классификация невозможна (см. .env)."
        )

    client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
    stats = {"total": 0, "stock": 0, "non_stock": 0, "errors": 0}

    conn = get_connection(cfg.db_path)
    try:
        rows = conn.execute(
            "SELECT id, preview_path FROM assets WHERE status = ?", (STATUS_NEW,)
        ).fetchall()
        stats["total"] = len(rows)

        for row in rows:
            try:
                verdict = _classify_image(client, Path(row["preview_path"]))
                _apply_verdict(conn, row["id"], verdict)
                conn.commit()
                if verdict["stock_worthy"]:
                    stats["stock"] += 1
                else:
                    stats["non_stock"] += 1
                log.info(
                    "Снимок %d: %s — %s",
                    row["id"],
                    "сток" if verdict["stock_worthy"] else "не-сток",
                    verdict["reason"],
                )
            except Exception:
                stats["errors"] += 1
                log.exception("Ошибка классификации снимка %d", row["id"])
    finally:
        conn.close()

    log.info(
        "Классификация завершена: всего %(total)d, сток %(stock)d, "
        "не-сток %(non_stock)d, ошибок %(errors)d",
        stats,
    )
    return stats
