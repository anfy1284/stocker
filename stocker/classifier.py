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

from . import costs
from .config import Config
from .db import (
    STATUS_NEW,
    STATUS_NON_STOCK,
    STATUS_STOCK_CANDIDATE,
    get_connection,
)
from .improver import improve_prompt
from .prompts import get_active_prompt

log = logging.getLogger(__name__)

# Haiku 4.5 — дёшево и достаточно для классификации/комплаенса (см. ТЗ §4).
MODEL = "claude-haiku-4-5"
MAX_TOKENS = 1024

# Системный промпт версионируется и хранится в БД (см. prompts.py); он больше
# не константа. Здесь — только неизменная реплика пользователя к каждому кадру.
USER_PROMPT = "Оцени этот снимок как кандидата на коммерческий микросток."

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


def _classify_image(
    client: anthropic.Anthropic, preview_path: Path, system_prompt: str
) -> tuple[dict[str, object], object]:
    """Один vision-запрос к Haiku; возвращает (вердикт, usage для учёта расходов)."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
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
    return json.loads(text), response.usage


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
        # Перед разбором новой пачки — доработать промпт по накопленным правкам.
        try:
            improve_prompt(conn, client)
        except Exception:
            log.exception(
                "Не удалось доработать промпт по правкам; продолжаю на текущей версии"
            )
        system_prompt = get_active_prompt(conn)  # активная версия из БД
        rows = conn.execute(
            "SELECT id, preview_path FROM assets WHERE status = ?", (STATUS_NEW,)
        ).fetchall()
        stats["total"] = len(rows)

        for row in rows:
            try:
                verdict, usage = _classify_image(
                    client, Path(row["preview_path"]), system_prompt
                )
                _apply_verdict(conn, row["id"], verdict)
                costs.record(conn, MODEL, "classify", row["id"], usage)
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
