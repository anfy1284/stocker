"""Доработка промпта классификатора по правкам пользователя (умная модель).

Пользователь в интерфейсе переносит снимки «в сток / из стока» и поясняет
решение. Перед каждым новым прогоном классификации накопленные правки
скармливаются умной модели (Opus), которая аккуратно дорабатывает системный
промпт: понимает текущий смысл целиком и меняет ТОЛЬКО те моменты, по которым
правки противоречат текущему поведению, остальное оставляя нетронутым. Новая
версия становится активной; правки помечаются обработанными.
"""

from __future__ import annotations

import json
import logging

import anthropic

from . import prompts

log = logging.getLogger(__name__)

# Умная модель для доработки промпта (не Haiku — нужна аккуратность рассуждения).
IMPROVER_MODEL = "claude-opus-4-8"
IMPROVER_MAX_TOKENS = 8000

DECISION_TO_STOCK = "to_stock"
DECISION_FROM_STOCK = "from_stock"

_META_SYSTEM = (
    "Ты — редактор системного промпта для ИИ-классификатора фотографий "
    "(отбор «сток / не-сток» для коммерческого микростока). Тебе дают ТЕКУЩИЙ "
    "промпт и список правок пользователя: его решения по конкретным снимкам "
    "(перенёс в сток или из стока) с пояснениями. Задача — аккуратно доработать "
    "промпт, чтобы он учитывал пояснения.\n\n"
    "ПРАВИЛА (строго):\n"
    "1. Сначала внимательно прочитай и пойми смысл ТЕКУЩЕГО промпта целиком.\n"
    "2. Меняй ТОЛЬКО те смысловые моменты, по которым пояснения пользователя "
    "противоречат текущему поведению. Все остальные правила, смыслы и "
    "формулировки текущего промпта СОХРАНИ дословно.\n"
    "3. Не переобучайся на единичном кадре: обобщай пояснение до правила лишь "
    "если пользователь явно указывает общий принцип; частный случай формулируй "
    "осторожно и узко.\n"
    "4. НЕ трогай технические инструкции о формате ответа (структура вердикта, "
    "поля stock_worthy/reason/category/has_logo/has_brand/has_text, язык "
    "reason/notes) — они должны остаться как есть.\n"
    "5. Не удаляй существующие правила без явного указания пользователя.\n"
    "6. Не раздувай промпт: вноси минимальные точечные правки.\n\n"
    "Верни новый полный текст промпта (new_prompt) и краткое описание на "
    "русском, что именно изменил и почему (change_note)."
)

_IMPROVER_SCHEMA = {
    "type": "object",
    "properties": {
        "new_prompt": {"type": "string"},
        "change_note": {"type": "string"},
    },
    "required": ["new_prompt", "change_note"],
    "additionalProperties": False,
}


def add_feedback(
    conn,
    asset_id: int | None,
    decision: str,
    comment: str,
    prompt_version: int | None,
) -> None:
    """Записывает правку пользователя (вызывается из интерфейса)."""
    from datetime import datetime

    conn.execute(
        "INSERT INTO feedback (asset_id, decision, comment, prompt_version, "
        "processed, created_at) VALUES (?, ?, ?, ?, 0, ?)",
        (asset_id, decision, comment, prompt_version, datetime.now().isoformat()),
    )
    conn.commit()


def _pending_feedback(conn) -> list:
    return conn.execute(
        """
        SELECT f.id, f.decision, f.comment, a.category, a.classification_reason
        FROM feedback f LEFT JOIN assets a ON a.id = f.asset_id
        WHERE f.processed = 0 ORDER BY f.id
        """
    ).fetchall()


def _format_feedback(items: list) -> str:
    lines = []
    for it in items:
        moved = "В СТОК" if it["decision"] == DECISION_TO_STOCK else "ИЗ СТОКА"
        was = f'категория «{it["category"]}», причина: «{it["classification_reason"]}»' \
            if it["category"] is not None else "(вердикт неизвестен)"
        comment = (it["comment"] or "").strip() or "(без пояснения)"
        lines.append(
            f"- Классификатор поставил: {was}. Пользователь перенёс: {moved}. "
            f"Пояснение: «{comment}»."
        )
    return "\n".join(lines)


def improve_prompt(conn, client: anthropic.Anthropic) -> int | None:
    """Дорабатывает промпт по необработанным правкам. Возвращает № новой версии.

    Если правок нет — ничего не делает и возвращает None.
    """
    items = _pending_feedback(conn)
    if not items:
        return None

    current = prompts.get_active_prompt(conn)
    user_text = (
        "ТЕКУЩИЙ ПРОМПТ:\n<<<\n" + current + "\n>>>\n\n"
        f"ПРАВКИ ПОЛЬЗОВАТЕЛЯ ({len(items)}):\n" + _format_feedback(items)
    )

    response = client.messages.create(
        model=IMPROVER_MODEL,
        max_tokens=IMPROVER_MAX_TOKENS,
        system=_META_SYSTEM,
        messages=[{"role": "user", "content": user_text}],
        thinking={"type": "adaptive"},
        output_config={
            "effort": "medium",
            "format": {"type": "json_schema", "schema": _IMPROVER_SCHEMA},
        },
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("умная модель отклонила доработку промпта (refusal)")

    text = next(b.text for b in response.content if b.type == "text")
    result = json.loads(text)

    note = f"ИИ-доработка по {len(items)} правкам: {result['change_note']}"
    version = prompts.add_version(
        conn, result["new_prompt"], note=note, source=prompts.SOURCE_AI, activate=True
    )

    ids = [it["id"] for it in items]
    conn.executemany("UPDATE feedback SET processed = 1 WHERE id = ?", [(i,) for i in ids])
    conn.commit()

    log.info("Промпт доработан по %d правкам → версия %d", len(items), version)
    return version
