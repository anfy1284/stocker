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
import time
from collections.abc import Callable
from datetime import datetime
from io import BytesIO
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from PIL import Image

from . import costs, organize
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

# Разбор идёт через Batch API — вдвое дешевле обычных вызовов (пакетный разбор
# архива, Шаг 12). Плюс для классификации шлём УМЕНЬШЕННУЮ картинку (стоковость
# видна и на 512px), что дополнительно срезает входные токены.
BATCH_DISCOUNT = 0.5
CLASSIFY_MAX_SIDE = 512
CLASSIFY_QUALITY = 70
_POLL_INTERVAL = 8  # секунд между опросами статуса батча

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


def _small_image_b64(preview_path: Path) -> str:
    """Base64 уменьшенной (≤512px) версии превью — меньше входных токенов."""
    with Image.open(preview_path) as im:
        im = im.convert("RGB")
        im.thumbnail((CLASSIFY_MAX_SIDE, CLASSIFY_MAX_SIDE))
        buf = BytesIO()
        im.save(buf, "JPEG", quality=CLASSIFY_QUALITY)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def _build_request(asset_id: int, preview_path: Path, system_prompt: str) -> Request:
    """Одна строка батча: vision-запрос к Haiku по уменьшенной картинке."""
    return Request(
        custom_id=str(asset_id),
        params=MessageCreateParamsNonStreaming(
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
                                "data": _small_image_b64(preview_path),
                            },
                        },
                        {"type": "text", "text": USER_PROMPT},
                    ],
                }
            ],
            output_config={
                "format": {"type": "json_schema", "schema": _VERDICT_SCHEMA}
            },
        ),
    )


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


def _apply_result(conn, cfg: Config, asset_id: int, message, stats: dict) -> None:
    """Разбирает ответ модели по одному снимку: вердикт → БД → папка → расход."""
    if getattr(message, "stop_reason", None) == "refusal":
        raise RuntimeError("модель отклонила запрос (refusal)")
    text = next(b.text for b in message.content if b.type == "text")
    verdict = json.loads(text)
    _apply_verdict(conn, asset_id, verdict)
    # Пакетный разбор вдвое дешевле — учитываем это в стоимости.
    costs.record(conn, MODEL, "classify", asset_id, message.usage, discount=BATCH_DISCOUNT)
    new_status = (
        STATUS_STOCK_CANDIDATE if verdict["stock_worthy"] else STATUS_NON_STOCK
    )
    organize.relocate(conn, cfg, asset_id, new_status)  # сток-кандидаты → stock/
    conn.commit()
    if verdict["stock_worthy"]:
        stats["stock"] += 1
    else:
        stats["non_stock"] += 1


def run_classification(
    cfg: Config, on_progress: Callable[[dict], None] | None = None
) -> dict[str, int]:
    """Классифицирует все «новые» снимки пакетом через Batch API.

    ``on_progress`` (если задан) вызывается с копией статистики (плюс ``done`` —
    сколько снимков уже обработано) при отправке, во время опроса батча и после
    применения результатов — веб-интерфейс двигает прогресс-бар.
    """
    if not cfg.has_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY не задан — классификация невозможна (см. .env)."
        )

    client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
    stats = {"total": 0, "stock": 0, "non_stock": 0, "errors": 0, "done": 0}

    def report() -> None:
        if on_progress:
            on_progress(dict(stats))

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
        report()
        if not rows:
            log.info("Нет новых снимков для классификации.")
            return stats

        requests = [
            _build_request(row["id"], Path(row["preview_path"]), system_prompt)
            for row in rows
        ]
        batch = client.messages.batches.create(requests=requests)
        log.info("Батч классификации создан: %s (%d снимков)", batch.id, len(requests))

        # Опрос до завершения. Прогресс — по счётчикам обработанных батчем запросов.
        while True:
            info = client.messages.batches.retrieve(batch.id)
            if info.processing_status == "ended":
                break
            counts = info.request_counts
            stats["done"] = (
                counts.succeeded + counts.errored + counts.canceled + counts.expired
            )
            report()
            time.sleep(_POLL_INTERVAL)

        # Результаты приходят в произвольном порядке — раскладываем по custom_id.
        stats["done"] = 0
        for result in client.messages.batches.results(batch.id):
            asset_id = int(result.custom_id)
            try:
                if result.result.type != "succeeded":
                    raise RuntimeError(f"результат батча: {result.result.type}")
                _apply_result(conn, cfg, asset_id, result.result.message, stats)
            except Exception:
                stats["errors"] += 1
                log.exception("Ошибка классификации снимка %d", asset_id)
            stats["done"] += 1
            report()
    finally:
        conn.close()

    log.info(
        "Классификация завершена: всего %(total)d, сток %(stock)d, "
        "не-сток %(non_stock)d, ошибок %(errors)d",
        stats,
    )
    return stats
