"""Учёт расходов на ИИ-вызовы (основа финансовой аналитики).

Каждый вызов модели пишет строку в ``api_costs`` с токенами и стоимостью в USD.
Позже это соединяется с доходом со стоков → P&L-дашборд (в плюсе мы или ИИ
съедает больше, чем приносит).
"""

from __future__ import annotations

import logging
from datetime import datetime

log = logging.getLogger(__name__)

# Цены за 1 млн токенов (USD): (вход, выход). Кэш: чтение ~0.1×, запись ~1.25×.
PRICING = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-opus-4-8": (5.0, 25.0),
}

_MILLION = 1_000_000


def _u(usage, name: str) -> int:
    """Безопасно берёт поле usage (может отсутствовать/быть None)."""
    return int(getattr(usage, name, 0) or 0)


def compute_cost(model: str, usage, discount: float = 1.0) -> float:
    """Стоимость вызова в USD по usage ответа Anthropic.

    ``discount`` умножает итог: 0.5 для Batch API (пакетный разбор вдвое дешевле).
    """
    rates = PRICING.get(model)
    if rates is None:
        log.warning("Нет цены для модели %s — считаю стоимость 0", model)
        return 0.0
    in_rate, out_rate = rates
    inp = _u(usage, "input_tokens")
    out = _u(usage, "output_tokens")
    cache_read = _u(usage, "cache_read_input_tokens")
    cache_write = _u(usage, "cache_creation_input_tokens")
    return discount * (
        inp * in_rate
        + out * out_rate
        + cache_read * in_rate * 0.1
        + cache_write * in_rate * 1.25
    ) / _MILLION


def record(
    conn, model: str, operation: str, asset_id: int | None, usage, discount: float = 1.0
) -> float:
    """Пишет строку расхода в ``api_costs``; возвращает стоимость вызова."""
    cost = compute_cost(model, usage, discount)
    conn.execute(
        "INSERT INTO api_costs (created_at, model, operation, asset_id, "
        "input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens, "
        "cost_usd) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            datetime.now().isoformat(),
            model,
            operation,
            asset_id,
            _u(usage, "input_tokens"),
            _u(usage, "output_tokens"),
            _u(usage, "cache_read_input_tokens"),
            _u(usage, "cache_creation_input_tokens"),
            cost,
        ),
    )
    return cost
