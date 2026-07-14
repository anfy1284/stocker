"""Веб-интерфейс ревью (M4, Шаг 5): локальный Flask-сервер.

Показывает кучи снимков (сток/не-сток — необработанные; одобрено/брак —
обработанные), даёт пользователю решать «в сток / из стока» (свайпом или
кнопками). Решение переводит снимок в очередь на отправку либо в брак; при
расхождении с вердиктом ИИ пользователь поясняет решение — пояснение копится
в фидбэк и позже дорабатывает промпт. Отдельно — история версий промпта
(активировать/откатить/редактировать).
"""

from __future__ import annotations

import logging
from pathlib import Path

import anthropic
from flask import Flask, abort, jsonify, request, send_file, send_from_directory

from . import improver, prompts
from .config import Config, load_config
from .db import (
    STATUS_APPROVED,
    STATUS_NON_STOCK,
    STATUS_REJECTED,
    STATUS_STOCK_CANDIDATE,
    get_connection,
    init_db,
)

log = logging.getLogger(__name__)
WEB_DIR = Path(__file__).parent / "web"

# Кучи, доступные в интерфейсе (в порядке показа).
PILES = (STATUS_STOCK_CANDIDATE, STATUS_NON_STOCK, STATUS_APPROVED, STATUS_REJECTED)

# Размер миниатюры для сетки (полное превью 1600px — только в крупном просмотре).
THUMB_MAX_SIDE = 320


def create_app(cfg: Config | None = None) -> Flask:
    cfg = cfg or load_config()
    app = Flask(__name__)

    def _conn():
        return get_connection(cfg.db_path)

    @app.get("/")
    def index():
        return send_file(WEB_DIR / "index.html")

    @app.get("/previews/<path:name>")
    def preview(name: str):
        return send_from_directory(cfg.previews_dir, name)

    @app.get("/thumb/<path:name>")
    def thumb(name: str):
        """Маленькая миниатюра для сетки (кэшируется на диск) — лёгкая загрузка."""
        thumbs = cfg.previews_dir.parent / "thumbs"
        thumbs.mkdir(parents=True, exist_ok=True)
        dest = thumbs / name
        if not dest.exists():
            src = cfg.previews_dir / name
            if not src.exists():
                abort(404)
            from PIL import Image

            with Image.open(src) as im:
                im = im.convert("RGB")
                im.thumbnail((THUMB_MAX_SIDE, THUMB_MAX_SIDE))
                im.save(dest, "JPEG", quality=80)
        return send_from_directory(thumbs, name)

    @app.get("/api/summary")
    def summary():
        conn = _conn()
        try:
            counts = {
                s: conn.execute(
                    "SELECT count(1) FROM assets WHERE status = ?", (s,)
                ).fetchone()[0]
                for s in PILES
            }
        finally:
            conn.close()
        return jsonify(counts)

    @app.get("/api/assets")
    def assets():
        status = request.args.get("status", STATUS_STOCK_CANDIDATE)
        conn = _conn()
        try:
            rows = conn.execute(
                "SELECT id, preview_path, status, category, classification_reason, "
                "has_logo, has_brand, has_text FROM assets WHERE status = ? ORDER BY id",
                (status,),
            ).fetchall()
            out = []
            for r in rows:
                fb = conn.execute(
                    "SELECT comment, created_at FROM feedback "
                    "WHERE asset_id = ? AND comment IS NOT NULL ORDER BY id",
                    (r["id"],),
                ).fetchall()
                out.append(
                    {
                        "id": r["id"],
                        "preview": Path(r["preview_path"]).name,
                        "status": r["status"],
                        "category": r["category"],
                        "reason": r["classification_reason"],
                        "flags": [
                            n
                            for n, v in (
                                ("логотип", r["has_logo"]),
                                ("бренд", r["has_brand"]),
                                ("текст", r["has_text"]),
                            )
                            if v
                        ],
                        "comments": [
                            {"text": f["comment"], "at": f["created_at"]} for f in fb
                        ],
                    }
                )
        finally:
            conn.close()
        return jsonify(out)

    @app.post("/api/assets/<int:asset_id>/decision")
    def decision(asset_id: int):
        data = request.get_json(force=True)
        action = data.get("action")  # approve | reject
        comment = (data.get("comment") or "").strip()
        if action not in ("approve", "reject"):
            abort(400, "action должен быть approve или reject")

        conn = _conn()
        try:
            row = conn.execute(
                "SELECT status FROM assets WHERE id = ?", (asset_id,)
            ).fetchone()
            if row is None:
                abort(404)
            old = row["status"]
            contradiction = (action == "approve" and old == STATUS_NON_STOCK) or (
                action == "reject" and old == STATUS_STOCK_CANDIDATE
            )
            new_status = STATUS_APPROVED if action == "approve" else STATUS_REJECTED
            conn.execute(
                "UPDATE assets SET status = ? WHERE id = ?", (new_status, asset_id)
            )
            # Фидбэк — только при расхождении с ИИ (сигнал для доработки промпта).
            if contradiction and comment:
                dec = (
                    improver.DECISION_TO_STOCK
                    if action == "approve"
                    else improver.DECISION_FROM_STOCK
                )
                improver.add_feedback(
                    conn, asset_id, dec, comment, prompts.get_active_version(conn)
                )
            conn.commit()
        finally:
            conn.close()
        return jsonify({"ok": True, "status": new_status, "contradiction": contradiction})

    @app.get("/api/prompts")
    def prompt_list():
        conn = _conn()
        try:
            vers = prompts.list_versions(conn)
            out = [
                {
                    "version": v["version"],
                    "note": v["note"],
                    "source": v["source"],
                    "active": bool(v["is_active"]),
                    "at": v["created_at"],
                    "text": v["text"],
                }
                for v in vers
            ]
        finally:
            conn.close()
        return jsonify(out)

    @app.post("/api/prompts/activate")
    def prompt_activate():
        version = request.get_json(force=True).get("version")
        conn = _conn()
        try:
            prompts.activate_version(conn, int(version))
        finally:
            conn.close()
        return jsonify({"ok": True})

    @app.post("/api/prompts")
    def prompt_edit():
        data = request.get_json(force=True)
        text = (data.get("text") or "").strip()
        if not text:
            abort(400, "пустой промпт")
        conn = _conn()
        try:
            version = prompts.add_version(
                conn,
                text,
                note=data.get("note") or "Ручная правка",
                source=prompts.SOURCE_MANUAL,
                activate=True,
            )
        finally:
            conn.close()
        return jsonify({"ok": True, "version": version})

    return app


def run_server(cfg: Config, host: str = "0.0.0.0", port: int = 8000) -> None:
    """Запускает сервер, слушая LAN (доступен с телефона в той же сети)."""
    cfg.ensure_dirs()
    init_db(cfg.db_path)
    app = create_app(cfg)
    log.info("Веб-интерфейс запущен: http://localhost:%d (и по LAN-адресу ПК)", port)
    app.run(host=host, port=port, threaded=True)
