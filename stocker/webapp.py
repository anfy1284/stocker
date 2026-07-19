"""Веб-интерфейс ревью (M4, Шаг 5): локальный Flask-сервер.

Показывает кучи снимков (сток/не-сток — необработанные; одобрено/брак —
обработанные), даёт пользователю решать «в сток / из стока» (свайпом или
кнопками). Решение переводит снимок в очередь на отправку либо в брак; при
расхождении с вердиктом ИИ пользователь поясняет решение — пояснение копится
в фидбэк и позже дорабатывает промпт. Отдельно — история версий промпта
(активировать/откатить/редактировать).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

import anthropic
from flask import Flask, abort, jsonify, request, send_file, send_from_directory

from . import classifier, improver, intake, metadata, organize, prompts, upload
from .config import Config, load_config
from .db import (
    STATUS_APPROVED,
    STATUS_DESCRIBED,
    STATUS_NEW,
    STATUS_NON_STOCK,
    STATUS_PREFILTERED,
    STATUS_REJECTED,
    STATUS_STOCK_CANDIDATE,
    STATUS_UPLOADED,
    get_connection,
    init_db,
)

log = logging.getLogger(__name__)
WEB_DIR = Path(__file__).parent / "web"

# Кучи, доступные в интерфейсе (в порядке показа). «Нераспределённые» (new) —
# первыми: сюда попадают новые файлы из inbox, ждущие ИИ-разбора.
PILES = (
    STATUS_NEW,
    STATUS_STOCK_CANDIDATE,
    STATUS_NON_STOCK,
    STATUS_PREFILTERED,
    STATUS_APPROVED,
    STATUS_UPLOADED,
    STATUS_REJECTED,
)

# Суточный интервал авто-разбора (приём + классификация новых файлов).
_AUTO_INTERVAL = 24 * 3600
# Частый фоновый приём — чтобы новые файлы из inbox быстро попадали в «Нераспределённые».
_INTAKE_INTERVAL = 60

# Статусы «одобренной» кучи: одобрены пользователем на триаже. Внутри —
# ``described`` (метаданные готовы) и ``approved`` (метаданные ещё не сгенерированы).
APPROVED_PILE = (STATUS_APPROVED, STATUS_DESCRIBED)

# Размер миниатюры для сетки (полное превью 1600px — только в крупном просмотре).
THUMB_MAX_SIDE = 320


def create_app(cfg: Config | None = None, enable_scheduler: bool = False) -> Flask:
    cfg = cfg or load_config()
    app = Flask(__name__)

    def _conn():
        return get_connection(cfg.db_path)

    # --- Фоновый разбор «Нераспределённых»: приём из inbox + пакетная классификация.
    # Запускается кнопкой в интерфейсе и автоматически раз в сутки (планировщик).
    cls_lock = threading.Lock()
    cls_state = {
        "running": False,
        "batch": False,  # True = пакетный (Batch API, долго); False = синхронный
        "phase": None,   # intake | classify | None
        "added": 0,      # принято новых файлов на этапе intake
        "total": 0,      # снимков в разборе
        "done": 0,
        "stock": 0,
        "non_stock": 0,
        "errors": 0,
        "error": None,   # фатальная ошибка (нет ключа, сбой батча)
    }

    def _classify_worker(use_batch: bool):
        try:
            with cls_lock:
                cls_state["phase"] = "intake"
            added = intake.run_intake(cfg)["added"]
            with cls_lock:
                cls_state.update(added=added, phase="classify")

            def on_progress(s: dict) -> None:
                with cls_lock:
                    cls_state.update(
                        total=s["total"], done=s["done"], stock=s["stock"],
                        non_stock=s["non_stock"], errors=s["errors"],
                    )

            classifier.run_classification(cfg, on_progress=on_progress, use_batch=use_batch)
        except Exception as exc:  # noqa: BLE001 — показываем причину пользователю
            log.exception("Фатальная ошибка авто-разбора")
            with cls_lock:
                cls_state["error"] = str(exc)
        finally:
            with cls_lock:
                cls_state.update(running=False, phase=None)

    def _start_classify(use_batch: bool) -> bool:
        """Запускает разбор, если он ещё не идёт. Возвращает, стартовал ли.

        ``use_batch=False`` — синхронно (кнопка, быстрый отклик); ``True`` —
        пакетно через Batch API (суточный авто-разбор, вдвое дешевле).
        """
        with cls_lock:
            if cls_state["running"]:
                return False
            cls_state.update(
                running=True, batch=use_batch, phase=None, added=0, total=0, done=0,
                stock=0, non_stock=0, errors=0, error=None,
            )
        threading.Thread(
            target=_classify_worker, args=(use_batch,), daemon=True
        ).start()
        return True

    # --- Фоновый локальный предотбор: приём из inbox + прогон нейросети на GPU.
    # Бесплатно отсеивает явный не-сток до платного классификатора. Тяжёлый
    # torch/open_clip импортируется лениво в воркере — сервер стартует без них.
    pf_lock = threading.Lock()
    pf_state = {
        "running": False,
        "phase": None,   # intake | prefilter | None
        "added": 0,      # принято новых файлов на этапе intake
        "total": 0,
        "done": 0,
        "kept": 0,       # оставлено (пойдут в API)
        "filtered": 0,   # отсеяно в «Предотсев»
        "errors": 0,
        "error": None,   # фатальная ошибка (нет зависимостей и т.п.)
    }

    def _prefilter_worker():
        try:
            with pf_lock:
                pf_state["phase"] = "intake"
            added = intake.run_intake(cfg)["added"]
            with pf_lock:
                pf_state.update(added=added, phase="prefilter")

            from . import prefilter

            def on_progress(s: dict) -> None:
                with pf_lock:
                    pf_state.update(
                        total=s["total"], done=s["done"], kept=s["kept"],
                        filtered=s["filtered"], errors=s["errors"],
                    )

            prefilter.run_prefilter(cfg, on_progress=on_progress)
        except Exception as exc:  # noqa: BLE001 — показываем причину пользователю
            log.exception("Фатальная ошибка предотбора")
            with pf_lock:
                pf_state["error"] = str(exc)
        finally:
            with pf_lock:
                pf_state.update(running=False, phase=None)

    def _start_prefilter() -> bool:
        with pf_lock:
            if pf_state["running"]:
                return False
            pf_state.update(
                running=True, phase=None, added=0, total=0, done=0, kept=0,
                filtered=0, errors=0, error=None,
            )
        threading.Thread(target=_prefilter_worker, daemon=True).start()
        return True

    def _scheduler():
        """Раз в сутки проверяет inbox и разбирает новые файлы (авто-режим)."""
        time.sleep(5)  # дать серверу подняться
        while True:
            if cfg.has_api_key:
                # Суточный разбор — пакетно (дёшево, скорость не важна).
                if _start_classify(use_batch=True):
                    log.info("Авто-разбор: запущена суточная проверка новых файлов.")
            time.sleep(_AUTO_INTERVAL)

    def _intake_watch():
        """Часто принимает новые файлы из inbox, чтобы они быстро появлялись в
        «Нераспределённых» (без разбора — тот по кнопке/раз в сутки). Приём дешёвый
        и не ходит в ИИ. Пропускаем, если уже идёт разбор (он сам сделает приём)."""
        while True:
            time.sleep(_INTAKE_INTERVAL)
            try:
                with cls_lock:
                    busy = cls_state["running"]
                if not busy:
                    intake.run_intake(cfg)
            except Exception:  # noqa: BLE001 — фоновая задача, не роняем сервер
                log.exception("Ошибка фонового приёма новых файлов")

    def _startup_relocate():
        """Разово раскладывает уже имеющиеся файлы по папкам их статуса.

        Нужно после изменения раскладки (напр. появления отдельных папок «Не-Сток»
        и «Брак»): старые файлы, лежавшие в общей inbox, разъезжаются по своим
        папкам. Идемпотентно и устойчиво к занятым файлам — что не вышло, добьёт
        суточный авто-разбор/ручной ``stocker organize``. В фоне, чтобы не
        задерживать старт сервера на большом архиве."""
        try:
            organize.run_organize(cfg)
        except Exception:  # noqa: BLE001 — фоновая задача, не роняем сервер
            log.exception("Ошибка стартовой раскладки файлов по папкам")

    if enable_scheduler:
        threading.Thread(target=_startup_relocate, daemon=True).start()
        threading.Thread(target=_scheduler, daemon=True).start()
        threading.Thread(target=_intake_watch, daemon=True).start()

    # --- Фоновый процесс генерации метаданных ------------------------------
    # Один процесс на сервер: пользователь жмёт кнопку в шапке, генерация идёт
    # в отдельном потоке, интерфейс опрашивает статус и рисует прогресс-бар.
    meta_lock = threading.Lock()
    meta_state = {
        "running": False,
        "total": 0,
        "done": 0,
        "errors": 0,
        "error": None,  # текст фатальной ошибки (нет ключа и т.п.), иначе None
    }

    def _meta_worker():
        def on_progress(stats: dict[str, int]) -> None:
            with meta_lock:
                meta_state.update(stats)

        try:
            metadata.run_metadata(cfg, on_progress=on_progress)
        except Exception as exc:  # noqa: BLE001 — показываем текст пользователю
            log.exception("Фатальная ошибка генерации метаданных")
            with meta_lock:
                meta_state["error"] = str(exc)
        finally:
            with meta_lock:
                meta_state["running"] = False

    # --- Фоновая выгрузка на Shutterstock (аналогично метаданным) ----------
    up_lock = threading.Lock()
    up_state = {
        "running": False,
        "total": 0,
        "uploaded": 0,
        "errors": 0,
        "error": None,     # фатальная ошибка (нет доступов, обрыв соединения)
        "csv_name": None,  # имя готового CSV в export_dir (для скачивания в браузер)
    }

    def _upload_worker():
        def on_progress(stats: dict) -> None:
            with up_lock:
                up_state.update(
                    total=stats["total"],
                    uploaded=stats["uploaded"],
                    errors=stats["errors"],
                    # CSV пишется до заливки — имя появляется с первым же прогрессом,
                    # веб успевает скачать файл, не дожидаясь конца заливки.
                    csv_name=Path(stats["csv_path"]).name,
                )

        try:
            stats = upload.run_upload(cfg, on_progress=on_progress)
            with up_lock:
                up_state["csv_name"] = Path(stats["csv_path"]).name
                if stats.get("error"):
                    up_state["error"] = stats["error"]
        except Exception as exc:  # noqa: BLE001 — показываем текст пользователю
            log.exception("Фатальная ошибка выгрузки на Shutterstock")
            with up_lock:
                up_state["error"] = str(exc)
        finally:
            with up_lock:
                up_state["running"] = False

    @app.get("/")
    def index():
        # Без кэша: после обновления кода браузер всегда берёт свежий интерфейс.
        resp = send_file(WEB_DIR / "index.html")
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return resp

    @app.get("/previews/<path:name>")
    def preview(name: str):
        # Имя = хеш содержимого: после правки фото меняется само имя, поэтому
        # достаточно revalidate — старый URL не переиспользуется.
        resp = send_from_directory(cfg.previews_dir, name)
        resp.headers["Cache-Control"] = "no-cache"
        return resp

    @app.get("/thumb/<path:name>")
    def thumb(name: str):
        """Маленькая миниатюра для сетки (кэшируется на диск) — лёгкая загрузка."""
        thumbs = cfg.thumbs_dir
        thumbs.mkdir(parents=True, exist_ok=True)
        dest = thumbs / name
        src = cfg.previews_dir / name
        if not src.exists():
            abort(404)
        # Пересобираем, если миниатюры нет или превью новее (страховка на случай,
        # если файл превью переписан под тем же именем).
        if not dest.exists() or src.stat().st_mtime > dest.stat().st_mtime:
            from PIL import Image

            with Image.open(src) as im:
                im = im.convert("RGB")
                im.thumbnail((THUMB_MAX_SIDE, THUMB_MAX_SIDE))
                im.save(dest, "JPEG", quality=80)
        resp = send_from_directory(thumbs, name)
        resp.headers["Cache-Control"] = "no-cache"
        return resp

    def _count(conn, status: str) -> int:
        return conn.execute(
            "SELECT count(1) FROM assets WHERE status = ?", (status,)
        ).fetchone()[0]

    @app.get("/api/summary")
    def summary():
        conn = _conn()
        try:
            counts = {s: _count(conn, s) for s in PILES}
            described = _count(conn, STATUS_DESCRIBED)
            # Куча «Одобрено» = одобренные + уже описанные (метаданные готовы).
            counts[STATUS_APPROVED] += described
            # Сколько одобренных ещё ждут генерации метаданных (для баннера/кнопки).
            counts["meta_pending"] = _count(conn, STATUS_APPROVED)
        finally:
            conn.close()
        return jsonify(counts)

    @app.get("/api/assets")
    def assets():
        status = request.args.get("status", STATUS_STOCK_CANDIDATE)
        # «Одобрено» — сводная куча: одобренные + описанные (метаданные готовы).
        wanted = APPROVED_PILE if status == STATUS_APPROVED else (status,)
        placeholders = ",".join("?" * len(wanted))
        conn = _conn()
        try:
            rows = conn.execute(
                "SELECT id, original_path, preview_path, status, category, "
                "classification_reason, has_logo, has_brand, has_text, "
                "meta_description, meta_keywords, meta_category1, meta_category2, "
                "meta_generated_at, upload_name, uploaded_at, file_deleted, "
                "prefilter_aesthetic, prefilter_reason, prefilter_hard_reject "
                f"FROM assets WHERE status IN ({placeholders}) ORDER BY id",
                wanted,
            ).fetchall()
            out = []
            for r in rows:
                # Отработанные пользователь удаляет из папки done — ловим пропажу
                # оригинала лениво (запись и превью остаются, флаг проставляется).
                file_deleted = bool(r["file_deleted"])
                if (
                    r["status"] == STATUS_UPLOADED
                    and not file_deleted
                    and r["original_path"]
                    and not Path(r["original_path"]).exists()
                ):
                    conn.execute(
                        "UPDATE assets SET file_deleted = 1 WHERE id = ?", (r["id"],)
                    )
                    conn.commit()
                    file_deleted = True
                fb = conn.execute(
                    "SELECT comment, created_at FROM feedback "
                    "WHERE asset_id = ? AND comment IS NOT NULL ORDER BY id",
                    (r["id"],),
                ).fetchall()
                meta = None
                if r["meta_description"]:
                    meta = {
                        "description": r["meta_description"],
                        "keywords": json.loads(r["meta_keywords"] or "[]"),
                        "category1": r["meta_category1"],
                        "category2": r["meta_category2"],
                        "at": r["meta_generated_at"],
                    }
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
                        "meta": meta,
                        "upload": (
                            {"name": r["upload_name"], "at": r["uploaded_at"]}
                            if r["upload_name"]
                            else None
                        ),
                        "file_deleted": file_deleted,
                        "prefilter": (
                            {
                                "aesthetic": r["prefilter_aesthetic"],
                                "reason": r["prefilter_reason"],
                                "hard": bool(r["prefilter_hard_reject"]),
                            }
                            if r["status"] == STATUS_PREFILTERED
                            else None
                        ),
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
            if action == "approve":
                # Спасение из «Предотсева» — назад в разбор (снимок ещё не видел
                # платного классификатора), а не сразу в одобренные.
                new_status = STATUS_NEW if old == STATUS_PREFILTERED else STATUS_APPROVED
            else:
                new_status = STATUS_REJECTED
            conn.execute(
                "UPDATE assets SET status = ? WHERE id = ?", (new_status, asset_id)
            )
            # Одобренные уезжают в approved/ (фотошоп перед выгрузкой), брак — в inbox.
            organize.relocate(conn, cfg, asset_id, new_status)
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

    @app.post("/api/piles/non_stock/reject_all")
    def reject_all_non_stock():
        """Массово переносит всю кучу «Не-Сток» в брак (пользователь согласен с ИИ).

        Расхождения с ИИ здесь нет (ИИ отсеял, пользователь подтверждает), поэтому
        фидбэк не пишем — только смена статуса.
        """
        conn = _conn()
        try:
            cur = conn.execute(
                "UPDATE assets SET status = ? WHERE status = ?",
                (STATUS_REJECTED, STATUS_NON_STOCK),
            )
            conn.commit()
            count = cur.rowcount
        finally:
            conn.close()
        log.info("Массовый перенос в брак: %d снимков из «Не-Сток»", count)
        return jsonify({"ok": True, "count": count})

    @app.post("/api/classify/run")
    def classify_run():
        """Запускает разбор новых файлов. ?mode=batch — пакетно (дёшево, долго),
        иначе синхронно (быстро)."""
        if not cfg.has_api_key:
            return jsonify({"ok": False, "error": "no_key"}), 400
        use_batch = request.args.get("mode") == "batch"
        if not _start_classify(use_batch=use_batch):
            return jsonify({"ok": False, "running": True}), 409
        return jsonify({"ok": True})

    @app.get("/api/classify/status")
    def classify_status():
        """Текущее состояние разбора (для прогресс-бара)."""
        with cls_lock:
            state = dict(cls_state)
        state["has_key"] = cfg.has_api_key
        return jsonify(state)

    @app.post("/api/intake/run")
    def intake_run():
        """Принудительный приём: сразу подхватить новые файлы из inbox в
        «Нераспределённые» (без разбора). ИИ не задействован."""
        added = intake.run_intake(cfg)["added"]
        return jsonify({"ok": True, "added": added})

    @app.post("/api/prefilter/run")
    def prefilter_run():
        """Запускает локальный предотбор (приём из inbox + прогон нейросети на
        GPU). Бесплатно отсеивает явный не-сток до платного классификатора."""
        if not _start_prefilter():
            return jsonify({"ok": False, "running": True}), 409
        return jsonify({"ok": True})

    @app.get("/api/prefilter/status")
    def prefilter_status():
        """Текущее состояние предотбора (для прогресс-бара)."""
        with pf_lock:
            return jsonify(dict(pf_state))

    @app.post("/api/piles/prefiltered/delete_all")
    def delete_all_prefiltered():
        """Безвозвратно удаляет всю кучу «Предотсев»: файлы-оригиналы, превью,
        миниатюры и записи. Освобождает диск от локально отсеянного не-стока.

        Необратимо (в интерфейсе — под подтверждением). Порядок: сначала файлы
        (best-effort — отсутствие/занятость не срывают чистку), затем зависимые
        строки (история хешей и фидбэк ссылаются на снимок по внешнему ключу) и
        сама запись. Манифест хешей после этого перестраивается: удалённый мусор
        больше не «известен», поэтому при повторной загрузке будет принят заново
        и снова отсеян локально (бесплатно) — платного разбора это не касается.
        """
        conn = _conn()
        deleted = 0
        try:
            rows = conn.execute(
                "SELECT id, original_path, preview_path FROM assets WHERE status = ?",
                (STATUS_PREFILTERED,),
            ).fetchall()
            for r in rows:
                for p in (r["original_path"], r["preview_path"]):
                    if p:
                        try:
                            Path(p).unlink(missing_ok=True)
                        except OSError:
                            log.warning("Предотсев: не удалить файл %s", p)
                if r["preview_path"]:
                    try:
                        (cfg.thumbs_dir / Path(r["preview_path"]).name).unlink(
                            missing_ok=True
                        )
                    except OSError:
                        pass
                conn.execute("DELETE FROM asset_hashes WHERE asset_id = ?", (r["id"],))
                conn.execute("DELETE FROM feedback WHERE asset_id = ?", (r["id"],))
                conn.execute("DELETE FROM assets WHERE id = ?", (r["id"],))
                deleted += 1
            conn.commit()
        finally:
            conn.close()
        organize.write_manifest(cfg)
        log.info("Удалена куча «Предотсев»: %d снимков (файлы + записи)", deleted)
        return jsonify({"ok": True, "count": deleted})

    @app.post("/api/metadata/run")
    def metadata_run():
        """Запускает фоновую генерацию метаданных для одобренных снимков."""
        with meta_lock:
            if meta_state["running"]:
                return jsonify({"ok": False, "running": True}), 409
            meta_state.update(running=True, total=0, done=0, errors=0, error=None)
        threading.Thread(target=_meta_worker, daemon=True).start()
        return jsonify({"ok": True})

    @app.get("/api/metadata/status")
    def metadata_status():
        """Текущее состояние генерации метаданных (для прогресс-бара)."""
        with meta_lock:
            return jsonify(dict(meta_state))

    @app.post("/api/upload/run")
    def upload_run():
        """Запускает фоновую выгрузку описанных снимков на Shutterstock."""
        if not cfg.has_ftps_creds:
            # Без доступов льём впустую — сразу честно говорим интерфейсу.
            return jsonify({"ok": False, "error": "no_creds"}), 400
        with up_lock:
            if up_state["running"]:
                return jsonify({"ok": False, "running": True}), 409
            up_state.update(
                running=True, total=0, uploaded=0, errors=0, error=None, csv_path=None
            )
        threading.Thread(target=_upload_worker, daemon=True).start()
        return jsonify({"ok": True})

    @app.get("/api/upload/status")
    def upload_status():
        """Текущее состояние выгрузки (для прогресс-бара)."""
        with up_lock:
            state = dict(up_state)
        state["has_creds"] = cfg.has_ftps_creds
        return jsonify(state)

    @app.get("/api/upload/csv/<path:name>")
    def upload_csv(name: str):
        """Отдаёт готовый CSV из export_dir на скачивание в браузер.

        Файл сформирован до заливки — пользователь может скачать его сразу
        (и повторно), даже если сидит не за серверным ПК. ``send_from_directory``
        сам защищает от выхода за пределы каталога.
        """
        return send_from_directory(cfg.export_dir, name, as_attachment=True)

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
    app = create_app(cfg, enable_scheduler=True)  # суточный авто-разбор новых файлов
    log.info("Веб-интерфейс запущен: http://localhost:%d (и по LAN-адресу ПК)", port)
    app.run(host=host, port=port, threaded=True)
