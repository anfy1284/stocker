"""Выгрузка на Shutterstock (трек загрузки, Шаг B): FTPS-заливка + CSV.

Для описанных (``described``) снимков собирает строгий CSV с метаданными и
заливает полноразмерные оригиналы на ``ftps.shutterstock.com`` (явный TLS).

Разделение по официальной схеме Shutterstock:
  * изображения (full-res, из ``assets.original_path``) → по FTPS;
  * CSV с метаданными — НЕ по FTPS: пользователь применяет его на сайте
    (Submit → кнопка «CSV»). Shutterstock сопоставляет строки CSV с залитыми
    файлами по колонке ``Filename`` — поэтому имя в CSV и имя на сервере обязаны
    совпадать (храним его в ``assets.upload_name``).

Доступы (логин/пароль контрибьютора) читаются из окружения через конфиг —
в коде и БД их нет; ввод и submit на сайте пользователь делает сам.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from datetime import datetime
from ftplib import FTP_TLS
from pathlib import Path

from .config import Config
from .db import STATUS_DESCRIBED, STATUS_UPLOADED, get_connection

log = logging.getLogger(__name__)

FTPS_HOST = "ftps.shutterstock.com"
FTPS_PORT = 21
CONNECT_TIMEOUT = 60  # секунд на установку соединения

# Точный заголовок CSV Shutterstock (порядок и написание строгие; любое
# отклонение → весь файл отклоняется). E–G опциональны, но задаём их явно «No»,
# так как наш контент коммерческий (не editorial, не иллюстрация, не 18+).
CSV_HEADER = [
    "Filename",
    "Description",
    "Keywords",
    "Categories",
    "Illustration",
    "Mature Content",
    "Editorial",
]

MAX_DESCRIPTION = 200
MAX_KEYWORDS = 50


def _safe_name(asset_id: int, original_path: str) -> str:
    """Имя файла для заливки: ``<id>_<безопасный-стем><ext>``.

    Префикс id гарантирует уникальность в аккаунте (файлы копятся до submit),
    а очистка стема — совместимость с FTP (без пробелов/юникода/спецсимволов).
    """
    p = Path(original_path)
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", p.stem).strip("_") or "img"
    ext = p.suffix.lower() or ".jpg"
    return f"{asset_id}_{stem}{ext}"


def _sanitize_keyword(kw: str) -> str:
    """Убирает запятые внутри ключевого слова (иначе поедут колонки CSV)."""
    return kw.replace(",", " ").strip()


def _build_records(conn) -> tuple[list[dict], int]:
    """Собирает записи описанных снимков к выгрузке. Возвращает (записи, пропущено).

    Пропускаются снимки, у которых пропал оригинал или (неожиданно) нет описания.
    """
    rows = conn.execute(
        "SELECT id, original_path, meta_description, meta_keywords, "
        "meta_category1, meta_category2 FROM assets WHERE status = ? ORDER BY id",
        (STATUS_DESCRIBED,),
    ).fetchall()

    records: list[dict] = []
    skipped = 0
    for r in rows:
        original = Path(r["original_path"])
        if not original.exists():
            log.warning("Снимок %d: оригинал не найден (%s) — пропуск", r["id"], original)
            skipped += 1
            continue
        description = (r["meta_description"] or "").strip()
        if not description:
            log.warning("Снимок %d: нет описания — пропуск", r["id"])
            skipped += 1
            continue

        keywords = [
            _sanitize_keyword(k)
            for k in json.loads(r["meta_keywords"] or "[]")
            if _sanitize_keyword(k)
        ][:MAX_KEYWORDS]
        categories = [c for c in (r["meta_category1"], r["meta_category2"]) if c]

        records.append(
            {
                "id": r["id"],
                "original_path": str(original),
                "upload_name": _safe_name(r["id"], r["original_path"]),
                "description": description[:MAX_DESCRIPTION],
                "keywords": keywords,
                "categories": categories,
            }
        )
    return records, skipped


def _write_csv(records: list[dict], path: Path) -> None:
    """Пишет строгий Shutterstock-CSV. csv-модуль сам квотит поля с запятыми."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for r in records:
            writer.writerow(
                [
                    r["upload_name"],
                    r["description"],
                    ",".join(r["keywords"]),
                    ",".join(r["categories"]),
                    "No",
                    "No",
                    "No",
                ]
            )


def _ftps_connect(cfg: Config) -> FTP_TLS:
    """Открывает FTPS-соединение с явным TLS и защищённым каналом данных."""
    ftp = FTP_TLS()
    ftp.connect(FTPS_HOST, FTPS_PORT, timeout=CONNECT_TIMEOUT)
    ftp.auth()  # AUTH TLS на управляющем канале до передачи логина/пароля
    ftp.login(cfg.shutterstock_user, cfg.shutterstock_password)
    ftp.prot_p()  # шифруем и канал данных
    return ftp


def run_upload(cfg: Config, dry_run: bool = False) -> dict:
    """Выгружает описанные снимки на Shutterstock. Возвращает статистику.

    ``dry_run`` — только собрать CSV, без соединения и смены статусов (для
    проверки формата). Без заданных FTPS-доступов CSV тоже пишется, но заливки
    не происходит — пользователь сначала задаёт доступы в ``.env``.
    """
    cfg.ensure_dirs()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = cfg.export_dir / f"shutterstock_{stamp}.csv"
    stats = {
        "total": 0,
        "uploaded": 0,
        "errors": 0,
        "skipped": 0,
        "sent": False,
        "error": None,  # текст фатальной ошибки соединения, иначе None
        "csv_path": str(csv_path),
    }

    conn = get_connection(cfg.db_path)
    try:
        records, skipped = _build_records(conn)
        stats["total"] = len(records)
        stats["skipped"] = skipped

        if not records:
            log.info("Нет описанных снимков к выгрузке.")
            _write_csv(records, csv_path)
            return stats

        # Пробный прогон или нет доступов: пишем CSV по всем кандидатам, не льём.
        if dry_run or not cfg.has_ftps_creds:
            _write_csv(records, csv_path)
            if dry_run:
                log.info("Пробный прогон: CSV на %d снимков — %s", len(records), csv_path)
            else:
                log.warning(
                    "FTPS-доступы не заданы (STOCKER_SHUTTERSTOCK_USER/"
                    "STOCKER_SHUTTERSTOCK_PASSWORD в .env) — заливка пропущена, "
                    "готов только CSV: %s",
                    csv_path,
                )
            return stats

        # Реальная заливка: коннектимся и льём оригиналы по одному.
        log.info("Подключаюсь к %s…", FTPS_HOST)
        try:
            ftp = _ftps_connect(cfg)
        except Exception as exc:  # noqa: BLE001 — показываем причину пользователю
            log.error("Не удалось подключиться к %s: %s", FTPS_HOST, exc)
            _write_csv(records, csv_path)  # CSV всё равно оставляем пользователю
            stats["error"] = str(exc)
            return stats
        uploaded: list[dict] = []
        try:
            now = datetime.now().isoformat()
            for r in records:
                try:
                    with open(r["original_path"], "rb") as fh:
                        ftp.storbinary(f"STOR {r['upload_name']}", fh)
                    conn.execute(
                        "UPDATE assets SET status = ?, upload_name = ?, uploaded_at = ? "
                        "WHERE id = ?",
                        (STATUS_UPLOADED, r["upload_name"], now, r["id"]),
                    )
                    conn.commit()
                    uploaded.append(r)
                    stats["uploaded"] += 1
                    log.info(
                        "Залит снимок %d → %s (%d/%d)",
                        r["id"],
                        r["upload_name"],
                        stats["uploaded"],
                        len(records),
                    )
                except Exception:
                    stats["errors"] += 1
                    log.exception("Ошибка заливки снимка %d", r["id"])
        finally:
            try:
                ftp.quit()
            except Exception:
                ftp.close()

        # CSV описывает ровно то, что легло на сервер (сопоставление по Filename).
        _write_csv(uploaded, csv_path)
        stats["sent"] = True
    finally:
        conn.close()

    log.info(
        "Выгрузка завершена: залито %(uploaded)d, ошибок %(errors)d, "
        "пропущено %(skipped)d. CSV: %(csv_path)s",
        stats,
    )
    return stats
