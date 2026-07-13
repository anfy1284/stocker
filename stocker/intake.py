"""Приём (M1): регистрация файлов из папки-входящих в БД.

Шаг 2 работает только с JPEG. Для каждого нового файла: хеш содержимого и
дедупликация, чтение EXIF (дата съёмки, камера, ориентация), генерация
нормализованного JPEG-превью и запись в ``assets`` со статусом «новый».
Никаких решений о «стоковости» — только регистрация.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from pathlib import Path

from PIL import ExifTags, Image, ImageOps

from .config import Config
from .db import STATUS_NEW, get_connection

log = logging.getLogger(__name__)

# Шаг 2 — только JPEG. RAW/HEIC добавит Шаг 6.
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg"}
FILE_TYPE_JPEG = "jpeg"

# Нормализованное превью: 1600px по длинной стороне, JPEG q85.
PREVIEW_MAX_SIDE = 1600
PREVIEW_QUALITY = 85

_HASH_CHUNK = 1 << 20  # 1 МБ


def _hash_file(path: Path) -> str:
    """SHA-256 содержимого файла (для дедупликации точных дублей)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_exif_datetime(value: str) -> str | None:
    """EXIF-дата «YYYY:MM:DD HH:MM:SS» → ISO-строка; None, если не разобрать."""
    try:
        return datetime.strptime(value.strip(), "%Y:%m:%d %H:%M:%S").isoformat()
    except (ValueError, AttributeError):
        return None


def _read_exif(img: Image.Image) -> dict[str, object]:
    """Извлекает дату съёмки, камеру и ориентацию из EXIF (что есть)."""
    result: dict[str, object] = {
        "captured_at": None,
        "camera_make": None,
        "camera_model": None,
        "orientation": None,
    }
    exif = img.getexif()
    if not exif:
        return result

    make = exif.get(ExifTags.Base.Make)
    model = exif.get(ExifTags.Base.Model)
    orientation = exif.get(ExifTags.Base.Orientation)
    result["camera_make"] = str(make).strip() if make else None
    result["camera_model"] = str(model).strip() if model else None
    result["orientation"] = int(orientation) if orientation else None

    exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
    dto = exif_ifd.get(ExifTags.Base.DateTimeOriginal)
    if dto:
        result["captured_at"] = _parse_exif_datetime(str(dto))
    return result


def _make_preview(img: Image.Image, dest: Path) -> tuple[int, int]:
    """Пишет нормализованное превью; возвращает размеры кадра в его ориентации.

    Ориентация из EXIF применяется к пикселям (``exif_transpose``), поэтому
    возвращаемые размеры — «как видит человек».
    """
    oriented = ImageOps.exif_transpose(img)
    width, height = oriented.size
    if oriented.mode != "RGB":
        oriented = oriented.convert("RGB")
    oriented.thumbnail((PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE))
    dest.parent.mkdir(parents=True, exist_ok=True)
    oriented.save(dest, "JPEG", quality=PREVIEW_QUALITY)
    return width, height


def _scan(inbox_dir: Path) -> list[Path]:
    """Файлы поддерживаемых форматов в папке-входящих (рекурсивно)."""
    if not inbox_dir.exists():
        return []
    return sorted(
        p
        for p in inbox_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _hash_exists(conn, content_hash: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM assets WHERE content_hash = ? LIMIT 1", (content_hash,)
    ).fetchone()
    return row is not None


def _process_file(path: Path, content_hash: str, previews_dir: Path) -> dict[str, object]:
    """Собирает запись ``assets`` для нового файла (EXIF + превью)."""
    preview_path = previews_dir / f"{content_hash}.jpg"
    with Image.open(path) as img:
        exif = _read_exif(img)
        width, height = _make_preview(img, preview_path)

    return {
        "original_path": str(path.resolve()),
        "preview_path": str(preview_path),
        "content_hash": content_hash,
        "file_type": FILE_TYPE_JPEG,
        "file_size": path.stat().st_size,
        "width": width,
        "height": height,
        "captured_at": exif["captured_at"],
        "camera_make": exif["camera_make"],
        "camera_model": exif["camera_model"],
        "orientation": exif["orientation"],
        "status": STATUS_NEW,
        "created_at": datetime.now().isoformat(),
    }


def _insert(conn, record: dict[str, object]) -> None:
    columns = ", ".join(record)
    placeholders = ", ".join(f":{k}" for k in record)
    conn.execute(f"INSERT INTO assets ({columns}) VALUES ({placeholders})", record)


def run_intake(cfg: Config) -> dict[str, int]:
    """Сканирует входящие и заводит новые JPEG в БД. Возвращает статистику."""
    stats = {"scanned": 0, "added": 0, "duplicate": 0, "errors": 0}
    conn = get_connection(cfg.db_path)
    try:
        for path in _scan(cfg.inbox_dir):
            stats["scanned"] += 1
            try:
                content_hash = _hash_file(path)
                if _hash_exists(conn, content_hash):
                    stats["duplicate"] += 1
                    log.info("Дубль, пропуск: %s", path.name)
                    continue
                record = _process_file(path, content_hash, cfg.previews_dir)
                _insert(conn, record)
                conn.commit()
                stats["added"] += 1
                log.info("Принят: %s", path.name)
            except Exception:
                stats["errors"] += 1
                log.exception("Ошибка при приёме файла: %s", path)
    finally:
        conn.close()

    log.info(
        "Приём завершён: найдено %(scanned)d, добавлено %(added)d, "
        "дублей %(duplicate)d, ошибок %(errors)d",
        stats,
    )
    return stats
