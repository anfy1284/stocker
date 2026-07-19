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

from . import organize
from .config import Config
from .db import (
    STATUS_APPROVED,
    STATUS_DESCRIBED,
    STATUS_NEW,
    get_connection,
)

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


def _asset_by_hash(conn, content_hash: str):
    """Снимок, у которого такое содержимое было *когда-либо* (id, путь) — или None.

    Ищем по всей истории хешей, а не только по текущему: тогда вернувшийся в
    inbox оригинал (версия до правки) опознаётся как дубль уже принятого снимка.
    """
    return conn.execute(
        "SELECT a.id, a.original_path FROM asset_hashes h "
        "JOIN assets a ON a.id = h.asset_id WHERE h.content_hash = ? LIMIT 1",
        (content_hash,),
    ).fetchone()


def _record_hash(conn, asset_id: int, content_hash: str) -> None:
    """Заносит хеш в историю снимка (идемпотентно)."""
    conn.execute(
        "INSERT OR IGNORE INTO asset_hashes (content_hash, asset_id, created_at) "
        "VALUES (?, ?, ?)",
        (content_hash, asset_id, datetime.now().isoformat()),
    )


def _handle_known(conn, asset, path: Path) -> str:
    """Реакция на файл, чей хеш уже в БД. Возвращает вид события для статистики.

    Дедуп по содержимому, а не по расположению: если канонический файл снимка
    существует в другом месте (снимок мог уехать в stock/approved), то этот —
    повторная копия загрузчика, её удаляем. Если канонический потерян — усыновляем
    эту копию как новый ``original_path`` (самолечение). Если это он и есть —
    просто оставляем.
    """
    canonical = Path(asset["original_path"]) if asset["original_path"] else None
    if canonical and canonical.resolve() == path.resolve():
        return "duplicate"  # это и есть канонический файл — не трогаем
    if canonical and canonical.exists():
        try:
            path.unlink()
            log.info("Удалён повторный дубль (канонич. файл на месте): %s", path.name)
            return "cleaned"
        except OSError:
            log.warning("Не удалось удалить повторный дубль: %s", path)
            return "duplicate"
    conn.execute(
        "UPDATE assets SET original_path = ? WHERE id = ?",
        (str(path.resolve()), asset["id"]),
    )
    conn.commit()
    log.info("Канонический файл был потерян — усыновлена копия: %s", path.name)
    return "repaired"


def _process_file(path: Path, content_hash: str, previews_dir: Path) -> dict[str, object]:
    """Собирает запись ``assets`` для нового файла (EXIF + превью)."""
    preview_path = previews_dir / f"{content_hash}.jpg"
    with Image.open(path) as img:
        exif = _read_exif(img)
        width, height = _make_preview(img, preview_path)

    stat = path.stat()
    return {
        "original_path": str(path.resolve()),
        "preview_path": str(preview_path),
        "content_hash": content_hash,
        "file_type": FILE_TYPE_JPEG,
        "file_size": stat.st_size,
        "original_mtime": stat.st_mtime,
        "width": width,
        "height": height,
        "captured_at": exif["captured_at"],
        "camera_make": exif["camera_make"],
        "camera_model": exif["camera_model"],
        "orientation": exif["orientation"],
        "status": STATUS_NEW,
        "created_at": datetime.now().isoformat(),
    }


def _insert(conn, record: dict[str, object]) -> int:
    """Вставляет запись снимка, возвращает её id."""
    columns = ", ".join(record)
    placeholders = ", ".join(f":{k}" for k in record)
    cur = conn.execute(
        f"INSERT INTO assets ({columns}) VALUES ({placeholders})", record
    )
    return int(cur.lastrowid)


def _drop_cached(cfg: Config, preview_path: str | None) -> None:
    """Удаляет устаревшее превью и его миниатюру (после смены хеша снимка)."""
    if not preview_path:
        return
    name = Path(preview_path).name
    for cached in (cfg.previews_dir / name, cfg.thumbs_dir / name):
        try:
            cached.unlink(missing_ok=True)
        except OSError:
            log.warning("Не удалось удалить устаревший кеш: %s", cached)


def _apply_edit(conn, cfg: Config, asset, path: Path, new_hash: str) -> None:
    """Правка снимка на месте: та же карточка, новое содержимое.

    Перехешируем, пересобираем превью под новым именем (``{new_hash}.jpg`` — сам
    по себе сбрасывает кеши браузера и миниатюр), чистим старое превью/миниатюру,
    обновляем EXIF/размеры/размер файла. Решения не трогаем — снимок тот же.

    Метаданные Shutterstock (описание/ключи/категории) генерились по прежним
    пикселям и после правки могут не соответствовать кадру (напр. с фото убрали
    людей — «Hikers exploring…» уже неверно), поэтому их сбрасываем всегда —
    универсально для любого статуса. Описанный снимок (ещё не выгружен) при этом
    возвращаем в «Одобрено»: карточка станет «НЕТ МЕТА» и снова попадёт в
    генерацию метаданных. Выгруженный оставляем выгруженным (мету всё равно
    чистим, но больше ничего с ним не делаем). Файл из своей папки НЕ двигаем
    (approved и described живут в одной папке, так что перекладка и не нужна).
    Вердикт классификатора («Причина»/категория/флаги) — с предыдущего этапа,
    оставляем. Новый хеш дописываем в историю (старый там остаётся — для
    опознания вернувшегося оригинала).
    """
    new_preview = cfg.previews_dir / f"{new_hash}.jpg"
    with Image.open(path) as img:
        exif = _read_exif(img)
        width, height = _make_preview(img, new_preview)

    stat = path.stat()
    # «Описано» → «Одобрено» (снова ждёт меты). Остальные статусы не трогаем.
    new_status = (
        STATUS_APPROVED if asset["status"] == STATUS_DESCRIBED else asset["status"]
    )
    conn.execute(
        "UPDATE assets SET content_hash = ?, preview_path = ?, file_size = ?, "
        "original_mtime = ?, width = ?, height = ?, captured_at = ?, "
        "camera_make = ?, camera_model = ?, orientation = ?, "
        "meta_description = NULL, meta_keywords = NULL, meta_category1 = NULL, "
        "meta_category2 = NULL, meta_generated_at = NULL, status = ? WHERE id = ?",
        (
            new_hash,
            str(new_preview),
            stat.st_size,
            stat.st_mtime,
            width,
            height,
            exif["captured_at"],
            exif["camera_make"],
            exif["camera_model"],
            exif["orientation"],
            new_status,
            asset["id"],
        ),
    )
    _record_hash(conn, asset["id"], new_hash)
    conn.commit()
    if Path(asset["preview_path"] or "").name != new_preview.name:
        _drop_cached(cfg, asset["preview_path"])


def _detect_edits(conn, cfg: Config, stats: dict[str, int]) -> None:
    """Ищет правки уже принятых файлов на месте (в папках любой кучи).

    Правка снимка внешним редактором — штатное действие, но файл лежит не в
    inbox, а в папке своей кучи, поэтому обычный скан его не видит. Здесь дёшево
    (по mtime/размеру) отбираем изменившиеся файлы и перехешируем только их; при
    смене хеша обновляем ту же карточку через :func:`_apply_edit`. На первом
    прогоне после обновления схемы ``original_mtime`` пуст — тогда сверяем хеш
    у всех, чтобы поймать правки, сделанные до появления этой логики.
    """
    rows = conn.execute(
        "SELECT id, original_path, content_hash, preview_path, file_size, "
        "original_mtime, status FROM assets"
    ).fetchall()
    for row in rows:
        if not row["original_path"]:
            continue
        path = Path(row["original_path"])
        if not path.exists():
            continue
        try:
            stat = path.stat()
            unchanged = (
                row["original_mtime"] is not None
                and abs(stat.st_mtime - row["original_mtime"]) < 1e-6
                and stat.st_size == row["file_size"]
            )
            if unchanged:
                continue
            new_hash = _hash_file(path)
            if new_hash == row["content_hash"]:
                # Содержимое то же (правились лишь мета/время файла) — только
                # запоминаем свежий mtime, чтобы больше не перехешировать зря.
                conn.execute(
                    "UPDATE assets SET original_mtime = ? WHERE id = ?",
                    (stat.st_mtime, row["id"]),
                )
                conn.commit()
                continue
            _apply_edit(conn, cfg, row, path, new_hash)
            stats["edited"] += 1
            log.info("Обновлён снимок после правки на месте: %s", path.name)
        except Exception:
            stats["errors"] += 1
            log.exception("Ошибка при обработке правки файла: %s", path)


def run_intake(cfg: Config) -> dict[str, int]:
    """Сканирует входящие и заводит новые JPEG в БД. Возвращает статистику."""
    stats = {
        "scanned": 0,
        "added": 0,
        "edited": 0,
        "duplicate": 0,
        "cleaned": 0,
        "repaired": 0,
        "errors": 0,
    }
    conn = get_connection(cfg.db_path)
    try:
        # Сперва подхватываем правки уже принятых файлов (в папках их куч),
        # затем сканируем inbox на новые/вернувшиеся исходники.
        _detect_edits(conn, cfg, stats)
        for path in _scan(cfg.inbox_dir):
            stats["scanned"] += 1
            try:
                content_hash = _hash_file(path)
                known = _asset_by_hash(conn, content_hash)
                if known is not None:
                    stats[_handle_known(conn, known, path)] += 1
                    continue
                record = _process_file(path, content_hash, cfg.previews_dir)
                asset_id = _insert(conn, record)
                _record_hash(conn, asset_id, content_hash)
                conn.commit()
                stats["added"] += 1
                log.info("Принят: %s", path.name)
            except Exception:
                stats["errors"] += 1
                log.exception("Ошибка при приёме файла: %s", path)
    finally:
        conn.close()

    # Обновляем манифест известных хешей — по нему будущий загрузчик не грузит
    # повторно уже принятые исходники (где бы их файлы сейчас ни лежали).
    organize.write_manifest(cfg)

    log.info(
        "Приём завершён: найдено %(scanned)d, добавлено %(added)d, обновлено "
        "правок %(edited)d, дублей %(duplicate)d, вычищено копий %(cleaned)d, "
        "восстановлено %(repaired)d, ошибок %(errors)d",
        stats,
    )
    return stats
