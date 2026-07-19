"""Физическая раскладка файлов по папкам согласно статусу + манифест дедупа.

Идея: **папка = функция статуса**, ровно одна физическая папка на каждую «кучу»
интерфейса. Файл всегда лежит там, где велит его статус, а ``assets.original_path``
всегда указывает на его текущее место. Так снимки любой кучи можно обрабатывать
внешними средствами (Photoshop/Lightroom для одобренных, ручное/батником удаление
брака и выгруженных), не заходя в приложение:

  * ``new``                        → ``inbox_dir``     («Нераспределённые», приём из inbox)
  * ``stock_candidate``            → ``stock_dir``     («Сток» — отбирать/фотошопить)
  * ``non_stock``                  → ``non_stock_dir`` («Не-Сток»)
  * ``prefiltered``                → ``prefiltered_dir`` («Предотсев» — локальный нейрофильтр)
  * ``approved`` / ``described``   → ``approved_dir``  («Одобрено» — фотошопить перед выгрузкой)
  * ``rejected``                   → ``rejected_dir``  («Брак» — можно удалять)
  * ``uploaded``                   → ``done_dir``      («Выгружено» — отработанные, можно удалять)

Каждая куча = отдельная папка; добавление новой кучи в будущем — новая папка в
``Config`` и строка в ``_STATUS_DIR`` ниже. Перемещение вызывается на переходах
статуса (классификатор, ревью, выгрузка).
Оно устойчиво: если файл занят (открыт в Photoshop) и переместить не вышло —
статус всё равно меняется, файл остаётся на месте, а ``run_organize`` позже
до-раскладывает всё (идемпотентно, самолечение; им же делается первичная раскладка).

Дедуп повторной загрузки строится на содержимом, а не на расположении: у каждого
принятого фото навсегда есть ``content_hash`` в БД. ``write_manifest`` выгружает
их в файл, по которому будущий загрузчик пропускает уже принятые исходники.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from .config import Config
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
)

log = logging.getLogger(__name__)

# Имя манифеста известных хешей в data_dir (интерфейс для будущего загрузчика).
MANIFEST_NAME = "ingested.sha256"

# Единый реестр «статус → папка его кучи» (одна физическая папка на кучу).
# Значение — имя атрибута пути в ``Config``. ``approved``/``described`` — одна
# куча «Одобрено», поэтому делят папку. Новая куча в будущем добавляется сюда.
_STATUS_DIR: dict[str, str] = {
    STATUS_NEW: "inbox_dir",
    STATUS_STOCK_CANDIDATE: "stock_dir",
    STATUS_NON_STOCK: "non_stock_dir",
    STATUS_PREFILTERED: "prefiltered_dir",
    STATUS_APPROVED: "approved_dir",
    STATUS_DESCRIBED: "approved_dir",
    STATUS_REJECTED: "rejected_dir",
    STATUS_UPLOADED: "done_dir",
}


def folder_for_status(cfg: Config, status: str) -> Path:
    """Папка, в которой должен лежать файл снимка с данным статусом.

    Неизвестный статус попадает в inbox (безопасный дефолт «на разбор»).
    """
    return getattr(cfg, _STATUS_DIR.get(status, "inbox_dir"))


def _is_placed(src: Path, target: Path, is_inbox: bool) -> bool:
    """Уже ли файл в целевой папке. Для inbox — где угодно внутри (intake сканит
    рекурсивно); для stock/approved — непосредственно в папке."""
    if is_inbox:
        return src.parent == target or target in src.parents
    return src.parent == target


def _unique_dest(target: Path, name: str, asset_id: int) -> Path:
    """Путь назначения без затирания: при совпадении имени добавляем id снимка."""
    dest = target / name
    if not dest.exists():
        return dest
    stem, suffix = Path(name).stem, Path(name).suffix
    return target / f"{stem}_{asset_id}{suffix}"


def relocate(conn, cfg: Config, asset_id: int, status: str) -> bool:
    """Перемещает файл снимка в папку его статуса и обновляет ``original_path``.

    Возвращает ``True``, если файл уже на месте или успешно перемещён; ``False``,
    если переместить не удалось (файл занят/пропал) — тогда ``run_organize``
    подхватит позже. Никогда не роняет вызывающий код (перемещение — не критично).
    """
    row = conn.execute(
        "SELECT original_path FROM assets WHERE id = ?", (asset_id,)
    ).fetchone()
    if row is None or not row["original_path"]:
        return False
    src = Path(row["original_path"])
    target = folder_for_status(cfg, status)
    is_inbox = target == cfg.inbox_dir

    if _is_placed(src, target, is_inbox):
        return True
    if not src.exists():
        log.warning("Снимок %d: файл не найден для перемещения (%s)", asset_id, src)
        return False

    dest = _unique_dest(target, src.name, asset_id)
    try:
        target.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
    except OSError as exc:
        # Чаще всего — файл открыт в редакторе. Не мешаем смене статуса.
        log.warning(
            "Снимок %d: не удалось переместить в %s (занят?): %s",
            asset_id,
            target.name,
            exc,
        )
        return False
    conn.execute(
        "UPDATE assets SET original_path = ? WHERE id = ?", (str(dest), asset_id)
    )
    log.info("Снимок %d → %s/%s", asset_id, target.name, dest.name)
    return True


def run_organize(cfg: Config) -> dict[str, int]:
    """Раскладывает файлы ВСЕХ снимков по папкам их статуса (идемпотентно).

    Первичная раскладка и самолечение после неудавшихся на лету перемещений.
    В конце обновляет манифест хешей.
    """
    cfg.ensure_dirs()
    stats = {"total": 0, "moved": 0, "in_place": 0, "failed": 0}
    conn = get_connection(cfg.db_path)
    try:
        rows = conn.execute("SELECT id, status, original_path FROM assets").fetchall()
        stats["total"] = len(rows)
        for row in rows:
            src = Path(row["original_path"]) if row["original_path"] else None
            target = folder_for_status(cfg, row["status"])
            is_inbox = target == cfg.inbox_dir
            if src and _is_placed(src, target, is_inbox):
                stats["in_place"] += 1
                continue
            if relocate(conn, cfg, row["id"], row["status"]):
                conn.commit()
                stats["moved"] += 1
            else:
                stats["failed"] += 1
    finally:
        conn.close()

    write_manifest(cfg)
    log.info(
        "Раскладка завершена: всего %(total)d, перемещено %(moved)d, "
        "на месте %(in_place)d, не удалось %(failed)d",
        stats,
    )
    return stats


def write_manifest(cfg: Config) -> Path:
    """Пишет файл со всеми известными хешами (по одному в строке).

    Будущий загрузчик перед копированием исходника считает его SHA-256 и
    пропускает, если хеш здесь есть, — так принятые фото не грузятся повторно,
    где бы их файлы сейчас ни лежали.
    """
    cfg.ensure_dirs()
    path = cfg.data_dir / MANIFEST_NAME
    conn = get_connection(cfg.db_path)
    try:
        # Вся история хешей, а не только текущие: так вернувшийся оригинал
        # (версия до правки на месте) тоже опознаётся как уже принятый.
        hashes = [r[0] for r in conn.execute("SELECT content_hash FROM asset_hashes")]
    finally:
        conn.close()
    path.write_text("\n".join(hashes) + ("\n" if hashes else ""), encoding="ascii")
    log.info("Манифест известных хешей обновлён: %s (%d)", path, len(hashes))
    return path
