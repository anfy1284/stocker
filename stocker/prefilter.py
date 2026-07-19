"""Локальный предотбор «явный не-сток / кандидат» (до платного API).

Архив семейных снимков огромный (терабайты), стоковых кадров в нём мало, а
платить API за разбор каждого — дорого. Поэтому перед классификатором (Haiku,
:mod:`stocker.classifier`) весь массив прогоняется дешёвой локальной моделью на
видеокарте: она бесплатно отсеивает очевидный не-сток, и в API уходят только
правдоподобные кандидаты.

Сигнал складывается из трёх дешёвых источников по одному эмбеддингу CLIP
(``open_clip`` ViT-L/14, веса OpenAI):

  * **эстетика** — LAION-предиктор (MLP-голова поверх эмбеддинга) даёт оценку
    1–10; бытовой снимок ~4–5, чистый «стоковый» кадр ~5.5+. Это главный
    непрерывный сигнал «стоковости» и единственный порог, которым режем.
  * **семантика (zero-shot)** — тот же эмбеддинг сравнивается с набором текстовых
    описаний; если кадр опознан как скриншот/документ/чек/мем (не фотография) —
    жёсткий отсев, от порога не зависит.
  * **технические гейты** — разрешение (из EXIF), смаз (дисперсия Лапласиана) и
    экспозиция: только заведомый брак, тоже жёсткий отсев.

Жёсткие отсевы (не-фото, брак) необратимы порогом — это заведомый мусор.
Вся «агрессивность» отбора живёт в пороге эстетики: он хранится в БД (колонка
``prefilter_aesthetic``), поэтому :func:`rethreshold` перекладывает кучу под
новый порог мгновенно, без повторного прогона сети. Ослабили порог — часть
кадров из «Предотсева» вернулась в ``new`` на разбор API.

Тяжёлые зависимости (torch/open_clip/cv2) импортируются лениво внутри функций,
чтобы остальной код (веб, приём) не платил за их загрузку и работал без них.
"""

from __future__ import annotations

import logging
import urllib.request
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path

from .config import Config
from .db import STATUS_NEW, STATUS_PREFILTERED, get_connection
from . import organize

log = logging.getLogger(__name__)

# --- Модель ---------------------------------------------------------------
# ViT-L/14 (веса OpenAI) — под неё обучен LAION-предиктор эстетики (768-мерный
# эмбеддинг). На GTX 1050 Ti (4 ГБ) идёт в fp32 (Pascal невыгоден в fp16).
CLIP_MODEL = "ViT-L-14"
CLIP_PRETRAINED = "openai"
EMBED_DIM = 768

# LAION improved-aesthetic-predictor — MLP-голова поверх нормированного эмбеддинга.
_AESTHETIC_URL = (
    "https://github.com/christophschuhmann/improved-aesthetic-predictor/"
    "raw/main/sac+logos+ava1-l14-linearMSE.pth"
)
_AESTHETIC_FILE = "sac+logos+ava1-l14-linearMSE.pth"

# --- Технические гейты (заведомый брак; жёсткий отсев, порогом не отменяется) ---
MIN_MEGAPIXELS = 4_000_000  # минимум Shutterstock — 4 Мп
_GATE_LONG_SIDE = 1024      # к этой стороне приводим кадр перед оценкой резкости
BLUR_MIN_VAR = 40.0         # дисперсия Лапласиана ниже — явный смаз/не в фокусе
DARK_MEAN = 26              # средняя яркость (0–255) ниже — провал в тень
BRIGHT_MEAN = 232           # выше — выбитый пересвет
DARK_FRAC = 0.65            # доля почти чёрных пикселей выше — тень
BRIGHT_FRAC = 0.55          # доля почти белых пикселей выше — пересвет

# Батч эмбеддингов CLIP. 16 кадров 224px в fp32 влезают в 4 ГБ с запасом.
DEFAULT_BATCH = 16

# Zero-shot метки: (описание, это_мусор, причина). «Мусор» — не-фотография
# (жёсткий отсев). Нейтральные «фото» держат вероятность, чтобы обычный снимок
# не попал в мусор по argmax; агрессивную отсечку по ним делает порог эстетики.
_LABELS: list[tuple[str, bool, str]] = [
    ("a screenshot of a phone or computer screen", True, "скриншот"),
    ("a screenshot of a chat or social media app", True, "скриншот"),
    ("a scan of a document or a page of text", True, "документ/текст"),
    ("a photo of a receipt or an invoice", True, "документ/чек"),
    ("a meme image with overlaid text", True, "мем/текст"),
    ("a professional stock photograph", False, ""),
    ("a high quality photograph of people", False, ""),
    ("a high quality photograph of nature or a landscape", False, ""),
    ("a high quality photograph of an object or food", False, ""),
    ("a casual family snapshot", False, ""),
]
# Порог вероятности, при котором «мусорный» argmax считаем жёстким отсевом.
_JUNK_PROB = 0.55


def _aesthetic_weights(cfg: Config) -> Path:
    """Путь к весам предиктора эстетики; при отсутствии — скачивает в ``models_dir``.

    Переопределяется ``STOCKER_AESTHETIC_WEIGHTS`` (локальный файл, если машина
    без интернета). Скачивание разовое — потом веса лежат рядом с БД.
    """
    import os

    override = os.environ.get("STOCKER_AESTHETIC_WEIGHTS")
    if override:
        return Path(override).expanduser().resolve()
    dest = cfg.models_dir / _AESTHETIC_FILE
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        log.info("Скачиваю веса эстетики → %s", dest)
        urllib.request.urlretrieve(_AESTHETIC_URL, dest)
        log.info("Веса эстетики загружены (%d КБ).", dest.stat().st_size // 1024)
    return dest


def _build_aesthetic_mlp(weights_path: Path):
    """MLP-голова LAION (768→1) с загруженными весами, в режиме eval."""
    import torch
    from torch import nn

    mlp = nn.Sequential(
        nn.Linear(EMBED_DIM, 1024),
        nn.Dropout(0.2),
        nn.Linear(1024, 128),
        nn.Dropout(0.2),
        nn.Linear(128, 64),
        nn.Dropout(0.1),
        nn.Linear(64, 16),
        nn.Linear(16, 1),
    )
    # В опубликованных весах ключи вида ``layers.0.weight`` — снимаем префикс.
    state = torch.load(weights_path, map_location="cpu")
    state = {k.replace("layers.", ""): v for k, v in state.items()}
    mlp.load_state_dict(state)
    mlp.eval()
    return mlp


class PrefilterModel:
    """Загруженная локальная модель: CLIP + голова эстетики + текстовые эмбеддинги.

    Тяжёлое (веса) грузится один раз в конструкторе; :meth:`score_images`
    прогоняет батч кадров и возвращает по одному вердикту на кадр.
    """

    def __init__(self, cfg: Config, device: str | None = None) -> None:
        import open_clip
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Предотбор: устройство %s, модель %s/%s",
                 self.device, CLIP_MODEL, CLIP_PRETRAINED)

        model, _, preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_PRETRAINED
        )
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess
        self.aesthetic = _build_aesthetic_mlp(_aesthetic_weights(cfg)).to(self.device)

        tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
        self._junk = [is_junk for _, is_junk, _ in _LABELS]
        self._reasons = [reason for _, _, reason in _LABELS]
        with torch.no_grad():
            tokens = tokenizer([text for text, _, _ in _LABELS]).to(self.device)
            tfeat = self.model.encode_text(tokens)
            self.text_features = tfeat / tfeat.norm(dim=-1, keepdim=True)
        scale = getattr(self.model, "logit_scale", None)
        self.logit_scale = float(scale.exp()) if scale is not None else 100.0

    def score_images(
        self, images: list, dims: list[tuple[int | None, int | None]]
    ) -> list[dict]:
        """Оценивает батч PIL-кадров. ``dims`` — (width, height) из EXIF на кадр.

        Возвращает по словарю на кадр: ``aesthetic`` (float), ``hard_reject``
        (bool), ``reason`` (str — почему отсеян жёстко, иначе "").
        """
        torch = self.torch
        batch = torch.stack([self.preprocess(im) for im in images]).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            aesthetic = self.aesthetic(feats).squeeze(-1).cpu().numpy()
            probs = (self.logit_scale * feats @ self.text_features.T).softmax(dim=-1)
            probs = probs.cpu().numpy()

        out: list[dict] = []
        for i, im in enumerate(images):
            reason = self._hard_reject(im, dims[i], probs[i])
            out.append(
                {
                    "aesthetic": float(aesthetic[i]),
                    "hard_reject": bool(reason),
                    "reason": reason,
                }
            )
        return out

    def _hard_reject(self, im, dim, prob_row) -> str:
        """Причина жёсткого отсева (тех.брак / не-фото) или "" если кадр — фото."""
        import numpy as np

        width, height = dim
        if width and height and width * height < MIN_MEGAPIXELS:
            mp = width * height / 1_000_000
            return f"низкое разрешение ({mp:.1f} Мп)"

        best = int(prob_row.argmax())
        if self._junk[best] and prob_row[best] >= _JUNK_PROB:
            return self._reasons[best]

        # Резкость и экспозиция — по приведённому к 1024px полутону.
        gray = im.convert("L")
        long_side = max(gray.size)
        if long_side > _GATE_LONG_SIDE:
            k = _GATE_LONG_SIDE / long_side
            gray = gray.resize((max(1, int(gray.size[0] * k)),
                                max(1, int(gray.size[1] * k))))
        arr = np.asarray(gray, dtype=np.float64)
        if _laplacian_var(arr) < BLUR_MIN_VAR:
            return "смаз / не в фокусе"
        mean = float(arr.mean())
        if mean < DARK_MEAN or (arr < 16).mean() > DARK_FRAC:
            return "провал в тень"
        if mean > BRIGHT_MEAN or (arr > 240).mean() > BRIGHT_FRAC:
            return "пересвет"
        return ""


def _laplacian_var(arr) -> float:
    """Дисперсия Лапласиана (мера резкости) без OpenCV — свёрткой numpy."""
    import numpy as np

    # 4-связный лапласиан: сумма соседей минус 4·центр.
    lap = (
        -4.0 * arr
        + np.roll(arr, 1, 0)
        + np.roll(arr, -1, 0)
        + np.roll(arr, 1, 1)
        + np.roll(arr, -1, 1)
    )
    # Края портит np.roll (заворот) — считаем дисперсию по внутренней области.
    inner = lap[1:-1, 1:-1] if lap.shape[0] > 2 and lap.shape[1] > 2 else lap
    return float(inner.var())


def _batched(rows: list, size: int) -> Iterator[list]:
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def _apply_threshold(hard_reject: bool, aesthetic: float, threshold: float) -> bool:
    """Проходит ли кадр предотбор: не жёсткий отсев и эстетика не ниже порога."""
    return (not hard_reject) and aesthetic >= threshold


def run_prefilter(
    cfg: Config,
    threshold: float | None = None,
    on_progress: Callable[[dict], None] | None = None,
    batch_size: int = DEFAULT_BATCH,
    device: str | None = None,
) -> dict[str, int]:
    """Прогоняет ещё не оценённые снимки ``new`` через локальную модель.

    Каждому проставляет ``prefilter_aesthetic`` / ``prefilter_hard_reject`` /
    ``prefilter_reason`` / ``prefiltered_at``. Не прошедшие порог уезжают в кучу
    ``prefiltered`` (и физически — в ``prefiltered_dir``); прошедшие остаются
    ``new`` и позже попадут в платный классификатор. Идемпотентно: берём только
    снимки с пустым ``prefiltered_at``.
    """
    threshold = cfg.prefilter_threshold if threshold is None else threshold
    stats = {"total": 0, "kept": 0, "filtered": 0, "errors": 0, "done": 0}

    def report() -> None:
        if on_progress:
            on_progress(dict(stats))

    conn = get_connection(cfg.db_path)
    try:
        rows = conn.execute(
            "SELECT id, preview_path, width, height FROM assets "
            "WHERE status = ? AND prefiltered_at IS NULL ORDER BY id",
            (STATUS_NEW,),
        ).fetchall()
        stats["total"] = len(rows)
        report()
        if not rows:
            log.info("Предотбор: нет неоценённых снимков.")
            return stats

        model = PrefilterModel(cfg, device=device)
        from PIL import Image

        for chunk in _batched(rows, batch_size):
            images, dims, valid = [], [], []
            for row in chunk:
                try:
                    im = Image.open(row["preview_path"]).convert("RGB")
                except Exception:
                    stats["errors"] += 1
                    stats["done"] += 1
                    log.exception("Предотбор: не открыть превью снимка %d", row["id"])
                    continue
                images.append(im)
                dims.append((row["width"], row["height"]))
                valid.append(row)
            if not images:
                report()
                continue
            try:
                verdicts = model.score_images(images, dims)
            except Exception:
                stats["errors"] += len(images)
                stats["done"] += len(images)
                log.exception("Предотбор: сбой инференса батча")
                report()
                continue

            for row, v in zip(valid, verdicts):
                keep = _apply_threshold(v["hard_reject"], v["aesthetic"], threshold)
                conn.execute(
                    "UPDATE assets SET prefilter_aesthetic = ?, "
                    "prefilter_hard_reject = ?, prefilter_reason = ?, "
                    "prefiltered_at = ? WHERE id = ?",
                    (
                        v["aesthetic"],
                        int(v["hard_reject"]),
                        v["reason"] or None,
                        datetime.now().isoformat(),
                        row["id"],
                    ),
                )
                if keep:
                    stats["kept"] += 1
                else:
                    organize.relocate(conn, cfg, row["id"], STATUS_PREFILTERED)
                    conn.execute(
                        "UPDATE assets SET status = ? WHERE id = ?",
                        (STATUS_PREFILTERED, row["id"]),
                    )
                    stats["filtered"] += 1
                stats["done"] += 1
            conn.commit()
            report()
    finally:
        conn.close()

    log.info(
        "Предотбор завершён (порог %.2f): всего %d, оставлено %d, отсеяно %d, "
        "ошибок %d",
        threshold, stats["total"], stats["kept"], stats["filtered"], stats["errors"],
    )
    return stats


def rethreshold(cfg: Config, threshold: float | None = None) -> dict[str, int]:
    """Пересобирает кучи ``new``/``prefiltered`` под новый порог — без сети.

    Двигает снимки только между двумя «до-API» кучами, опираясь на уже
    сохранённые скоры: снимки с решением API (сток-кандидат/не-сток и далее)
    не трогает. Жёсткие отсевы (``prefilter_hard_reject``) остаются в
    «Предотсеве» при любом пороге — это заведомый мусор.
    """
    threshold = cfg.prefilter_threshold if threshold is None else threshold
    stats = {"to_new": 0, "to_filtered": 0}
    conn = get_connection(cfg.db_path)
    try:
        rows = conn.execute(
            "SELECT id, status, prefilter_aesthetic, prefilter_hard_reject "
            "FROM assets WHERE prefiltered_at IS NOT NULL "
            "AND status IN (?, ?)",
            (STATUS_NEW, STATUS_PREFILTERED),
        ).fetchall()
        for row in rows:
            keep = _apply_threshold(
                bool(row["prefilter_hard_reject"]),
                row["prefilter_aesthetic"] if row["prefilter_aesthetic"] is not None else -1.0,
                threshold,
            )
            target = STATUS_NEW if keep else STATUS_PREFILTERED
            if target == row["status"]:
                continue
            organize.relocate(conn, cfg, row["id"], target)
            conn.execute(
                "UPDATE assets SET status = ? WHERE id = ?", (target, row["id"])
            )
            stats["to_new" if keep else "to_filtered"] += 1
        conn.commit()
    finally:
        conn.close()
    log.info(
        "Порог предотбора %.2f: вернулось в разбор %d, ушло в предотсев %d",
        threshold, stats["to_new"], stats["to_filtered"],
    )
    return stats
