"""Конфигурация Stocker.

Все настраиваемые пути и секреты читаются из переменных окружения (и/или
файла ``.env`` в корне проекта). Секреты в коде не хранятся: ключ Anthropic
берётся только из ``ANTHROPIC_API_KEY``.

Дефолты складывают все рабочие данные в ``<корень проекта>/data``:
входящие, превью, логи и файл БД. Любой путь можно переопределить
отдельной переменной ``STOCKER_*``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Корень проекта — родитель пакета (…/stocker/stocker/config.py → …/stocker).
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    """Собранная конфигурация приложения."""

    project_root: Path
    data_dir: Path
    inbox_dir: Path
    stock_dir: Path
    non_stock_dir: Path
    approved_dir: Path
    rejected_dir: Path
    done_dir: Path
    previews_dir: Path
    logs_dir: Path
    export_dir: Path
    db_path: Path
    anthropic_api_key: str | None
    shutterstock_user: str | None
    shutterstock_password: str | None
    log_level: str

    @property
    def has_api_key(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def has_ftps_creds(self) -> bool:
        """Заданы ли логин и пароль для FTPS-заливки на Shutterstock."""
        return bool(self.shutterstock_user and self.shutterstock_password)

    def ensure_dirs(self) -> None:
        """Создаёт все рабочие каталоги (идемпотентно)."""
        for path in (
            self.data_dir,
            self.inbox_dir,
            self.stock_dir,
            self.non_stock_dir,
            self.approved_dir,
            self.rejected_dir,
            self.done_dir,
            self.previews_dir,
            self.logs_dir,
            self.export_dir,
            self.db_path.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def as_display_dict(self) -> dict[str, str]:
        """Безопасное для показа представление (без утечки секретов)."""
        return {
            "project_root": str(self.project_root),
            "data_dir": str(self.data_dir),
            "inbox_dir": str(self.inbox_dir),
            "stock_dir": str(self.stock_dir),
            "non_stock_dir": str(self.non_stock_dir),
            "approved_dir": str(self.approved_dir),
            "rejected_dir": str(self.rejected_dir),
            "done_dir": str(self.done_dir),
            "previews_dir": str(self.previews_dir),
            "logs_dir": str(self.logs_dir),
            "export_dir": str(self.export_dir),
            "db_path": str(self.db_path),
            "log_level": self.log_level,
            "anthropic_api_key": "<задан>" if self.has_api_key else "<не задан>",
            "shutterstock": "<задан>" if self.has_ftps_creds else "<не задан>",
        }


def _path_from_env(name: str, default: Path) -> Path:
    """Путь из переменной окружения ``name`` или ``default``."""
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


def load_config() -> Config:
    """Загружает ``.env`` (если есть) и собирает конфигурацию из окружения."""
    load_dotenv(PROJECT_ROOT / ".env")

    data_dir = _path_from_env("STOCKER_DATA_DIR", PROJECT_ROOT / "data")
    return Config(
        project_root=PROJECT_ROOT,
        data_dir=data_dir,
        inbox_dir=_path_from_env("STOCKER_INBOX_DIR", data_dir / "inbox"),
        stock_dir=_path_from_env("STOCKER_STOCK_DIR", data_dir / "stock"),
        non_stock_dir=_path_from_env("STOCKER_NON_STOCK_DIR", data_dir / "non_stock"),
        approved_dir=_path_from_env("STOCKER_APPROVED_DIR", data_dir / "approved"),
        rejected_dir=_path_from_env("STOCKER_REJECTED_DIR", data_dir / "rejected"),
        done_dir=_path_from_env("STOCKER_DONE_DIR", data_dir / "done"),
        previews_dir=_path_from_env("STOCKER_PREVIEWS_DIR", data_dir / "previews"),
        logs_dir=_path_from_env("STOCKER_LOGS_DIR", data_dir / "logs"),
        export_dir=_path_from_env("STOCKER_EXPORT_DIR", data_dir / "export"),
        db_path=_path_from_env("STOCKER_DB_PATH", data_dir / "stocker.db"),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        shutterstock_user=os.environ.get("STOCKER_SHUTTERSTOCK_USER"),
        shutterstock_password=os.environ.get("STOCKER_SHUTTERSTOCK_PASSWORD"),
        log_level=os.environ.get("STOCKER_LOG_LEVEL", "INFO").upper(),
    )
