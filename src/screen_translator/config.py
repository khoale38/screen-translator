from __future__ import annotations

import json
import os
import shutil
import sys
import threading
from dataclasses import asdict, dataclass, field, fields
from functools import lru_cache
from pathlib import Path

from platformdirs import user_data_dir

APP_NAME = "ScreenTranslator"


def data_dir() -> Path:
    override = os.environ.get("SCREEN_TRANSLATOR_HOME")
    path = Path(override) if override else Path(user_data_dir(APP_NAME, appauthor=False))
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir() -> Path:
    if not os.environ.get("SCREEN_TRANSLATOR_HOME"):
        beside_app = installed_models_dir()
        if beside_app:
            return beside_app
    path = data_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def app_dir() -> Path | None:
    """The folder the Windows app was installed or unzipped to; None when running from source or on macOS."""
    if getattr(sys, "frozen", False) and sys.platform == "win32":
        return Path(sys.executable).resolve().parent
    return None


@lru_cache(maxsize=1)
def installed_models_dir() -> Path | None:
    """Models live in the folder picked at install, unless it is read-only (an all-users install in Program Files)."""
    folder = app_dir()
    if folder is None:
        return None
    path = folder / "models"
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write-test"
        probe.write_bytes(b"")
        probe.unlink()
    except OSError:
        return None
    move_models(data_dir() / "models", path)
    return path


def move_models(old: Path, new: Path) -> None:
    """Bring models downloaded by earlier versions along, so they are not downloaded again."""
    if not old.is_dir():
        return
    for model in old.glob("*.gguf*"):
        if not (new / model.name).exists():
            try:
                shutil.move(str(model), new / model.name)
            except OSError:
                continue
    try:
        old.rmdir()
    except OSError:
        pass


@dataclass
class Region:
    """Fractions (0..1) of the selected monitor, so it survives resolution and Retina scaling changes."""

    left: float
    top: float
    width: float
    height: float


@dataclass
class Settings:
    source_lang: str = "auto"
    target_lang: str = "en"
    monitor: int = 1
    region: Region | None = None
    capture_rect: dict | None = None
    area: list[int] | None = None
    areas: list[list[int]] = field(default_factory=list)
    capture_rects: list[dict] = field(default_factory=list)
    screen_rect: dict | None = None
    screen_height: int = 0
    min_text_px: int = 12
    overlay_mode: str = "cover"
    scope: str = "area"
    screen_name: str = ""
    screen_interval_s: float = 1.0
    overlay_opacity: float = 0.92
    overlay_box: bool = True
    text_size: str = "x1"
    capture_interval_s: float = 0.4
    stable_ticks: int = 2
    min_text_chars: int = 2
    context_lines: int = 8
    context_ttl_s: float = 300.0
    glossary: dict[str, str] = field(default_factory=dict)
    notes: str = ""
    model: str = "qwen3.5-4b"
    custom_model_path: str = ""
    gpu_layers: int = -1
    check_updates: bool = True
    listen: bool = False
    port: int = 8765

    @classmethod
    def from_dict(cls, raw: dict) -> "Settings":
        known = {f.name for f in fields(cls)}
        values = {k: v for k, v in raw.items() if k in known}
        if values.get("scope") == "speech":
            values["scope"], values["listen"] = "area", True
        region = values.get("region")
        if isinstance(region, dict):
            values["region"] = Region(**{k: float(region[k]) for k in ("left", "top", "width", "height")})
        return cls(**values)

    def to_dict(self) -> dict:
        return asdict(self)


class SettingsStore:
    def __init__(self, path: Path | None = None):
        self.path = path or data_dir() / "settings.json"
        self._lock = threading.Lock()
        self.settings = self._load()

    def _load(self) -> Settings:
        if self.path.exists():
            try:
                return Settings.from_dict(json.loads(self.path.read_text(encoding="utf-8")))
            except (ValueError, TypeError):
                pass
        return Settings()

    def update(self, patch: dict) -> Settings:
        with self._lock:
            merged = {**self.settings.to_dict(), **patch}
            self.settings = Settings.from_dict(merged)
            self.path.write_text(json.dumps(self.settings.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
            return self.settings
