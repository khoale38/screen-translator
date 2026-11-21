"""A record of what the app read, translated and how long each step took, to paste into a bug report."""

from __future__ import annotations

import logging
import platform
import sys
from collections import deque
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable

from .config import Settings, data_dir

log = logging.getLogger("screen_translator")
log.setLevel(logging.INFO)
log.propagate = False
_recent: deque[str] = deque(maxlen=400)
_FORMAT = logging.Formatter("%(asctime)s.%(msecs)03d %(message)s", "%H:%M:%S")


class _Recent(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        _recent.append(self.format(record))


_memory = _Recent()
_memory.setFormatter(_FORMAT)
log.addHandler(_memory)


def log_path() -> Path:
    return data_dir() / "screen-translator.log"


def start() -> None:
    """Also keep the log on disk, so it survives a crash."""
    if any(isinstance(h, RotatingFileHandler) for h in log.handlers):
        return
    try:
        handler = RotatingFileHandler(log_path(), maxBytes=1_000_000, backupCount=1, encoding="utf-8")
    except OSError:
        return
    handler.setFormatter(_FORMAT)
    log.addHandler(handler)
    log.info("Started %s", _version())


def _version() -> str:
    from .updates import installed_version

    return f"Screen Translator {installed_version()}"


def _short(text: str, limit: int = 80) -> str:
    text = " / ".join(str(text).split("\n"))
    return repr(text if len(text) <= limit else text[: limit - 1] + "…")


def describe(event: dict) -> str | None:
    """One line for an event worth keeping; None for in-progress updates."""
    kind = event.get("type")
    if kind == "line" and event.get("final"):
        first = f", first word {event['first_ms']} ms" if event.get("first_ms") is not None else ""
        return (f"line {event.get('area')}: {event.get('ms')} ms{first}, {event.get('context_lines')} lines of context: "
                f"{_short(event.get('source', ''))} -> {_short(event.get('translation', ''))}")
    if kind == "ocr":
        return f"read area {event.get('area')} in {event.get('ms')} ms: {_short(event.get('text', ''))}"
    if kind == "screen":
        parts = event.get("ocr") or {}
        split = f" (find {parts['find']} ms, read {parts['read']} ms)" if "find" in parts and "read" in parts else ""
        return f"screen read in {event.get('ms')} ms{split}: {len(event.get('blocks', []))} pieces of text"
    if kind == "screen_text" and event.get("final") and "ms" in event:
        steps = event.get("steps") or {}
        detail = (f" (list {steps['list']} ms, {steps['steps']} steps of {steps['step']} ms)"
                  if steps.get("steps") else "")
        return (f"screen piece {event.get('ms')} ms of a batch of {event.get('batch', 1)}{detail}: "
                f"{_short(event.get('source', ''))} -> {_short(event.get('translation', ''))}")
    if kind == "heard" and event.get("final"):
        return f"heard in {event.get('ms')} ms: {_short(event.get('text', ''))}"
    if kind == "devices":
        return f"model on {event.get('model') or 'unknown'}, text reading on {event.get('ocr') or 'unknown'}"
    if kind == "status":
        return f"status {event.get('state')}" + (f": {event['error']}" if event.get("error") else "")
    if kind == "dub":
        return f"dub error: {event['error']}" if event.get("error") else f"dub voice {event.get('voice')} in {event.get('ms')} ms"
    if kind == "download" and event.get("error"):
        return f"download error: {event['error']}"
    return None


def recording(emit: Callable[[dict], None]) -> Callable[[dict], None]:
    def emit_and_log(event: dict) -> None:
        try:
            line = describe(event)
            if line:
                log.info(line)
        except Exception:
            pass
        emit(event)

    return emit_and_log


def settings_line(settings: Settings) -> str:
    areas = len(settings.capture_rects) or (1 if settings.capture_rect or settings.region else 0)
    return (f"{settings.scope} ({areas} areas), {settings.source_lang} -> {settings.target_lang}, model "
            f"{settings.model}{' ' + settings.custom_model_path if settings.custom_model_path else ''}, "
            f"{settings.context_lines} lines of context, read every {settings.capture_interval_s} s "
            f"(screen {settings.screen_interval_s} s), {settings.stable_ticks} stable reads, skip text under "
            f"{settings.min_text_px} px, listen {settings.listen}, dub {settings.dub}, gpu_layers {settings.gpu_layers}, "
            f"watching {_short(settings.notes)}")


def report(settings: Settings, devices: dict[str, str]) -> str:
    lines = [
        f"{_version()} on {platform.platform()}, Python {sys.version.split()[0]}",
        f"Model on {devices.get('model') or 'not loaded yet'}; text reading on {devices.get('ocr') or 'not started yet'}",
        f"Settings: {settings_line(settings)}",
        "",
        *_recent,
    ]
    return "\n".join(lines)
