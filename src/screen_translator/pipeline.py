from __future__ import annotations

import os
import queue
import threading
import time
import traceback
from typing import Callable

from .capture import ScreenCapture, changed, ensure_screen_permission, thumbnail
from .config import Settings, SettingsStore
from .context import ContextWindow
from .models import DEFAULT_MODEL, MODELS
from .ocr import OcrEngine, default_engine
from .text import Stabilizer, TextEvent, assemble, normalize, similar
from .translator import LlamaTranslator, Translator, build_messages, label_speaker

Emit = Callable[[dict], None]


def resolve_model(settings: Settings) -> str:
    if settings.custom_model_path:
        if not os.path.isfile(settings.custom_model_path):
            raise FileNotFoundError(f"Custom model file not found: {settings.custom_model_path}")
        return settings.custom_model_path
    spec = MODELS.get(settings.model) or MODELS[DEFAULT_MODEL]
    if not spec.downloaded:
        raise FileNotFoundError(f"Model '{spec.label}' is not downloaded yet.")
    return str(spec.path)


def default_translator(settings: Settings) -> Translator:
    return LlamaTranslator(resolve_model(settings), gpu_layers=settings.gpu_layers)


class Pipeline:
    def __init__(
        self,
        store: SettingsStore,
        emit: Emit,
        ocr_factory: Callable[[], OcrEngine] = default_engine,
        translator_factory: Callable[[Settings], Translator] = default_translator,
        capture_factory: Callable[[], ScreenCapture] = ScreenCapture,
        permission_check: Callable[[], None] = ensure_screen_permission,
    ) -> None:
        self.store = store
        self.emit = emit
        self.ocr_factory = ocr_factory
        self.translator_factory = translator_factory
        self.capture_factory = capture_factory
        self.permission_check = permission_check
        self.context = ContextWindow(store.settings.context_lines, store.settings.context_ttl_s)
        self._events: queue.Queue[TextEvent] = queue.Queue()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._translator: Translator | None = None
        self._translator_key: tuple | None = None
        self._line_id = 0
        self._reset_stabilizer = threading.Event()
        self._translator_ready = threading.Event()
        self.state = "stopped"
        self.last_error = ""

    @property
    def running(self) -> bool:
        return any(t.is_alive() for t in self._threads)

    def _set_state(self, state: str, error: str = "") -> None:
        self.state, self.last_error = state, error
        self.emit({"type": "status", "state": state, "error": error})

    def start(self) -> None:
        if self.running:
            return
        resolve_model(self.store.settings)
        self.permission_check()
        self._stop.clear()
        self._translator_ready.clear()
        self._events = queue.Queue()
        self._threads = [
            threading.Thread(target=self._guard, args=(self._capture_loop,), name="capture", daemon=True),
            threading.Thread(target=self._guard, args=(self._translate_loop,), name="translate", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=5)
        self._threads = []
        self._set_state("stopped")

    def shutdown(self) -> None:
        """Free the model before interpreter exit; llama.cpp's Metal backend aborts if it is still alive."""
        self.stop()
        close = getattr(self._translator, "close", None)
        if close:
            close()
        self._translator = None

    def clear_context(self) -> None:
        self.context.clear()
        self._reset_stabilizer.set()
        self.emit({"type": "clear"})

    def apply_settings(self, settings: Settings) -> None:
        self.context.configure(settings.context_lines, settings.context_ttl_s)
        self._reset_stabilizer.set()

    def _guard(self, loop: Callable[[], None]) -> None:
        try:
            loop()
        except Exception as exc:
            traceback.print_exc()
            self._stop.set()
            self._set_state("error", f"{type(exc).__name__}: {exc}")

    def _capture_loop(self) -> None:
        while not self._translator_ready.wait(0.2):
            if self._stop.is_set():
                return
        capture = self.capture_factory()
        ocr = self.ocr_factory()
        settings = self.store.settings
        stabilizer = Stabilizer(settings.stable_ticks, settings.min_text_chars)
        previous_thumb = None
        text = ""
        try:
            while not self._stop.is_set():
                started = time.monotonic()
                settings = self.store.settings
                if self._reset_stabilizer.is_set():
                    self._reset_stabilizer.clear()
                    stabilizer = Stabilizer(settings.stable_ticks, settings.min_text_chars)
                    previous_thumb = None

                frame = capture.grab(settings.monitor, settings.region)
                thumb = thumbnail(frame)
                if changed(previous_thumb, thumb):
                    text = assemble(ocr.recognize(frame, settings.source_lang))
                    self.emit({"type": "ocr", "text": text, "ms": round((time.monotonic() - started) * 1000)})
                previous_thumb = thumb

                event = stabilizer.feed(text)
                if event and not self._is_own_output(event):
                    self._events.put(event)

                self._stop.wait(max(0.0, settings.capture_interval_s - (time.monotonic() - started)))
        finally:
            capture.close()

    def _is_own_output(self, event: TextEvent) -> bool:
        """Ignore text that is our own translation, e.g. the floating subtitle window sitting inside the region."""
        if event.kind != "text":
            return False
        return any(similar(event.text, normalize(turn.translation), 0.85) for turn in self.context.turns()[-3:])

    def _load_translator(self) -> Translator:
        settings = self.store.settings
        key = (settings.model, settings.custom_model_path, settings.gpu_layers)
        if self._translator is None or self._translator_key != key:
            self._set_state("loading")
            close = getattr(self._translator, "close", None)
            if close:
                close()
            self._translator = None
            self._translator = self.translator_factory(settings)
            self._translator_key = key
        return self._translator

    def _translate_loop(self) -> None:
        translator = self._load_translator()
        self._translator_ready.set()
        self._set_state("running")
        while not self._stop.is_set():
            try:
                event = self._events.get(timeout=0.2)
            except queue.Empty:
                continue
            if event.kind == "clear":
                self.emit({"type": "clear"})
                continue
            self._translate(translator, event)

    def _translate(self, translator: Translator, event: TextEvent) -> None:
        settings = self.store.settings
        history = self.context.turns()
        replace = event.extends_previous and bool(history)
        if replace:
            history = history[:-1]
        else:
            self._line_id += 1
        line_id = self._line_id

        messages = build_messages(
            event.text, history, settings.source_lang, settings.target_lang,
            settings.glossary, settings.notes,
        )
        started = time.monotonic()

        def partial(text: str) -> None:
            self.emit({"type": "line", "id": line_id, "source": event.text, "translation": text, "final": False})

        partial("")
        translation = label_speaker(event.text, translator.translate(messages, partial))
        self.context.add(event.text, translation, replace_last=replace)
        self.emit({
            "type": "line", "id": line_id, "source": event.text, "translation": translation, "final": True,
            "ms": round((time.monotonic() - started) * 1000), "context_lines": len(history),
        })
