from __future__ import annotations

import os
import queue
import threading
import time
import traceback
from statistics import median
from typing import Callable

from .capture import ScreenCapture, block_colors, changed, ensure_screen_permission, thumbnail
from .config import Settings, SettingsStore
from .context import ContextWindow
from .languages import already_in
from .models import DEFAULT_MODEL, MODELS
from .ocr import OcrEngine, default_engine
from .text import Block, Stabilizer, TextEvent, assemble, group_blocks, normalize, similar
from .translator import (
    LlamaTranslator, Translator, build_batch_messages, build_messages, label_speaker, parse_numbered,
)

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
        self._last_area: int | None = None
        self._reset_stabilizer = threading.Event()
        self._translator_ready = threading.Event()
        self._screen_lock = threading.Lock()
        self._screen_blocks: list[Block] = []
        self._screen_changed = threading.Event()
        self._cache: dict[tuple[str, str, str], str] = {}
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
        if self.store.settings.scope == "screen":
            loops = (self._screen_capture_loop, self._screen_translate_loop)
        else:
            loops = (self._capture_loop, self._translate_loop)
        self._threads = [
            threading.Thread(target=self._guard, args=(loop,), name=name, daemon=True)
            for loop, name in zip(loops, ("capture", "translate"))
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
        self._cache.clear()
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

    def _wait_for_translator(self) -> bool:
        while not self._translator_ready.wait(0.2):
            if self._stop.is_set():
                return False
        return True

    def _capture_loop(self) -> None:
        if not self._wait_for_translator():
            return
        capture = self.capture_factory()
        ocr = self.ocr_factory()
        stabilizers: list[Stabilizer] = []
        thumbs: dict[int, object] = {}
        texts: dict[int, str] = {}
        heights: dict[int, float] = {}
        try:
            while not self._stop.is_set():
                started = time.monotonic()
                settings = self.store.settings
                rects = settings.capture_rects or [settings.capture_rect]
                if self._reset_stabilizer.is_set() or len(stabilizers) != len(rects):
                    self._reset_stabilizer.clear()
                    stabilizers = [Stabilizer(settings.stable_ticks, settings.min_text_chars) for _ in rects]
                    thumbs, texts, heights = {}, {}, {}

                for area, rect in enumerate(rects):
                    read_at = time.monotonic()
                    frame = capture.grab_rect(rect) if rect else capture.grab(settings.monitor, settings.region)
                    thumb = thumbnail(frame)
                    if changed(thumbs.get(area), thumb):
                        lines = ocr.recognize(frame, settings.source_lang)
                        texts[area] = assemble(lines)
                        heights[area] = median(line.box[3] for line in lines) if lines else 0.0
                        self.emit({"type": "ocr", "area": area, "text": texts[area],
                                   "ms": round((time.monotonic() - read_at) * 1000)})
                    thumbs[area] = thumb

                    event = stabilizers[area].feed(texts.get(area, ""))
                    if event and not self._is_own_output(event):
                        event.area, event.line_height = area, heights.get(area, 0.0)
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

    def _ready_translator(self) -> Translator:
        translator = self._load_translator()
        self._translator_ready.set()
        self._set_state("running")
        return translator

    def _translate_loop(self) -> None:
        translator = self._ready_translator()
        while not self._stop.is_set():
            try:
                event = self._events.get(timeout=0.2)
            except queue.Empty:
                continue
            if event.kind == "clear":
                self.emit({"type": "clear", "area": event.area})
                continue
            self._translate(translator, event)

    def _translate(self, translator: Translator, event: TextEvent) -> None:
        settings = self.store.settings
        if already_in(settings.target_lang, event.text):
            self.emit({"type": "clear", "area": event.area})
            return
        history = self.context.turns()
        replace = event.extends_previous and bool(history) and self._last_area == event.area
        self._last_area = event.area
        if replace:
            history = history[:-1]
        else:
            self._line_id += 1
        line_id = self._line_id

        known = None if replace else self._cache.get(self._cache_key(event.text))
        if known is not None:
            self.emit({
                "type": "line", "id": line_id, "area": event.area, "source": event.text, "translation": known,
                "final": True, "line_height": event.line_height, "ms": 0, "context_lines": len(history),
            })
            return

        messages = build_messages(
            event.text, history, settings.source_lang, settings.target_lang,
            settings.glossary, settings.notes,
        )
        started = time.monotonic()

        def partial(text: str) -> None:
            self.emit({"type": "line", "id": line_id, "area": event.area, "source": event.text,
                       "translation": text, "final": False, "line_height": event.line_height})

        partial("")
        translation = label_speaker(event.text, translator.translate(messages, partial))
        self._remember(event.text, translation)
        self.context.add(event.text, translation, replace_last=replace)
        self.emit({
            "type": "line", "id": line_id, "area": event.area, "source": event.text, "translation": translation,
            "final": True, "line_height": event.line_height,
            "ms": round((time.monotonic() - started) * 1000), "context_lines": len(history),
        })

    def _remember(self, text: str, translation: str) -> None:
        if len(self._cache) >= 2000:
            self._cache.pop(next(iter(self._cache)))
        self._cache[self._cache_key(text)] = translation

    def _cache_key(self, text: str) -> tuple[str, str, str]:
        settings = self.store.settings
        return text, settings.source_lang, settings.target_lang

    def _publish_screen(self, blocks: list[Block], ms: int) -> None:
        with self._screen_lock:
            self._screen_blocks = blocks
        self._screen_changed.set()
        self.emit({"type": "screen", "ms": ms, "blocks": [
            {"text": b.text, "box": b.box, "line_height": b.line_height, "bg": b.bg, "fg": b.fg,
             "translation": self._cache.get(self._cache_key(b.text))}
            for b in blocks
        ]})

    def _screen_capture_loop(self) -> None:
        if not self._wait_for_translator():
            return
        capture = self.capture_factory()
        ocr = self.ocr_factory()
        previous_thumb = None
        blocks: list[Block] = []
        candidate: frozenset = frozenset()
        seen = 0
        shown: list[Block] = []
        ms = 0
        try:
            while not self._stop.is_set():
                started = time.monotonic()
                settings = self.store.settings
                if self._reset_stabilizer.is_set():
                    self._reset_stabilizer.clear()
                    previous_thumb, candidate, seen, shown = None, frozenset(), 0, []
                    self._publish_screen([], 0)

                grab = lambda: capture.grab_rect(settings.screen_rect) if settings.screen_rect else capture.grab(settings.monitor, None)
                frame = grab()
                thumb = thumbnail(frame)
                settled = False
                if changed(previous_thumb, thumb):
                    screen_height = settings.screen_height or frame.height
                    blocks = [
                        b for b in group_blocks(ocr.recognize(frame, settings.source_lang), frame.width, frame.height)
                        if len(b.text) >= settings.min_text_chars
                        and b.line_height * screen_height >= settings.min_text_px
                        and not already_in(settings.target_lang, b.text)
                    ]
                    for b in blocks:
                        b.bg, b.fg = block_colors(frame, b.box)
                    ms = round((time.monotonic() - started) * 1000)
                    # A second look straight after OCR: if nothing moved, publish now instead of after the next tick.
                    after = thumbnail(grab())
                    settled = not changed(thumb, after)
                    thumb = after
                previous_thumb = thumb

                signature = _signature(blocks)
                seen = seen + 1 if signature == candidate else 1
                candidate = signature
                if signature != _signature(shown):
                    if seen >= settings.stable_ticks or settled:
                        shown = blocks
                        self._publish_screen(shown, ms)
                    else:
                        # Hide what moved or changed straight away; show the new layout once it holds still.
                        still = [b for b in shown if _key(b) in signature]
                        if len(still) != len(shown):
                            shown = still
                            self._publish_screen(shown, ms)

                self._stop.wait(max(0.0, settings.screen_interval_s - (time.monotonic() - started)))
        finally:
            capture.close()

    def _screen_translate_loop(self) -> None:
        translator = self._ready_translator()
        while not self._stop.is_set():
            self._screen_changed.clear()
            with self._screen_lock:
                blocks = list(self._screen_blocks)
            pending = list(dict.fromkeys(b.text for b in blocks if self._cache_key(b.text) not in self._cache))
            if not pending:
                self._screen_changed.wait(0.2)
                continue
            batch = _take_batch(pending)
            if len(batch) == 1:
                self._translate_block(translator, batch[0])
                continue
            for text in self._translate_batch(translator, batch):
                self._translate_block(translator, text)

    def _translate_batch(self, translator: Translator, texts: list[str]) -> list[str]:
        """Translate several blocks in one call; returns the ones the model skipped, to be done one by one."""
        settings = self.store.settings
        history = self.context.turns()[-4:]
        messages = build_batch_messages(texts, history, settings.source_lang, settings.target_lang,
                                        settings.glossary, settings.notes)
        started = time.monotonic()
        shown: dict[int, str] = {}

        def partial(raw: str) -> None:
            for number, line in parse_numbered(raw, len(texts), complete=False).items():
                if shown.get(number) != line:
                    shown[number] = line
                    self.emit({"type": "screen_text", "source": texts[number - 1], "translation": line,
                               "final": False})

        done = parse_numbered(translator.translate(messages, partial), len(texts))
        ms = round((time.monotonic() - started) * 1000)
        for number, line in done.items():
            source = texts[number - 1]
            translation = label_speaker(source, line)
            self._remember(source, translation)
            self.context.add(source, translation)
            self.emit({"type": "screen_text", "source": source, "translation": translation, "final": True,
                       "ms": ms, "context_lines": len(history), "batch": len(texts)})
        return [text for number, text in enumerate(texts, 1) if number not in done]

    def _translate_block(self, translator: Translator, text: str) -> None:
        settings = self.store.settings
        history = self.context.turns()
        messages = build_messages(text, history, settings.source_lang, settings.target_lang,
                                  settings.glossary, settings.notes)
        started = time.monotonic()

        def partial(translation: str) -> None:
            self.emit({"type": "screen_text", "source": text, "translation": translation, "final": False})

        translation = label_speaker(text, translator.translate(messages, partial))
        self._remember(text, translation)
        self.context.add(text, translation)
        self.emit({"type": "screen_text", "source": text, "translation": translation, "final": True,
                   "ms": round((time.monotonic() - started) * 1000), "context_lines": len(history)})


def _take_batch(texts: list[str], max_items: int = 12, max_chars: int = 1200) -> list[str]:
    batch, size = [], 0
    for text in texts:
        if batch and (len(batch) >= max_items or size + len(text) > max_chars):
            break
        batch.append(text)
        size += len(text)
    return batch


def _key(block: Block) -> tuple:
    return block.text, tuple(round(v * 200) for v in block.box)


def _signature(blocks: list[Block]) -> frozenset:
    return frozenset(_key(b) for b in blocks)
