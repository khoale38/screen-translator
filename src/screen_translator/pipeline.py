from __future__ import annotations

import os
import queue
import re
import threading
import time
import traceback
from collections import deque
from statistics import median
from typing import TYPE_CHECKING, Callable

import numpy as np

from . import diagnostics, online
from .dub import STALE_S, gender_of, speakable
from .capture import ScreenCapture, block_colors, changed, ensure_screen_permission, thumbnail
from .config import Settings, SettingsStore
from .address import Relationships, body_of, speaker_of
from .romanize import NOT_LATIN, han_viet, role_label, romanized
from .context import ContextWindow
from .languages import already_in
from .models import DEFAULT_MODEL, MODELS
from .ocr import OcrEngine, default_engine
from .text import Block, Stabilizer, TextEvent, assemble, char_size, group_blocks, normalize, similar
from .translator import (
    LlamaTranslator, Translator, build_batch_messages, build_messages, build_screen_messages, is_name_tab, label_speaker,
    model_label, unlabelled, without_echo,
    parse_numbered, strip_item_letter,
)

if TYPE_CHECKING:
    from .audio import AudioSource

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


def default_audio() -> "AudioSource":
    from .audio import AudioSource

    return AudioSource()


def default_recognizer():
    from .speech import default_recognizer

    return default_recognizer()


class Superseded(Exception):
    """Newer speech arrived, or the text left the screen, while it was still being translated."""


class Yielded(Exception):
    """Speech needed the model, so this text translation stepped aside and will be retried."""


class SharedTranslator:
    """One model for on-screen text and speech. Speech goes first: text waits, or stops mid-way and is redone."""

    def __init__(self, inner: Translator) -> None:
        self.inner = inner
        self._lock = threading.Lock()
        self._urgent = threading.Event()

    def translate(self, messages: list[dict], on_partial: Callable[[str], None] | None = None,
                  urgent: bool = False) -> str:
        if urgent:
            self._urgent.set()
        else:
            while self._urgent.is_set():
                time.sleep(0.01)
        with self._lock:
            if urgent:
                self._urgent.clear()

            def partial(text: str) -> None:
                if not urgent and self._urgent.is_set():
                    raise Yielded()
                if on_partial:
                    on_partial(text)

            return self.inner.translate(messages, partial)

    def warm_up(self, messages: list[dict]) -> None:
        """The first translation sets up the GPU work and saves the instructions; done early, it holds up no line."""
        if hasattr(self.inner, "warm_up"):
            with self._lock:
                self.inner.warm_up(messages)

    @property
    def parallel(self) -> bool:
        return hasattr(self.inner, "translate_each")

    @property
    def timing(self) -> dict[str, int]:
        return dict(getattr(self.inner, "timing", {}))

    def translate_each(self, conversations: list[list[dict]], lengths: list[int],
                       on_piece: Callable[[int, str, bool], None]) -> list[str]:
        while self._urgent.is_set():
            time.sleep(0.01)
        with self._lock:
            def piece(index: int, text: str, done: bool) -> None:
                if self._urgent.is_set():
                    raise Yielded()
                on_piece(index, text, done)

            return self.inner.translate_each(conversations, lengths, piece)


def default_dubber(pipeline: "Pipeline"):
    from .dub import Dubber

    return Dubber(lambda: pipeline.store.settings, emit=pipeline.emit)


def default_judge(settings: Settings | None = None):
    """The model that works out how characters are related; on the CPU so the game keeps the graphics card."""
    from .models import MODELS

    if settings is not None and online.is_online(settings):
        return online.ClaudeTranslator(online.api_key(settings))
    for key in ("qwen3.5-4b", "qwen3.5-2b"):
        if key in MODELS and MODELS[key].downloaded:
            return LlamaTranslator(str(MODELS[key].path), gpu_layers=0, n_ctx=2048)
    raise FileNotFoundError("No Qwen model is downloaded.")


def default_translator(settings: Settings) -> Translator:
    if online.is_online(settings):
        return online.ClaudeTranslator(online.api_key(settings))
    return LlamaTranslator(resolve_model(settings), gpu_layers=settings.gpu_layers)



_ROMAJI = re.compile(r"(?:(?:[kgsztdnhbpmrwy]|sh|ch|ts|j|f|ky|gy|ny|hy|my|ry|by|py|kk|tt|ss|pp)?[aiueo]|n)+"
                     r"(?: (?:(?:[kgsztdnhbpmrwy]|sh|ch|ts|j|f|ky|gy|ny|hy|my|ry|by|py|kk|tt|ss|pp)?[aiueo]|n)+)?")


_LOOP = re.compile(r"\b(\w+(?:\s+\w+){0,2})(?:[\s,]+\1\b){5,}")


def _collapsed(translation: str) -> str:
    """A small model stuck on one word ("ba ba ba …"), cut to one; five in a row may be meant ("No, no, no, no, no!")."""
    return _LOOP.sub(r"\1", translation)


def _without_hanzi(translation: str) -> str:
    """Hanzi still left after asking again: a name is read in Hán-Việt, anything else means nothing to a reader."""
    def read(run: str) -> str:
        name = han_viet(run) if 2 <= len(run) <= 3 and run[0] in _COMMON_SURNAMES else None
        return f" {name} " if name else " "

    text = re.sub(r"[\u3400-\u9fff]+", lambda m: read(m.group(0)), translation)
    return re.sub(r"\s{2,}", " ", re.sub(r"\s+([,.!?])", r"\1", text)).strip()


_COMMON_SURNAMES = set("王李张張刘劉陈陳杨楊黄黃赵趙吴吳周徐孙孫马馬朱胡郭何高林罗羅郑鄭梁谢謝宋唐许許韩韓冯馮邓鄧曹彭曾萧蕭田董袁潘蒋蔣"
                       "蔡余杜叶葉程苏蘇魏吕呂丁任沈姚卢盧姜崔钟鍾谭譚陆陸汪范金石廖贾賈夏韦韋傅方白邹鄒孟熊秦邱江尹薛段雷侯龙龍史陶黎"
                       "顾顧毛邵万萬钱錢严嚴武戴莫孔向汤湯慕上欧歐司令楚燕凌柳云雲叶")


def _body(translation: str) -> str:
    """A translation without its speaker label, to tell a copied line from a new one."""
    return re.sub(r"^[^:\n]{1,24}:\s*", "", translation.strip())


def _punctuated(text: str) -> str:
    """Recognized speech without spaces or commas, with a comma after a call it opens or ends with ("네, 마마.")."""
    titles = (r"할머니|할아버지|어머님|아버님|장모님|장인어른|엄마|아빠|오빠|언니|누나|형|선배님|선배|선생님|사장님|부장님|"
              r"팀장님|과장님")
    # A greeting, "don't worry", "time to eat" or a report after the title is said to that person, honorific or not.
    told = r"안녕|식사|걱정|보고서"
    # Not "할아버지어디가셨어" or "형지금없어요": an honorific verb or "isn't here" means the title is talked about.
    if not re.search(r"셨|시[어었고는지]|세요|계셔|계세|께서|주무|없어|안\s*계", text):
        told += r"|저|나|제가|내가|우리|빨리|왜|어디|지금|이거|잠깐|같이|오늘"
    text = re.sub(rf"^(\W*(?:{titles}))\s*(?=(?:{told})[가-힣])", r"\1, ", text)
    text = re.sub(r"^(\W*(?:마마|중전마마|자기야|여보))(?=[가-힣])(?!세요|게)", r"\1, ", text)
    text = re.sub(r"(?<=[가-힣])((?:마마|자기야|여보)\W*)$", r", \1", text)
    return re.sub(r"^(\W*[爹娘爸妈])(?=女儿|儿子|孩儿|我|您|你)", r"\1，", text)


def _sounds_alike(written: str, spelled: str) -> bool:
    """A model's Latin spelling of a katakana name that matches the rule spelling's first sound: Cecil for Seshiru."""
    if not re.fullmatch(r"[A-Za-z][A-Za-z' -]{1,19}", written):
        return False
    group = lambda ch: next((g for g in ("kcq", "sczj", "lr", "bv", "fh", "td", "aeiouy") if ch in g), ch)
    # Not a name cut short ("Misa" for ミサキ, spelled Misaki).
    short, full = written.replace(" ", "").lower(), spelled.replace(" ", "").lower()
    whole = not (full.startswith(short) and len(short) < len(full))
    return whole and bool(set(group(written[0].lower())) & set(group(spelled[0].lower())))

class Pipeline:
    def __init__(
        self,
        store: SettingsStore,
        emit: Emit,
        ocr_factory: Callable[[], OcrEngine] = default_engine,
        translator_factory: Callable[[Settings], Translator] = default_translator,
        capture_factory: Callable[[], ScreenCapture] = ScreenCapture,
        permission_check: Callable[[], None] = ensure_screen_permission,
        audio_factory: Callable[[], object] = default_audio,
        recognizer_factory: Callable[[], tuple] = default_recognizer,
        dubber_factory: Callable[["Pipeline"], object] = default_dubber,
        judge_factory: Callable[[Settings], object] = default_judge,
    ) -> None:
        self.store = store
        self.emit = diagnostics.recording(emit)
        self.ocr_factory = ocr_factory
        self.translator_factory = translator_factory
        self.capture_factory = capture_factory
        self.permission_check = permission_check
        self.audio_factory = audio_factory
        self.recognizer_factory = recognizer_factory
        self.dubber_factory = dubber_factory
        self._dubber = None
        self.context = ContextWindow(store.settings.context_lines, store.settings.context_ttl_s)
        self.relationships = Relationships(lambda: judge_factory(self.store.settings), lambda: self.store.settings,
                                           self.emit, judge_key=lambda: online.is_online(self.store.settings))
        self._events: queue.Queue[TextEvent] = queue.Queue()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._translator: Translator | None = None
        self._shared: SharedTranslator | None = None
        self._parallel_failed = False
        self._load_lock = threading.Lock()
        self._translator_key: tuple | None = None
        self._line_id = 0
        self._last_area: int | None = None
        self._reset_stabilizer = threading.Event()
        self._translator_ready = threading.Event()
        self._screen_lock = threading.Lock()
        self._screen_blocks: list[Block] = []
        self._screen_reading: list[Block] = []
        self._screen_changed = threading.Event()
        self._cache: dict[tuple[str, str, str], str] = {}
        self._previews: dict[int, tuple[str, str, int, int | None]] = {}
        self.devices: dict[str, str] = {}
        self._heard_lock = threading.Lock()
        self._heard: dict | None = None
        self._finals: deque[dict] = deque()
        self._names: dict[str, str] = {}
        self._kana_recent: deque[bool] = deque(maxlen=12)
        self._last_voice: tuple[int, str, float] | None = None
        self._previous_voice = ""
        self._previous_pitch = 0.0
        self._heard_changed = threading.Event()
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
        settings = self.store.settings
        if not online.is_online(settings):
            resolve_model(settings)
        elif not online.api_key(settings):
            raise online.OnlineError(online.MISSING_KEY)
        if settings.listen:
            from .models import SPEECH

            if not SPEECH.downloaded:
                raise FileNotFoundError("Speech recognition is not downloaded yet.")
        self.permission_check()
        diagnostics.log.info("Start: %s", diagnostics.settings_line(settings))
        self._stop.clear()
        self._translator_ready.clear()
        self._events = queue.Queue()
        self._heard = None
        self._finals = deque()
        if settings.scope == "screen":
            loops = [(self._screen_capture_loop, "capture"), (self._screen_translate_loop, "translate")]
        else:
            loops = [(self._capture_loop, "capture"), (self._translate_loop, "translate")]
        if settings.listen:
            loops += [(self._listen_loop, "listen"), (self._speech_translate_loop, "speech")]
        self._threads = [
            threading.Thread(target=self._guard, args=(loop,), name=name, daemon=True) for loop, name in loops
        ]
        for thread in self._threads:
            thread.start()

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=5)
        self._threads = []
        if self._dubber is not None:
            self._dubber.close()
            self._dubber = None
        self._set_state("stopped")

    def shutdown(self) -> None:
        """Free the model before interpreter exit; llama.cpp's Metal backend aborts if it is still alive."""
        self.stop()
        self.relationships.close()
        close = getattr(self._translator, "close", None)
        if close:
            close()
        self._translator = None

    def clear_context(self) -> None:
        self.context.clear()
        self.relationships.clear()
        self._last_voice, self._previous_voice, self._previous_pitch = None, "", 0.0
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
            diagnostics.log.exception("%s stopped", threading.current_thread().name)
            self._stop.set()
            self._set_state("error", str(exc) if isinstance(exc, online.OnlineError) else f"{type(exc).__name__}: {exc}")

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
                        self._report_devices(ocr)
                        texts[area] = assemble(lines, width=frame.width, height=frame.height, menus=True)
                        heights[area] = median(char_size(l, frame.width, frame.height) for l in lines) if lines else 0.0
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

    def _report_devices(self, ocr: OcrEngine | None) -> None:
        devices = {"model": getattr(self._translator, "device", ""), "ocr": getattr(ocr, "device", "")}
        if devices != self.devices:
            self.devices = devices
            print(f"Model runs on {devices['model'] or 'unknown'}; OCR runs on {devices['ocr'] or 'unknown'}", flush=True)
            self.emit({"type": "devices", **devices})

    def _is_own_output(self, event: TextEvent) -> bool:
        """Ignore text that is our own translation, e.g. the floating subtitle window sitting inside the region."""
        if event.kind not in ("text", "preview"):
            return False
        return any(similar(event.text, normalize(turn.translation), 0.85) for turn in self.context.turns()[-3:])

    def _load_translator(self) -> SharedTranslator:
        settings = self.store.settings
        key = (settings.model, settings.custom_model_path, settings.gpu_layers, online.api_key(settings))
        with self._load_lock:
            if self._translator is None or self._translator_key != key:
                self._set_state("loading")
                close = getattr(self._translator, "close", None)
                if close:
                    close()
                self._translator = None
                self._translator = self.translator_factory(settings)
                self._shared = SharedTranslator(self._translator)
                self._translator_key = key
            return self._shared

    def _ready_translator(self) -> SharedTranslator:
        translator = self._load_translator()
        self._translator_ready.set()
        self._set_state("running")
        return translator

    def _translate_loop(self) -> None:
        translator = self._ready_translator()
        settings = self.store.settings
        translator.warm_up(build_messages("OK", [], settings.source_lang, settings.target_lang,
                                          settings.glossary, settings.notes))
        while not self._stop.is_set():
            try:
                event = self._events.get(timeout=0.2)
            except queue.Empty:
                continue
            if event.kind == "clear":
                self._previews.pop(event.area, None)
                self.emit({"type": "clear", "area": event.area})
                continue
            if self._newer_in(event):
                continue
            try:
                if event.kind == "preview":
                    self._preview(translator, event)
                else:
                    self._translate(translator, event)
            except Superseded:
                continue
            except online.OnlineHiccup as exc:
                self._hiccup(exc)
            except Yielded:
                with self._events.mutex:
                    self._events.queue.appendleft(event)
                    self._events.not_empty.notify()

    def _newer_in(self, event: TextEvent) -> bool:
        """Whether the area's text has changed or gone since this event, which makes translating it pointless."""
        with self._events.mutex:
            return any(queued.area == event.area and (queued.kind == "clear" or not similar(queued.text, event.text, 0.85))
                       for queued in self._events.queue)

    def _extended(self, event: TextEvent, history: list) -> bool:
        return bool(history) and self._last_area == event.area and event.text.startswith(history[-1].source)

    def _preview(self, translator: Translator, event: TextEvent) -> None:
        """Translate text on its first read and show it; it becomes the line once a second read confirms it."""
        settings = self.store.settings
        if already_in(settings.target_lang, event.text) or self._cache.get(self._cache_key(event.text)) is not None:
            return
        history = self.context.turns()
        extends = self._extended(event, history)
        if extends:
            history = history[:-1]
        line_id = self._line_id if extends else self._line_id + 1
        messages = build_messages(event.text, history, settings.source_lang, settings.target_lang,
                                  settings.glossary, settings.notes, address=self.relationships.address(event.text))
        started = time.monotonic()
        first: list[int] = []

        def partial(text: str) -> None:
            if self._newer_in(event):
                raise Superseded()
            if text and not first:
                first.append(round((time.monotonic() - started) * 1000))
            self.emit({"type": "line", "id": line_id, "area": event.area, "source": event.text,
                       "translation": text, "final": False, "line_height": event.line_height})

        translation = self._checked(translator, event.text,
                                    self._labelled(event.text, translator.translate(messages, partial), translator), history)
        self._previews[event.area] = (event.text, translation, round((time.monotonic() - started) * 1000),
                                      first[0] if first else None)
        partial(translation)

    def _translate(self, translator: Translator, event: TextEvent) -> None:
        settings = self.store.settings
        if already_in(settings.target_lang, event.text):
            self.emit({"type": "clear", "area": event.area})
            return
        history = self.context.turns()
        # The line it extends may have been dropped half-way; then there is nothing to replace.
        replace = event.extends_previous and self._extended(event, history)
        self._last_area = event.area
        if replace:
            history = history[:-1]
        else:
            self._line_id += 1
        line_id = self._line_id
        address = self.relationships.address(event.text)
        note = address.note() if address and not address.guessed else ""

        preview = self._previews.pop(event.area, None)
        # A typewriter's half-shown line is a prefix of the full one and still scores as similar; it does not count.
        grew = len(_bare(event.text)) > len(_bare(preview[0])) and _bare(event.text).startswith(_bare(preview[0])) \
            if preview else False
        if preview and similar(preview[0], event.text, 0.92) and not grew:
            _source, translation, ms, first_ms = preview
            self._remember(event.text, translation)
            self.context.add(event.text, translation, replace_last=replace, note=note)
            self.relationships.heard(event.text)
            self.emit({
                "type": "line", "id": line_id, "area": event.area, "source": event.text, "translation": translation,
                "final": True, "line_height": event.line_height, "ms": ms, "context_lines": len(history),
                "first_ms": first_ms, "early": True,
            })
            return

        known = None if replace else self._cache.get(self._cache_key(event.text))
        if known is not None:
            self.relationships.heard(event.text)
            self.emit({
                "type": "line", "id": line_id, "area": event.area, "source": event.text, "translation": known,
                "final": True, "line_height": event.line_height, "ms": 0, "context_lines": len(history),
            })
            return

        messages = build_messages(
            event.text, history, settings.source_lang, settings.target_lang,
            settings.glossary, settings.notes, address=address,
        )
        started = time.monotonic()
        first: list[int] = []

        def partial(text: str) -> None:
            if self._newer_in(event):
                raise Superseded()
            if text and not first:
                first.append(round((time.monotonic() - started) * 1000))
            self.emit({"type": "line", "id": line_id, "area": event.area, "source": event.text,
                       "translation": text, "final": False, "line_height": event.line_height})

        partial("")
        translation = self._checked(translator, event.text,
                                    self._labelled(event.text, translator.translate(messages, partial), translator), history)
        self._remember(event.text, translation)
        self.context.add(event.text, translation, replace_last=replace, note=note)
        self.relationships.heard(event.text)
        self.emit({
            "type": "line", "id": line_id, "area": event.area, "source": event.text, "translation": translation,
            "final": True, "line_height": event.line_height,
            "ms": round((time.monotonic() - started) * 1000), "context_lines": len(history),
            "first_ms": first[0] if first else None,
        })

    def _remember(self, text: str, translation: str) -> None:
        if len(self._cache) >= 2000:
            self._cache.pop(next(iter(self._cache)))
        self._cache[self._cache_key(text)] = translation

    def _cache_key(self, text: str) -> tuple[str, str, str]:
        settings = self.store.settings
        return text, settings.source_lang, settings.target_lang

    def _read_screen(self, blocks: list[Block]) -> None:
        """Text just read, translated straight away so it is ready by the time it holds still and is shown."""
        with self._screen_lock:
            self._screen_reading = blocks
        self._screen_changed.set()

    def _on_screen(self) -> set[str]:
        with self._screen_lock:
            return {b.text for b in self._screen_blocks + self._screen_reading}

    def _publish_screen(self, blocks: list[Block], ms: int, ocr_timing: dict | None = None) -> None:
        with self._screen_lock:
            self._screen_blocks = blocks
        self._screen_changed.set()
        self.emit({"type": "screen", "ms": ms, "ocr": ocr_timing or {}, "blocks": [
            {"text": b.text, "box": b.box, "line_height": b.line_height, "bg": b.bg, "fg": b.fg, "vertical": b.vertical,
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
                    self._read_screen([])
                    self._publish_screen([], 0)

                grab = lambda: capture.grab_rect(settings.screen_rect) if settings.screen_rect else capture.grab(settings.monitor, None)
                frame = grab()
                thumb = thumbnail(frame)
                settled = False
                if changed(previous_thumb, thumb):
                    screen_height = settings.screen_height or frame.height
                    blocks = [
                        b for b in group_blocks(ocr.recognize(frame, settings.source_lang), frame.width, frame.height)
                        if (len(b.text) >= settings.min_text_chars or re.fullmatch("[\u3040-\u9fff\uac00-\ud7a3][!?！？。…]*", b.text))
                        and b.line_height * screen_height >= settings.min_text_px
                        and not already_in(settings.target_lang, b.text)
                        and not _cut_off(b, frame.height)
                    ]
                    self._report_devices(ocr)
                    for b in blocks:
                        b.bg, b.fg = block_colors(frame, b.box)
                    self._read_screen(blocks)
                    ms = round((time.monotonic() - started) * 1000)
                    # A second look straight after OCR: if nothing moved, publish now instead of after the next tick.
                    after = thumbnail(grab())
                    settled = not changed(thumb, after)
                    # Compared with the screen that was read, not the last tick, so slow changes such as a countdown
                    # add up; if it moved during the read, the next tick reads it again.
                    previous_thumb = after if settled else thumb

                # Stable means the same words: over a moving game scene the boxes never quite hold still.
                texts = _texts(blocks)
                seen = seen + 1 if texts == candidate else 1
                candidate = texts
                if _signature(blocks) != _signature(shown):
                    if texts == _texts(shown) or seen >= settings.stable_ticks or settled:
                        shown = blocks
                        self._publish_screen(shown, ms, getattr(ocr, "timing", None))
                    else:
                        # Hide what changed straight away; show the new words once they hold.
                        still = [b for b in shown if b.text in texts]
                        if len(still) != len(shown):
                            shown = still
                            self._publish_screen(shown, ms, getattr(ocr, "timing", None))

                self._stop.wait(max(0.0, settings.screen_interval_s - (time.monotonic() - started)))
        finally:
            capture.close()

    def _screen_translate_loop(self) -> None:
        translator = self._ready_translator()
        if translator.parallel:
            self._warm_up(translator)
        while not self._stop.is_set():
            self._screen_changed.clear()
            with self._screen_lock:
                blocks = self._screen_blocks + self._screen_reading
            pending = list(dict.fromkeys(b.text for b in blocks if self._cache_key(b.text) not in self._cache))
            if not pending:
                self._screen_changed.wait(0.2)
                continue
            batch = _take_batch(pending)
            try:
                # A lone leftover from a busy screen goes side by side too; alone it would take the slow path.
                if translator.parallel and not self._parallel_failed and (len(batch) > 1 or len(blocks) > 1):
                    try:
                        self._translate_each(translator, batch)
                        continue
                    except (RuntimeError, ValueError) as exc:
                        print(f"Translating pieces side by side failed ({exc}); translating them as a list", flush=True)
                        self._parallel_failed = True
                for text in batch if len(batch) == 1 else self._translate_batch(translator, batch):
                    self._translate_block(translator, text)
            except (Yielded, Superseded):
                continue
            except online.OnlineHiccup as exc:
                self._hiccup(exc)
                self._screen_changed.wait(2.0)

    def _hiccup(self, exc: Exception) -> None:
        diagnostics.log.warning("Skipped a line: %s", exc)
        self.emit({"type": "notice", "text": str(exc)})

    def _warm_up(self, translator: SharedTranslator) -> None:
        """Load the instructions and set up the GPU work once, so the first real screen is as quick as the rest."""
        settings = self.store.settings
        sample = ["OK", "Menu"]
        conversations = [build_screen_messages(i, sample, [], settings.source_lang, settings.target_lang,
                                               settings.glossary, settings.notes) for i in range(len(sample))]
        try:
            translator.translate_each(conversations, [len(text) for text in sample], lambda *_: None)
        except (RuntimeError, ValueError, Yielded) as exc:
            print(f"Warming up side-by-side translation failed: {exc}", flush=True)

    def _translate_each(self, translator: SharedTranslator, texts: list[str]) -> None:
        """Every piece as its own translation, all decoded side by side; each shows as soon as it is done."""
        settings = self.store.settings
        history = self.context.turns()[-4:]
        # Dialogue on a game screen ("Name\nline" from a name tab) gets the same who-talks-to-whom pronouns as subtitles.
        addresses = {text: self.relationships.address(text) for text in texts}
        notes = {text: address.note() if address else "" for text, address in addresses.items()}
        conversations = [
            build_screen_messages(index, texts, history, settings.source_lang, settings.target_lang, settings.glossary,
                                  settings.notes, note=notes[text], address=addresses[text])
            for index, text in enumerate(texts)
        ]
        started = time.monotonic()
        finished: list[tuple[str, str]] = []
        retry: list[str] = []
        left = set(texts)

        def piece(index: int, translation: str, done: bool) -> None:
            source = texts[index]
            if not done and self._screen_changed.is_set() and not left & self._on_screen():
                raise Superseded()
            translation = strip_item_letter(translation, index)
            if not done:
                self.emit({"type": "screen_text", "source": source, "translation": translation, "final": False})
                return
            if re.fullmatch(r"\s*[A-L]?[.):]?\s*", translation):
                # A bare letter is the model naming an item instead of translating it; translate that one alone.
                retry.append(source)
                left.discard(source)
                return
            if "\n" in source:
                translation = translation.replace(" / ", "\n")
            if self.store.settings.target_lang == "vi" and role_label(source.strip()):
                # A role tab alone on screen ("간호사", "執事") is read from the table, not guessed next to its neighbour.
                translation = role_label(source.strip())
            if translation.strip() == source.strip() and romanized(source):
                # A name HY-MT handed back in kana or hangul, which a Vietnamese viewer cannot read.
                translation = self._names.setdefault(source.strip(), romanized(source))
            if is_name_tab(source) or speaker_of(source) and source.lstrip()[:1] in "【〔":
                translation = self._labelled(source, translation, translator)
            # A line answered with an earlier line's translation was copied from the history.
            copied = any(translation == turn.translation for turn in history if turn.source != source)
            translation = self._keys_kept(source, self._capitalised(translation))
            row = re.match(r"\s*([^\W\d_][^:：\n]{0,15})[:：]\s*(\S.*)", translation)
            if self._leaks(translation) and row and not NOT_LATIN.search(row.group(1)) and re.match(
                    r"[^:：\n]{1,8}[:：]", source):
                # A status row ("Tên: 이도윤") whose value is a name: spell the value, keep the translated key.
                translation = f"{row.group(1)}: {self._readable(source, row.group(2))}"
            # HY-MT sometimes translates the screen instruction itself instead of the row.
            echoed = re.match(r"\s*(?:Bối cảnh|Background)\b", translation) or "giao diện của trò chơi" in translation
            if self._leaks(translation) or (copied and len(source) < 40) or echoed or _LOOP.search(translation):
                # A small model slipping into the source's script mid-sentence ("Sư huynh đã替你 chắn"): ask again alone.
                retry.append(source)
                left.discard(source)
                return
            self._remember(source, translation)
            left.discard(source)
            finished.append((source, translation))
            self.emit({"type": "screen_text", "source": source, "translation": translation, "final": True,
                       "ms": round((time.monotonic() - started) * 1000), "context_lines": len(history),
                       "batch": len(texts), "steps": translator.timing})

        translator.translate_each(conversations, [len(text) for text in texts], piece)
        # A short header ("報酬") answered with its neighbour's sentence: the longer source keeps it, the rest go again.
        by_answer: dict[str, list[str]] = {}
        for source, translation in finished:
            by_answer.setdefault(translation, []).append(source)
        for sources in by_answer.values():
            for source in sorted(sources, key=len)[:-1]:
                retry.append(source)
                finished[:] = [(s, t) for s, t in finished if s != source]
        done = dict(finished)
        for source in texts:
            if source in done:
                # A guess from one bubble or line guides its own translation, not the ones after it.
                kept = notes[source] if addresses[source] and not addresses[source].guessed else ""
                self.context.add(source, done[source], note=kept)
                if is_name_tab(source):
                    self.relationships.heard(source)
        for source in retry:
            self._translate_block(translator, source, retried=True)

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
            translation = label_speaker(source, line, self._names)
            self._remember(source, translation)
            self.context.add(source, translation)
            self.emit({"type": "screen_text", "source": source, "translation": translation, "final": True,
                       "ms": ms, "context_lines": len(history), "batch": len(texts)})
        return [text for number, text in enumerate(texts, 1) if number not in done]

    def _translate_block(self, translator: Translator, text: str, retried: bool = False) -> None:
        settings = self.store.settings
        # A screen item asked again alone, after leaking its script or copying a neighbour, gets no earlier lines to
        # copy from and no pronoun note, the usual cause of a half-Chinese answer.
        history = [] if retried else self.context.turns()
        address = None if retried else self.relationships.address(text)
        messages = build_messages(text, history, settings.source_lang, settings.target_lang,
                                  settings.glossary, settings.notes, address=address)
        started = time.monotonic()

        def partial(translation: str) -> None:
            # A typewriter's half-typed line that has already left the screen is not worth finishing.
            if self._screen_changed.is_set() and text not in self._on_screen():
                raise Superseded()
            self.emit({"type": "screen_text", "source": text, "translation": translation, "final": False})

        translation = translator.translate(messages, partial)
        if not retried or is_name_tab(text) or speaker_of(text) and text.lstrip()[:1] in "【〔":
            translation = self._labelled(text, translation, translator)
        translation = self._capitalised(self._readable(text, translation, names_only=retried and not is_name_tab(text)))
        if retried and self._leaks(translation):
            translation = _without_hanzi(translation)
        self._remember(text, translation)
        self.context.add(text, translation, note=address.note() if address and not address.guessed else "")
        self.relationships.heard(text)
        self.emit({"type": "screen_text", "source": text, "translation": translation, "final": True,
                   "ms": round((time.monotonic() - started) * 1000), "context_lines": len(history)})


    def _listen_loop(self) -> None:
        from .speech import RATE, Segmenter, pitch_hz

        if not self._wait_for_translator():
            return
        recognizer, vad = self.recognizer_factory()
        if self.store.settings.source_lang not in ("auto", "zh", "zh-Hant", "en", "ja", "ko", "yue"):
            # SenseVoice hears only these; anything else comes out as confident nonsense.
            self.emit({"type": "notice", "text": "Speech can only be heard in Chinese, Cantonese, English, Japanese and "
                                                 "Korean. For other languages, translate the video's subtitles instead."})
        source = self.audio_factory()
        self._report_devices(None)
        segmenter = Segmenter(vad)
        utterance = 1
        try:
            while not self._stop.is_set():
                block = source.read()
                if block is None:
                    continue
                if self._dubber is not None and self._dubber.speaking() and not getattr(
                        source, "excludes_own_sound", False):
                    block = np.zeros_like(block)
                heard = segmenter.feed(block)
                if heard is None:
                    continue
                pitch = pitch_hz(heard.samples)
                # Crowd noise passes the VAD but has no voice in it; the recognizer would hear "Yeah." in it.
                if heard.final and not pitch and len(heard.samples) < RATE:
                    continue
                started = time.monotonic()
                result = recognizer.recognize(heard.samples, self.store.settings.source_lang)
                text, spoken = normalize(result[0]), result[1]
                if heard.final and not any(ch.isalnum() for ch in text):
                    continue
                emotion = result[2] if len(result) > 2 else ""
                self.emit({"type": "heard", "id": utterance, "text": text, "final": heard.final, "lang": spoken,
                           "ms": round((time.monotonic() - started) * 1000)})
                entry = {"id": utterance, "text": text, "final": heard.final, "ending": heard.ending,
                         "lang": spoken, "emotion": emotion, "at": time.monotonic(),
                         "pitch": pitch, "seconds": len(heard.samples) / RATE, "paused": heard.cut == "pause"}
                with self._heard_lock:
                    # Every finished sentence is kept; of a sentence still being spoken only the newest guess is.
                    if heard.final:
                        self._finals.append(entry)
                        self._heard = None
                    else:
                        self._heard = entry
                self._heard_changed.set()
                if heard.final:
                    utterance += 1
        finally:
            source.close()

    def _take_heard(self) -> dict | None:
        with self._heard_lock:
            if self.store.settings.dub:
                # When translating falls behind the speaker, old sentences give way so the dub stays with the video.
                while len(self._finals) > 1 and time.monotonic() - self._finals[0]["at"] > STALE_S:
                    self._finals.popleft()
            if self._finals:
                heard = self._finals.popleft()
            else:
                heard, self._heard = self._heard, None
            if not self._finals and self._heard is None:
                self._heard_changed.clear()
            return heard

    def _speech_translate_loop(self) -> None:
        """Translates the newest speech: live captions while someone talks, then the finished sentence at once."""
        translator = self._ready_translator()
        done: tuple[int, str, str, str] | None = None
        while not self._stop.is_set():
            if not self._heard_changed.wait(0.2):
                continue
            heard = self._take_heard()
            if heard is None:
                continue
            settings = self.store.settings
            text = heard["text"]
            if not text:
                continue
            if _spoken_in(settings.target_lang, heard.get("lang", "")) or already_in(settings.target_lang, text):
                self._speech_line(heard, text, heard["final"], ms=0, context_lines=0)
                continue
            # The same words with new punctuation can change who is called ("お父さん行こうよ、…"): reuse only the same cue.
            cue = self.relationships.address(_punctuated(text), gender_of(heard.get("pitch", 0.0)))
            if done and done[:3] == (heard["id"], _bare(text), cue.note() if cue else ""):
                translation = self._unrepeated(translator, heard, done[3]) if heard["final"] else done[3]
            else:
                try:
                    translation = self._translate_speech(translator, heard)
                except Superseded:
                    continue
                except online.OnlineHiccup as exc:
                    self._hiccup(exc)
                    continue
                done = (heard["id"], _bare(text), cue.note() if cue else "", translation)
            if heard["final"]:
                self._remember(text, translation)
                address = self.relationships.address(text)
                self.context.add(text, translation, note=address.note() if address and not address.guessed else "")
                self._speech_line(heard, translation, True, ms=round((time.monotonic() - heard["at"]) * 1000),
                                  context_lines=len(self.context.turns()) - 1)
                if settings.dub:
                    self._dub(translation, heard)

    def _dub(self, translation: str, heard: dict) -> None:
        from .dub import Line

        if self._dubber is None:
            self._dubber = self.dubber_factory(self)
        if self._leaks(translation):
            # Kana and hangul read out as their sounds ("Tteokbokki"); hanzi would come out as noise or silence.
            translation = re.sub(r"[\u3040-\u30ff\uac00-\ud7a3]+", lambda m: romanized(m.group(0)) or "", translation)
            translation = re.sub(r"[\u3400-\u9fff]+", "", translation).strip()
        self._dubber.say(Line(translation, self.store.settings.target_lang, heard.get("emotion", ""),
                              heard.get("pitch", 0.0), heard.get("seconds", 0.0), heard.get("at", 0.0),
                              heard.get("paused", True)))

    def _speech_line(self, heard: dict, translation: str, final: bool, **extra) -> None:
        if final and self._leaks(translation):
            translation = re.sub(r"[\u3040-\u30ff\uac00-\ud7a3]+", lambda m: romanized(m.group(0)) or "", translation)
        self.emit({"type": "line", "id": f"speech-{heard['id']}", "area": "speech", "source": heard["text"],
                   "translation": translation, "final": final, **extra})

    def _labelled(self, source: str, translation: str, translator) -> str:
        """The translation with its speaker's name, written as the model writes it rather than in kana or hangul."""
        name = speaker_of(source)
        if name and name not in self._names and (not name.isascii() or self._fixed_name(name)):
            written = model_label(translation)
            fixed = self._fixed_name(name, source)
            # A spelling by rule beats the model's, which may translate a name's meaning ("최 대리" as "Đại lý Tối"),
            # except a katakana loanword name the model spells as its original ("セシル" is Cecil, not Seshiru).
            # Never a model call here: it may run inside a side-by-side translation, which would deadlock.
            if fixed and written and re.fullmatch("[ァ-ヴー・]+", name) and _sounds_alike(written, fixed):
                self._names[name] = written
            elif fixed:
                self._names[name] = fixed
            elif written and not NOT_LATIN.search(written) and (not self._japanese_kanji(name, source)
                                                                  or _ROMAJI.fullmatch(written.lower())):
                # A Japanese kanji name only from a romaji answer: not "Nếu Lá" (若葉 read for its meaning).
                self._names[name] = written
        latin_target = self.store.settings.target_lang not in ("ja", "zh", "zh-Hant", "ko")
        if latin_target:
            translation = without_echo(source, translation)
        return label_speaker(source, translation, self._names, raw_names=not latin_target)

    @staticmethod
    def _keys_kept(source: str, translation: str) -> str:
        """Keys in brackets ("[F]") as the game wrote them; a model may rewrite the letter ("[Đ]")."""
        keys = re.findall(r"\[([A-Za-z0-9]{1,3})\]", source)
        found = re.findall(r"\[([^\]\s]{1,3})\]", translation)
        if not keys or len(keys) != len(found):
            return translation
        it = iter(keys)
        return re.sub(r"\[([^\]\s]{1,3})\]", lambda m: f"[{next(it)}]", translation)

    def _capitalised(self, translation: str) -> str:
        """A line that opens with a called title copied from a term ("tiền bối, em…") still starts a sentence."""
        if self.store.settings.target_lang != "vi":
            return translation
        translation = re.sub(r"\s*ㅋ{2,}\s*", " haha ", re.sub(r"\s*[ㅠㅜ]{2,}\s*", " huhu ", translation))
        translation = re.sub(r"\s*ㅎ{2,}\s*", " hehe ", translation)
        translation = _collapsed(re.sub(r" {2,}", " ", translation))
        translation = re.sub(r"(?<![가-힣ㄱ-ㅎ])[ㄱ-ㅎ](?![가-힣ㄱ-ㅎ])", "", translation)
        translation = re.sub(r"(?<![가-힣])ㅇㅋ", "ok", translation).replace("ㅇㅇ", "ừ").replace("ㄴㄴ", "không").strip()
        if "\n" in translation:
            return "\n".join(self._capitalised(line) for line in translation.split("\n"))
        label, colon, body = translation.partition(": ")
        if colon and len(label) <= 24:
            # A stat key ("hp", "mp") stays as written.
            head = label if len(label) <= 3 and label.isascii() else self._capitalised(label)
            return f"{head}: {self._capitalised(body)}"
        # Not a brand that starts in lower case (iPhone, eBay) or an item count ("x3").
        if not translation[:1].islower() or re.match(r"[a-z]+[A-Z]|x\d", translation):
            return translation
        return translation[0].upper() + translation[1:]

    def _checked(self, translator: Translator, source: str, translation: str, history: list) -> str:
        """A subtitle answered in the source's script, or with an earlier line's translation, asked again alone."""
        # "Vâng." twice from two people is no copy.
        copied = len(_body(translation)) >= 12 and any(
            _body(translation) == _body(turn.translation) for turn in history if turn.source != source)
        if self._leaks(translation) or copied or _LOOP.search(translation):
            # Asked again without earlier lines to copy from, but with who is talking to whom.
            again = self._labelled(source, self._plainly(translator, source, self.relationships.address(source)),
                                   translator)
            translation = _without_hanzi(self._readable(source, again, names_only=True))
        return self._capitalised(translation)

    def _japanese_kanji(self, name: str, source: str) -> bool:
        if not re.fullmatch("[\u4e00-\u9fff々]+", name):
            return False
        lang = self.store.settings.source_lang
        return lang == "ja" or lang not in ("zh", "zh-Hant") and (
            bool(re.search("[\u3040-\u30ff]", source)) or any(self._kana_recent))

    def _leaks(self, translation: str) -> bool:
        """Source-script text left in a Vietnamese translation."""
        return self.store.settings.target_lang == "vi" and bool(NOT_LATIN.search(re.sub(r"【[^】]*】", "", translation)))

    def _readable(self, source: str, translation: str, names_only: bool = False) -> str:
        """Names left in hanzi, kana or hangul after a second try, spelled the way this game's names already are."""
        if not self._leaks(translation):
            return translation

        speaker = speaker_of(source) or ""

        def spelled(run: str) -> str:
            known = self._names.get(run)
            # Not a run that carries Korean grammar ("이라는", "획득했습니다") or sits glued to other words.
            if not known and run != speaker and (re.search("(?:을|를|이|가|은|는|라는|니다|어요|아요|했)$", run)
                                                  or len(run) > 4):
                return run
            # A retried row's hanzi might be a word, not a name; kana and hangul are spelled out either way.
            spell = len(run) <= 4 and not (names_only and re.search("[\u3400-\u9fff]", run))
            return known or (self._fixed_name(run, source) if spell else None) or run

        return re.sub(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7a3]+", lambda m: spelled(m.group(0)), translation)

    def _fixed_name(self, name: str, source: str = "") -> str | None:
        vietnamese = self.store.settings.target_lang == "vi"
        role = role_label(name) if vietnamese else None
        # Kanji names are read the Japanese way (美咲 is Misaki), so Hán-Việt is only for Chinese text.
        self._kana_recent.append(bool(re.search("[\u3040-\u30ff]", source)))
        lang = self.store.settings.source_lang
        han = len(re.findall("[\u4e00-\u9fff]", body_of(source)))
        titled = re.search("(?:师姐|师兄|师妹|师弟|王爷|太后|皇上|掌门|长老|宗主|掌柜)$", name)
        chinese = vietnamese and (lang in ("zh", "zh-Hant") or lang not in ("ja", "ko") and not any(self._kana_recent)
                                  and (han >= 4 or titled) and not re.search("[\u3040-\u30ff]", source))
        return role or romanized(name) or (han_viet(name) if chinese else None)

    def _voice_note(self, heard: dict) -> str:
        """Speech has no speaker names, but the voice says whether a man or a woman is talking, and when it changes."""
        gender = gender_of(heard.get("pitch", 0.0))
        if not gender:
            return ""
        pitch = heard.get("pitch", 0.0)
        if self._last_voice and self._last_voice[0] != heard["id"]:
            self._previous_voice, self._previous_pitch = self._last_voice[1], self._last_voice[2]
        self._last_voice = (heard["id"], gender, pitch)
        person = "man" if gender == "male" else "woman"
        before, then = self._previous_voice, self._previous_pitch
        if before and before != gender:
            switched = f" (the line before was a {'man' if before == 'male' else 'woman'}'s)"
        elif before and then and max(pitch, then) >= 1.25 * min(pitch, then):
            switched = " (a different person from the line before)"
        else:
            switched = ""
        return f"Voice: a {person}{switched}. Use this only to choose the speaker's pronouns for themselves."

    def _translate_speech(self, translator: Translator, heard: dict) -> str:
        settings = self.store.settings
        known = self._cache.get(self._cache_key(heard["text"]))
        if known is not None:
            return known
        budget = speakable(settings.target_lang, heard.get("seconds", 0.0)) if settings.dub else ""
        # On the line itself, not in the notes: the notes open the prompt, and changing them rereads all of it.
        messages = build_messages(_punctuated(heard["text"]), self.context.turns(), settings.source_lang,
                                  settings.target_lang,
                                  settings.glossary, settings.notes,
                                  address=self.relationships.address(_punctuated(heard["text"]), gender_of(heard.get("pitch", 0.0))),
                                  budget=budget, aside=self._voice_note(heard))

        started = time.monotonic()

        def partial(translation: str) -> None:
            if self._stop.is_set():
                raise Superseded()
            with self._heard_lock:
                newer = self._finals[-1] if self._finals else self._heard
            if heard["final"] or not newer:
                pass
            elif newer["id"] != heard["id"]:
                raise Superseded()
            elif _bare(newer["text"]) != _bare(heard["text"]) and (
                    newer["final"] or (newer["ending"] and time.monotonic() - started < 0.3)):
                raise Superseded()
            self._speech_line(heard, translation, False)

        translation = translator.translate(messages, partial, urgent=True)
        if heard["final"]:
            translation = self._unrepeated(translator, heard, translation)
            if self._leaks(translation) or _LOOP.search(translation):
                translation = _collapsed(self._readable(heard["text"], self._plainly(translator, heard["text"])))
                # Hangul and kana left in are spelled out ("떡볶이" is tteokbokki); hanzi would be read as nothing.
                translation = re.sub(r"[\u3040-\u30ff\uac00-\ud7a3]+", lambda m: romanized(m.group(0)) or "", translation)
                translation = re.sub(r"[\u3400-\u9fff]+", "", translation).strip()
        # The label goes before labelling, which would turn "Con: Mẹ ơi…" into a spoken call "Con, mẹ ơi…".
        return self._labelled(heard["text"], unlabelled(heard["text"], translation), translator)

    def _unrepeated(self, translator: Translator, heard: dict, translation: str) -> str:
        """Given a long note on a short or garbled line, a small model can repeat an earlier answer; ask again plainly."""
        turns = self.context.turns()
        last = turns[-1].translation.strip() if turns and _bare(turns[-1].source) != _bare(heard["text"]) else ""
        # Unless the speaker really did start the same way again.
        repeated = turns and _bare(heard["text"])[:6] == _bare(turns[-1].source)[:6]
        if len(last) >= 12 and not repeated and translation.strip().startswith(last) and len(translation.strip()) > len(last) + 3:
            # HY-MT can repeat the previous line before the new one: keep only the new part.
            translation = translation.strip()[len(last):].lstrip(" ,.;:")
        earlier = {turn.translation for turn in turns if _bare(turn.source) != _bare(heard["text"])}
        # HY-MT can answer a short or garbled line with the last few translations strung together.
        # Two or more earlier lines inside one answer; one alone may be a real line starting the same ("Xin chào, ai…").
        strung = sum(1 for line in earlier if len(line) >= 8 and line in translation) >= 2
        if translation not in earlier and not strung:
            return translation
        again = self._plainly(translator, heard["text"])
        # Still the earlier line: better unsaid than said for the wrong person.
        return "" if again in earlier or sum(1 for line in earlier if len(line) >= 8 and line in again) >= 2 else again

    def _plainly(self, translator: Translator, text: str, address=None) -> str:
        """The line asked again on its own, with no earlier lines to copy from."""
        settings = self.store.settings
        plain = build_messages(text, [], settings.source_lang, settings.target_lang, settings.glossary, settings.notes,
                               address=address)
        return translator.translate(plain, urgent=True)


def _cut_off(block: Block, frame_height: int) -> bool:
    """Text touching the top or bottom edge is half out of view, like a webtoon bubble mid-scroll."""
    edge = min(0.005, 3 / max(1, frame_height))
    return block.box[1] <= edge or block.box[1] + block.box[3] >= 1 - edge


def _spoken_in(target: str, spoken: str) -> bool:
    return bool(spoken) and spoken == {"zh-Hant": "zh"}.get(target, target).split("-")[0]


def _bare(text: str) -> str:
    """Text without spaces and punctuation; recognizers often only add a comma once a sentence is finished."""
    return "".join(ch for ch in text if ch.isalnum())


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


def _texts(blocks: list[Block]) -> frozenset:
    return frozenset(b.text for b in blocks)


def _signature(blocks: list[Block]) -> frozenset:
    return frozenset(_key(b) for b in blocks)
