from __future__ import annotations

import json
import os
import sys
import threading
import urllib.error
import urllib.request
import webbrowser
from importlib import resources

from PySide6.QtCore import QObject, QPoint, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QAction, QColor, QFont, QFontMetrics, QGuiApplication, QIcon, QKeySequence, QPainter, QPen, QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QMenu, QProgressBar, QPushButton,
    QSystemTrayIcon, QVBoxLayout, QWidget,
)

from . import languages, updates
from .appearance import Style, paint_patch, paint_translation, pixel_font, place_translation, text_sizes
from .capture import exclude_window_from_capture
from .config import SettingsStore
from .hotkeys import GlobalHotkeys, label as shortcut
from .models import MODELS
from .server import serve

GREEN = QColor(46, 204, 113)
HEADER = 26
EDGE = 10
TAB = 330


class Bridge(QObject):
    event = Signal(dict)
    update_found = Signal(object)


class Api:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def call(self, path: str, body: dict | None = None) -> dict:
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            self.base_url + path, data=data,
            headers={"Content-Type": "application/json", "Origin": self.base_url},
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as exc:
            return {"error": json.loads(exc.read() or b"{}").get("error", str(exc))}


def float_above_everything(widget: QWidget) -> int:
    """Keep a window over full-screen apps and every Space; returns the macOS window number (0 elsewhere)."""
    if QGuiApplication.platformName() == "cocoa":
        import objc

        window = objc.objc_object(c_void_p=int(widget.winId())).window()
        window.setLevel_(25)
        window.setCollectionBehavior_((1 << 0) | (1 << 4) | (1 << 8))
        window.setHidesOnDeactivate_(False)
        return int(window.windowNumber())
    return 0


def hide_from_screen_capture(widget: QWidget) -> None:
    if QGuiApplication.platformName() == "windows":
        import ctypes

        ctypes.windll.user32.SetWindowDisplayAffinity(int(widget.winId()), 0x11)


def window_flags() -> Qt.WindowType:
    flags = Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
    # A Tool window keeps the frames out of the Windows taskbar; on macOS it would vanish when the app loses focus.
    return flags | (Qt.Tool if sys.platform == "win32" else Qt.Window)


class SourceFrame(QWidget):
    """The green box you drag over the text. The header strip sits above the captured area so it is never OCR'd."""

    moved = Signal(QRect)
    removed = Signal()

    def __init__(self) -> None:
        super().__init__(None, window_flags())
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setMinimumSize(80, HEADER + 30)
        self.setMouseTracking(True)
        self._drag: tuple[str, QPoint, QRect] | None = None
        self.number = 1
        self.removable = False

    def _close_box(self) -> QRect:
        return QRect(TAB - HEADER, 0, HEADER, HEADER - 2) if self.removable else QRect()

    def area(self) -> QRect:
        return self.geometry().adjusted(0, HEADER, 0, 0)

    def set_area(self, area: QRect) -> None:
        self.setGeometry(area.adjusted(0, -HEADER, 0, 0))

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        body = self.rect().adjusted(1, HEADER, -1, -1)
        painter.fillRect(body, QColor(46, 204, 113, 28))
        painter.setPen(QPen(GREEN, 2))
        painter.drawRect(body)
        tab = QRect(0, 0, min(self.width(), TAB), HEADER - 2)
        painter.fillRect(tab, GREEN)
        painter.setPen(Qt.white)
        painter.setFont(QFont(self.font().family(), 11, QFont.DemiBold))
        painter.drawText(tab.adjusted(8, 0, -HEADER, 0), Qt.AlignVCenter,
                         f"Area {self.number} · drag over the text, resize from edges")
        if self.removable:
            painter.drawText(self._close_box(), Qt.AlignCenter, "✕")

    def _zone(self, pos: QPoint) -> str:
        zone = ""
        if pos.y() >= self.height() - EDGE:
            zone += "b"
        elif pos.y() <= EDGE and pos.x() > TAB:
            zone += "t"
        if pos.x() <= EDGE:
            zone += "l"
        elif pos.x() >= self.width() - EDGE:
            zone += "r"
        return zone or "move"

    def mousePressEvent(self, event) -> None:
        if self._close_box().contains(event.position().toPoint()):
            self.removed.emit()
            return
        self._drag = (self._zone(event.position().toPoint()), event.globalPosition().toPoint(), self.geometry())

    def mouseMoveEvent(self, event) -> None:
        if self._drag is None:
            cursors = {"l": Qt.SizeHorCursor, "r": Qt.SizeHorCursor, "t": Qt.SizeVerCursor, "b": Qt.SizeVerCursor,
                       "bl": Qt.SizeBDiagCursor, "br": Qt.SizeFDiagCursor, "tl": Qt.SizeFDiagCursor,
                       "tr": Qt.SizeBDiagCursor}
            self.setCursor(cursors.get(self._zone(event.position().toPoint()), Qt.SizeAllCursor))
            return
        zone, start, geo = self._drag
        delta = event.globalPosition().toPoint() - start
        if zone == "move":
            self.move(geo.topLeft() + delta)
            return
        rect = QRect(geo)
        if "l" in zone:
            rect.setLeft(min(geo.left() + delta.x(), geo.right() - self.minimumWidth()))
        if "r" in zone:
            rect.setRight(max(geo.right() + delta.x(), geo.left() + self.minimumWidth()))
        if "t" in zone:
            rect.setTop(min(geo.top() + delta.y(), geo.bottom() - self.minimumHeight()))
        if "b" in zone:
            rect.setBottom(max(geo.bottom() + delta.y(), geo.top() + self.minimumHeight()))
        self.setGeometry(rect)

    def mouseReleaseEvent(self, _event) -> None:
        if self._drag is None:
            return
        self._drag = None
        self.moved.emit(self.area())


class TranslationOverlay(QWidget):
    """Click-through box drawn over the original text and hidden from our own screen capture."""

    def __init__(self) -> None:
        super().__init__(None, window_flags() | Qt.WindowTransparentForInput | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.text = ""
        self.final = True
        self.style = Style()
        self._font_size = 20
        self._placement: tuple[QRect, float] | None = None
        self.hidden_by_user = False
        self.window_number = 0

    def ensure_native(self) -> None:
        if not self.window_number:
            self.setGeometry(QRect(-10000, -10000, 10, 10))
            self.show()
            self.window_number = float_above_everything(self)
            hide_from_screen_capture(self)
            self.hide()
        exclude_window_from_capture(self.window_number)

    def show_text(self, text: str, final: bool, area: QRect, line_height: float = 0.0) -> None:
        self.text, self.final = text, final
        self._placement = (area, line_height)
        if not text:
            return
        screen = QGuiApplication.screenAt(area.center()) or QGuiApplication.primaryScreen()
        rect, self._font_size = place_translation(text, area, self.style, line_height, screen.geometry())
        self.setGeometry(rect)
        if self.hidden_by_user:
            return
        self.show()
        self.raise_()
        self.update()

    def restyle(self, style: Style) -> None:
        self.style = style
        if self.isVisible() and self.text and self._placement:
            self.show_text(self.text, self.final, *self._placement)

    def paintEvent(self, _event) -> None:
        paint_translation(QPainter(self), self.rect(), self.text, self._font_size, self.style, self.final)


class ScreenOverlay(QWidget):
    """Whole-screen mode: one click-through layer that repaints each translated block over its original text."""

    def __init__(self) -> None:
        super().__init__(None, window_flags() | Qt.WindowTransparentForInput | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.blocks: list[dict] = []
        self.translations: dict[str, tuple[str, bool]] = {}
        self.skip = QRect()
        self.text_size = "x1"
        self._layouts: dict[int, tuple[QRectF, int]] = {}
        self.window_number = 0

    def cover(self, geometry: QRect) -> None:
        self.blocks, self._layouts = [], {}
        self.setGeometry(geometry)
        self.show()
        if not self.window_number:
            self.window_number = float_above_everything(self)
            hide_from_screen_capture(self)
        exclude_window_from_capture(self.window_number)

    def set_blocks(self, blocks: list[dict]) -> None:
        self.blocks = blocks
        for block in blocks:
            if block.get("translation"):
                self.translations[block["text"]] = (block["translation"], True)
        self._layouts = {i: self._layout(b) for i, b in enumerate(blocks) if b["text"] in self.translations}
        self.update()

    def _unchanged(self, block: dict) -> bool:
        text, final = self.translations.get(block["text"], ("", False))
        return final and " ".join(text.split()).casefold() == " ".join(block["text"].split()).casefold()

    def set_translation(self, source: str, text: str, final: bool) -> None:
        self.translations[source] = (text, final)
        touched = False
        for i, block in enumerate(self.blocks):
            if block["text"] == source:
                self._layouts[i] = self._layout(block)
                touched = True
        if touched:
            self.update()

    def forget(self) -> None:
        self.translations.clear()
        self.set_blocks([])

    def _rect(self, block: dict) -> QRectF:
        x, y, w, h = block["box"]
        return QRectF(x * self.width(), y * self.height(), w * self.width(), h * self.height()).adjusted(-3, -2, 3, 2)

    def _layout(self, block: dict) -> tuple[QRectF, int]:
        rect = self._rect(block)
        text = self.translations[block["text"]][0]
        sizes = text_sizes(block["line_height"] * self.height(), rect.height(), self.text_size)
        flags = Qt.TextWordWrap
        for size in sizes:
            needed = QFontMetrics(pixel_font(size)).boundingRect(
                rect.toRect().adjusted(2, 0, -2, 0), flags, text)
            if needed.height() <= rect.height() and needed.width() <= rect.width() - 4:
                return rect, size
        size = sizes[-1]
        needed = QFontMetrics(pixel_font(size)).boundingRect(
            QRect(0, 0, int(rect.width()) - 4, 10_000), flags, text)
        rect.setHeight(max(rect.height(), needed.height() + 4))
        return rect, size

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        skip = QRectF(self.skip.translated(-self.geometry().topLeft()))
        for i, block in enumerate(self.blocks):
            rect = self._rect(block)
            if (not skip.isEmpty() and skip.intersects(rect)) or self._unchanged(block):
                continue
            if i not in self._layouts:
                painter.setPen(QPen(QColor(46, 204, 113, 140), 1, Qt.DashLine))
                painter.setBrush(Qt.NoBrush)
                painter.drawRoundedRect(rect, 3, 3)
                continue
            rect, size = self._layouts[i]
            text, final = self.translations[block["text"]]
            paint_patch(painter, rect, text, size, block["bg"], block["fg"], final)


class ScreenFlash(QWidget):
    """A brief green frame with the screen's number, so you can see which monitor you just picked."""

    def __init__(self) -> None:
        super().__init__(None, window_flags() | Qt.WindowTransparentForInput | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.label = ""
        self._timer = QTimer(self, singleShot=True, interval=1200)
        self._timer.timeout.connect(self.hide)

    def show_on(self, screen, number: int) -> None:
        self.label = f"Screen {number}"
        self.setGeometry(screen.geometry())
        self.show()
        float_above_everything(self)
        hide_from_screen_capture(self)
        self.raise_()
        self.update()
        self._timer.start()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(GREEN, 8))
        painter.drawRect(self.rect().adjusted(4, 4, -4, -4))
        badge = QRect(0, 0, 260, 90)
        badge.moveCenter(self.rect().center())
        painter.setPen(Qt.NoPen)
        painter.setBrush(GREEN)
        painter.drawRoundedRect(badge, 14, 14)
        painter.setPen(Qt.white)
        painter.setFont(pixel_font(40))
        painter.drawText(badge, Qt.AlignCenter, self.label)


TEXT_SIZES = {"x0.75": "Smaller text", "x1": "Normal text", "x1.25": "Larger text", "x1.5": "Largest text"}


class MainWindow(QWidget):

    def __init__(self, api: Api, store: SettingsStore, bridge: Bridge) -> None:
        super().__init__()
        self.api, self.store = api, store
        self.running = False
        self.setWindowTitle("Screen Translator")
        self.setMinimumWidth(460)

        self.sources: list[SourceFrame] = []
        self.overlays: list[TranslationOverlay] = []
        self.screen_overlay = ScreenOverlay()
        bridge.event.connect(self.on_event)
        self.bridge = bridge
        bridge.update_found.connect(self.show_update)

        self.scope = QComboBox()
        self.scope.addItem("A selected area (subtitles, dialogue)", "area")
        self.scope.addItem("The whole screen", "screen")
        self.listen = QCheckBox("Also translate what's said (speech, voice acting)")
        self.listen.setToolTip("Listens to the PC's sound and subtitles speech at the bottom of the screen.")
        self.speech_overlay = TranslationOverlay()
        self.dub = QCheckBox("Read them aloud too (dubbing)")
        self.dub.setToolTip("Speaks each translated line in a voice that follows the speaker's gender, pitch and "
                            "emotion. Uses the voices installed in Windows.")
        self.dub_voice = QComboBox()
        self.dub_voice.setToolTip("Match the speaker picks a male or female voice from how they sound. Pick an "
                                  "accent or one voice to always use it.")
        self.dub_original = QComboBox()
        for value, label in (("duck", "Turn game sound down"), ("keep", "Keep game sound"),
                             ("mute", "Mute game sound")):
            self.dub_original.addItem(label, value)
        self.dub_row = QWidget()
        dub_layout = QHBoxLayout(self.dub_row)
        dub_layout.setContentsMargins(0, 0, 0, 0)
        dub_layout.addWidget(self.dub_voice, 2)
        dub_layout.addWidget(self.dub_original, 1)
        self.dub_label = QLabel("Dub voice")
        self.screen_pick = QComboBox()
        self.flash = ScreenFlash()
        self.refresh_screens()
        self.source_lang = QComboBox()
        self.target_lang = QComboBox()
        for code, label in languages.LANGUAGES.items():
            self.source_lang.addItem(label, code)
            if code != "auto":
                self.target_lang.addItem(label, code)
        self.model = QComboBox()
        for spec in MODELS.values():
            self.model.addItem(spec.label, spec.key)
        self.position = QComboBox()
        self.position.addItem("Over the text", "cover")
        self.position.addItem("Below the text", "below")
        self.text_size = QComboBox()
        self.text_size.setToolTip("Normal draws the translation about the size of the original text.")
        for key, label in TEXT_SIZES.items():
            self.text_size.addItem(label, key)
        self.box = QCheckBox("Dark box")
        self.overlay_row = QWidget()
        overlay_layout = QHBoxLayout(self.overlay_row)
        overlay_layout.setContentsMargins(0, 0, 0, 0)
        overlay_layout.addWidget(self.position, 1)
        overlay_layout.addWidget(self.text_size, 1)
        overlay_layout.addWidget(self.box)
        self.min_text = QComboBox()
        for px, label in ((0, "Off: translate all text"), (12, "Smaller than 12 px (tiny print)"),
                          (16, "Smaller than 16 px (chat, captions)"), (20, "Smaller than 20 px")):
            self.min_text.addItem(label, px)
        self.min_text.setToolTip("Tiny text such as stream chat is often misread, which gives nonsense translations.")
        self.min_text_label = QLabel("Skip text")
        self.notes = QLineEdit()
        self.notes.setPlaceholderText("e.g. GTA V gameplay, a Korean drama, a Japanese visual novel")
        self.notes.setToolTip("Telling the model what you are watching makes slang and idioms translate properly.")
        self.download = QPushButton("Download model")
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setMaximumHeight(8)
        self.select = QPushButton("Select areas")
        self.select.setCheckable(True)
        self.add_area = QPushButton("+ Add area")
        self.forget = QPushButton("Forget context")
        self.more = QPushButton("Glossary && more…")
        self.start = QPushButton()
        self.start.setMinimumHeight(52)
        self.overlays_hidden = False
        self.hotkeys: GlobalHotkeys | None = None
        self.shortcuts = QLabel()
        self.shortcuts.setWordWrap(True)
        self.shortcuts.setStyleSheet("color: gray; font-size: 11px;")
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.last = QLabel()
        self.last.setWordWrap(True)
        self.last.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.speed = QLabel()
        self.speed.setWordWrap(True)
        self.speed.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.speed.setStyleSheet("color: gray; font-size: 11px;")
        self.devices: dict[str, str] = {}
        self.timings: dict[str, str] = {}
        self.copy_log = QPushButton("Copy log")
        self.copy_log.setToolTip("Copies what was read and translated, with timings, to paste into a bug report.")

        self.update_note = QLabel()
        self.update_note.setOpenExternalLinks(True)
        self.update_note.setWordWrap(True)
        self.update_note.setStyleSheet("background: #1e3a8a; color: white; padding: 6px 8px; border-radius: 4px;")
        self.update_note.hide()

        self.screen_label = QLabel("Screen")
        form = QGridLayout()
        rows = ((QLabel("Translate"), self.scope), (QLabel(""), self.listen), (QLabel(""), self.dub),
                (self.dub_label, self.dub_row), (self.screen_label, self.screen_pick),
                (QLabel("From"), self.source_lang), (QLabel("To"), self.target_lang),
                (QLabel("Model"), self.model), (QLabel("Overlay"), self.overlay_row),
                (self.min_text_label, self.min_text),
                (QLabel("Watching"), self.notes))
        for row, (label, field) in enumerate(rows):
            form.addWidget(label, row, 0)
            form.addWidget(field, row, 1)
        form.setColumnStretch(1, 1)
        download_row = QHBoxLayout()
        download_row.addWidget(self.download)
        download_row.addWidget(self.progress, 1)
        tools = QHBoxLayout()
        for button in (self.select, self.add_area, self.forget, self.more):
            tools.addWidget(button)
        layout = QVBoxLayout(self)
        layout.addWidget(self.update_note)
        layout.addLayout(form)
        layout.addLayout(download_row)
        layout.addLayout(tools)
        layout.addWidget(self.start)
        layout.addWidget(self.shortcuts)
        layout.addWidget(self.status)
        layout.addWidget(self.last)
        speed_row = QHBoxLayout()
        speed_row.addWidget(self.speed, 1)
        speed_row.addWidget(self.copy_log, 0, Qt.AlignBottom)
        layout.addLayout(speed_row)

        self.select.toggled.connect(self.toggle_source)
        self.add_area.clicked.connect(self.new_area)
        self.start.clicked.connect(self.toggle_running)
        self.forget.clicked.connect(lambda: self.api.call("/api/context/clear", {}))
        self.more.clicked.connect(lambda: webbrowser.open(f"{self.api.base_url}/panel"))
        self.download.clicked.connect(self.toggle_download)
        self.copy_log.clicked.connect(self.copy_diagnostics)
        for combo, key in ((self.source_lang, "source_lang"), (self.target_lang, "target_lang"),
                           (self.model, "model"), (self.scope, "scope"),
                           (self.screen_pick, "screen_name"), (self.min_text, "min_text_px")):
            combo.currentIndexChanged.connect(lambda _i, c=combo, k=key: self.api.call("/api/settings", {k: c.currentData()}))
        self.scope.currentIndexChanged.connect(lambda _i: self.show_scope())
        self.listen.toggled.connect(lambda on: (self.api.call("/api/settings", {"listen": on}), self.refresh_model(),
                                                self.show_dub()))
        self.dub.toggled.connect(lambda on: (self.api.call("/api/settings", {"dub": on}), self.show_dub()))
        self.dub_voice.activated.connect(
            lambda _i: self.api.call("/api/settings", {"dub_voice": self.dub_voice.currentData()}))
        self.dub_original.activated.connect(
            lambda _i: self.api.call("/api/settings", {"dub_original": self.dub_original.currentData()}))
        self.target_lang.currentIndexChanged.connect(lambda _i: QTimer.singleShot(0, self.refresh_voices))
        self.position.currentIndexChanged.connect(lambda _i: self.apply_style({"overlay_mode": self.position.currentData()}))
        self.text_size.currentIndexChanged.connect(lambda _i: self.apply_style({"text_size": self.text_size.currentData()}))
        self.box.toggled.connect(lambda on: self.apply_style({"overlay_box": on}))
        self.notes.editingFinished.connect(lambda: self.api.call("/api/settings", {"notes": self.notes.text().strip()}))
        self.screen_pick.activated.connect(lambda _i: self.use_screen())
        QShortcut(QKeySequence("Ctrl+Return"), self, self.toggle_running)
        for number in range(1, 10):
            QShortcut(QKeySequence(f"Ctrl+{number}"), self, lambda n=number: self.choose_screen(n))
        app = QGuiApplication.instance()
        app.screenAdded.connect(lambda _s: self.refresh_screens())
        app.screenRemoved.connect(lambda _s: QTimer.singleShot(0, self.refresh_screens))

        self.load()

    def load(self) -> None:
        s = self.store.settings
        for combo, value in ((self.source_lang, s.source_lang), (self.target_lang, s.target_lang),
                             (self.model, s.model), (self.scope, s.scope),
                             (self.screen_pick, s.screen_name), (self.min_text, s.min_text_px)):
            combo.blockSignals(True)
            combo.setCurrentIndex(max(0, combo.findData(value)))
            combo.blockSignals(False)
        self.notes.setText(s.notes)
        for widget, value, fallback in ((self.position, s.overlay_mode, "cover"), (self.text_size, s.text_size, "x1")):
            widget.blockSignals(True)
            widget.setCurrentIndex(widget.findData(value if widget.findData(value) >= 0 else fallback))
            widget.blockSignals(False)
        self.box.blockSignals(True)
        self.box.setChecked(s.overlay_box)
        self.box.blockSignals(False)
        for check, value in ((self.listen, s.listen), (self.dub, s.dub)):
            check.blockSignals(True)
            check.setChecked(value)
            check.blockSignals(False)
        self.dub_original.blockSignals(True)
        self.dub_original.setCurrentIndex(max(0, self.dub_original.findData(s.dub_original)))
        self.dub_original.blockSignals(False)
        self.show_dub()
        self.screen_overlay.text_size = self.text_size.currentData()
        for area in s.areas or ([s.area] if s.area else []):
            self.add_source(QRect(*area))
        if not self.sources:
            screen = QGuiApplication.primaryScreen().availableGeometry()
            self.add_source(QRect(screen.center().x() - 400, screen.bottom() - 220, 800, 120))
        self.refresh_model()
        self.show_scope()
        self.set_running(False)

    def whole_screen(self) -> bool:
        return self.scope.currentData() == "screen"

    def listening(self) -> bool:
        return self.listen.isChecked()

    def show_scope(self) -> None:
        screen = self.whole_screen()
        for widget in (self.screen_label, self.screen_pick):
            widget.setVisible(screen and self.screen_pick.count() > 1)
        for widget in (self.min_text_label, self.min_text):
            widget.setVisible(screen)
        for widget in (self.position, self.box):
            widget.setVisible(not screen)
        self.select.setVisible(not screen)
        self.add_area.setVisible(not screen and self.select.isChecked())
        if screen:
            self.select.setChecked(False)
        if not self.running:
            self.set_running(False)

    def show_dub(self) -> None:
        self.dub.setVisible(self.listening())
        on = self.listening() and self.dub.isChecked()
        for widget in (self.dub_label, self.dub_row):
            widget.setVisible(on)
        if on and self.dub_voice.count() == 0:
            self.refresh_voices()

    def refresh_voices(self) -> None:
        if not (self.listening() and self.dub.isChecked()):
            self.dub_voice.clear()
            return
        result = self.api.call("/api/voices")
        choices = result.get("voices") or []
        self.dub_voice.blockSignals(True)
        self.dub_voice.clear()
        for choice in choices:
            self.dub_voice.addItem(choice["label"], choice["value"])
        self.dub_voice.setCurrentIndex(max(0, self.dub_voice.findData(self.store.settings.dub_voice)))
        self.dub_voice.blockSignals(False)
        if result.get("error"):
            self.status.setText(result["error"])
        elif len(choices) <= 1:
            self.status.setText(f"No voice for {self.target_lang.currentText()} is installed, so lines can't be "
                                "read aloud yet. Add one in Windows Settings → Time & language → Speech.")

    def subtitle_area(self) -> QRect:
        """Where spoken lines appear: low and centred on the screen being translated, like film subtitles."""
        screen = (self.chosen_screen() if self.whole_screen() else QGuiApplication.primaryScreen()).geometry()
        width = int(screen.width() * 0.7)
        return QRect(screen.center().x() - width // 2, screen.bottom() - int(screen.height() * 0.08) - 110, width, 110)

    def refresh_screens(self) -> None:
        current = self.screen_pick.currentData() or self.store.settings.screen_name
        screens = sorted(QGuiApplication.screens(), key=lambda s: (s.geometry().x(), s.geometry().y()))
        self.screen_pick.blockSignals(True)
        self.screen_pick.clear()
        for number, screen in enumerate(screens, 1):
            size = screen.geometry()
            laptop = " (laptop)" if "built-in" in screen.name().lower() else ""
            shortcut = f"   Ctrl+{number}" if number < 10 else ""
            self.screen_pick.addItem(f"{number} · {screen.name()}{laptop} · {size.width()}×{size.height()}{shortcut}",
                                     screen.name())
        self.screen_pick.setCurrentIndex(max(0, self.screen_pick.findData(current)))
        self.screen_pick.blockSignals(False)
        if hasattr(self, "screen_label"):
            self.show_scope()

    def choose_screen(self, number: int) -> None:
        if number > self.screen_pick.count():
            return
        if not self.running:
            self.scope.setCurrentIndex(self.scope.findData("screen"))
        self.screen_pick.setCurrentIndex(number - 1)
        self.use_screen()

    def use_screen(self) -> None:
        """Flash the picked screen so it is obvious which one it is, and move a running translation onto it."""
        screen = self.chosen_screen()
        self.flash.show_on(screen, self.screen_pick.currentIndex() + 1)
        if self.running and self.whole_screen():
            self.api.call("/api/settings", {"screen_rect": native_rect(screen.geometry()),
                                            "screen_height": screen.geometry().height()})
            self.screen_overlay.cover(screen.geometry())

    def chosen_screen(self):
        name = self.screen_pick.currentData()
        return next((s for s in QGuiApplication.screens() if s.name() == name), None) or QGuiApplication.primaryScreen()

    def missing_download(self) -> tuple[str, str, int] | None:
        """(key, label, size) of the next file Start needs, if any."""
        state = self.api.call("/api/state")
        model = next((m for m in state.get("models", []) if m["key"] == self.model.currentData()), None)
        if model and not model["downloaded"] and not self.store.settings.custom_model_path:
            return model["key"], "model", model["size"]
        speech = state.get("speech")
        if hasattr(self, "listen") and self.listening() and speech and not speech["downloaded"]:
            return "speech", "speech recognition", speech["size"]
        return None

    def refresh_model(self) -> None:
        if not hasattr(self, "download"):
            return
        download = self.api.call("/api/state").get("download") or {}
        active = bool(download.get("active"))
        missing = self.missing_download()
        self.download.setVisible(bool(missing) or active)
        self.progress.setVisible(active)
        self.download.setText("Cancel download" if active else
                              f"Download {missing[1]} ({missing[2] / 1e9:.1f} GB)" if missing else "")

    def toggle_download(self) -> None:
        state = self.api.call("/api/state")
        if (state.get("download") or {}).get("active"):
            self.api.call("/api/models/download/cancel", {})
        elif missing := self.missing_download():
            result = self.api.call(f"/api/models/{missing[0]}/download", {})
            if "error" in result:
                self.status.setText(result["error"])

    def add_source(self, area: QRect) -> SourceFrame:
        frame = SourceFrame()
        frame.set_area(area)
        frame.moved.connect(self.save_areas)
        frame.removed.connect(lambda f=frame: self.remove_source(f))
        self.sources.append(frame)
        self.overlays.append(TranslationOverlay())
        self.overlays[-1].style = self.style()
        self.number_sources()
        return frame

    def style(self) -> Style:
        return Style(self.box.isChecked(), self.store.settings.overlay_opacity, self.position.currentData(),
                     self.text_size.currentData())

    def apply_style(self, patch: dict) -> None:
        style = self.style()
        for overlay in self.overlays:
            overlay.restyle(style)
        if "text_size" in patch:
            self.screen_overlay.text_size = style.text_size
            self.screen_overlay.set_blocks(self.screen_overlay.blocks)
        self.api.call("/api/settings", patch)

    def new_area(self) -> None:
        last = self.sources[-1].area()
        above = last.translated(0, -(last.height() + HEADER + 20))
        screen = (QGuiApplication.screenAt(last.center()) or QGuiApplication.primaryScreen()).availableGeometry()
        frame = self.add_source(above if above.top() - HEADER >= screen.top() else last.translated(0, last.height() + HEADER + 20))
        self.show_frame(frame)
        self.save_areas()

    def remove_source(self, frame: SourceFrame) -> None:
        index = self.sources.index(frame)
        self.sources.pop(index).close()
        self.overlays.pop(index).close()
        self.number_sources()
        self.save_areas()

    def number_sources(self) -> None:
        for number, frame in enumerate(self.sources, 1):
            frame.number, frame.removable = number, len(self.sources) > 1
            frame.update()

    def show_frame(self, frame: SourceFrame) -> None:
        frame.show()
        float_above_everything(frame)
        frame.raise_()

    def toggle_source(self, visible: bool) -> None:
        for frame in self.sources:
            if visible:
                self.show_frame(frame)
            else:
                frame.hide()
        self.add_area.setVisible(visible)
        self.select.setText("Done" if visible else "Select areas")

    def save_areas(self, *_args) -> None:
        areas = [frame.area() for frame in self.sources]
        self.api.call("/api/settings", {
            "areas": [[a.x(), a.y(), a.width(), a.height()] for a in areas],
            "capture_rects": [native_rect(a) for a in areas],
        })

    def enable_global_hotkeys(self) -> None:
        self.hotkeys = GlobalHotkeys(self)
        self.hotkeys.triggered.connect(self.on_hotkey)
        taken = self.hotkeys.register()
        text = (f"Works in any app: {shortcut('T')} start/stop · {shortcut('H')} hide or show translations · "
                f"{shortcut('1')}–9 translate screen 1–9")
        if taken:
            text += f"<br><span style='color:#d64545'>Already used by another app: {', '.join(taken)}</span>"
        self.shortcuts.setText(text)

    def on_hotkey(self, action: str) -> None:
        if action == "toggle":
            self.toggle_running()
        elif action == "hide":
            self.set_overlays_hidden(not self.overlays_hidden)
        elif action.startswith("screen"):
            self.choose_screen(int(action[len("screen"):]))

    def set_overlays_hidden(self, hidden: bool) -> None:
        """Peek at the original text; translations keep arriving underneath and show again when un-hidden."""
        self.overlays_hidden = hidden
        for overlay in [*self.overlays, self.speech_overlay]:
            overlay.hidden_by_user = hidden
            if hidden:
                overlay.hide()
            elif self.running and overlay.text and overlay._placement:
                overlay.show_text(overlay.text, overlay.final, *overlay._placement)
        self.screen_overlay.setVisible(not hidden and self.running and self.whole_screen())
        if self.running:
            self.status.setText(f"Translations hidden. Press {shortcut('H')} to show them again." if hidden else
                                "Translating.")

    def toggle_running(self) -> None:
        if self.running:
            self.api.call("/api/stop", {})
            self.set_running(False)
            return
        self.select.setChecked(False)
        if self.whole_screen():
            geometry = self.chosen_screen().geometry()
            self.api.call("/api/settings", {"screen_rect": native_rect(geometry), "screen_height": geometry.height()})
            self.screen_overlay.skip = self.frameGeometry()
            self.screen_overlay.cover(geometry)
        else:
            self.save_areas()
            for overlay in self.overlays:
                overlay.ensure_native()
        if self.listening():
            self.speech_overlay.style = Style(True, self.store.settings.overlay_opacity, "cover",
                                              self.text_size.currentData())
            self.speech_overlay.ensure_native()
        result = self.api.call("/api/start", {})
        if "error" in result:
            self.status.setText(result["error"])
            return
        self.set_running(True)

    def set_running(self, running: bool) -> None:
        self.running = running
        self.start.setText("■  Stop" if running else "▶  Start")
        color = "#d64545" if running else "#2ea44f"
        self.start.setStyleSheet(f"QPushButton {{ background: {color}; color: white; font-size: 18px; "
                                 f"font-weight: 600; border-radius: 8px; }}")
        self.scope.setEnabled(not running)
        if running:
            self.timings = {}
            self.show_speed()
        if not running:
            self.set_overlays_hidden(False)
            for overlay in self.overlays:
                overlay.hide()
            self.screen_overlay.hide()
            self.speech_overlay.hide()
            if self.whole_screen():
                self.status.setText("Start to translate every piece of text on the screen in place.")
            else:
                self.status.setText("Click “Select areas”, drag the green box over the text (add more boxes if the "
                                    "text is in several places), then Start.")

    def check_for_updates(self) -> None:
        if self.store.settings.check_updates:
            threading.Thread(target=lambda: self.bridge.update_found.emit(updates.check()), name="update-check",
                             daemon=True).start()

    def show_update(self, release: "updates.Release | None") -> None:
        if release is None or not self.store.settings.check_updates:
            return
        self.update_note.setText(f"Screen Translator {release.version} is out. "
                                 f"<a style='color: #fcd34d' href='{release.url}'>Download it</a>")
        self.update_note.show()

    def on_event(self, event: dict) -> None:
        kind = event["type"]
        if kind == "line" and self.running:
            area = event.get("area", 0)
            if area == "speech":
                self.speech_overlay.show_text(event["translation"], event["final"], self.subtitle_area())
                if event["final"]:
                    self.show_speed(speech=f"speech translated {event['ms'] / 1000:.1f} s after it ended")
            elif area < len(self.overlays):
                self.overlays[area].show_text(event["translation"], event["final"], self.sources[area].area(),
                                              event.get("line_height", 0.0))
            if event["final"]:
                self.last.setText(f"<span style='color:gray'>{event['source']}</span><br>{event['translation']}"
                                  f"<br><span style='color:gray;font-size:11px'>{event['ms']} ms · "
                                  f"{event['context_lines']} lines of context</span>")
        elif kind == "screen" and self.running:
            self.screen_overlay.skip = self.frameGeometry()
            self.screen_overlay.set_blocks(event["blocks"])
            if event["blocks"]:
                parts = event.get("ocr") or {}
                detail = (f" (finding text {parts['find']} ms, reading it {parts['read']} ms)"
                          if "find" in parts and "read" in parts else "")
                self.show_speed(read=f"reading the screen {event['ms']} ms{detail}")
        elif kind == "screen_text" and self.running:
            self.screen_overlay.set_translation(event["source"], event["translation"], event["final"])
            if event["final"] and "ms" in event:
                pieces = event.get("batch", 1)
                steps = event.get("steps") or {}
                detail = (f" (reading the list {steps['list']} ms, then {steps['steps']} steps of {steps['step']} ms)"
                          if steps.get("steps") else "")
                self.show_speed(translate=f"translating {pieces} piece{'s' if pieces > 1 else ''} "
                                          f"{event['ms'] / 1000:.1f} s{detail}")
        elif kind == "ocr" and self.running:
            self.show_speed(read=f"reading the text {event['ms']} ms")
        elif kind == "heard" and self.running and event["final"]:
            self.show_speed(hear=f"recognizing speech {event['ms']} ms")
        elif kind == "devices":
            self.devices = event
            self.show_speed()
        elif kind == "dub":
            if event.get("error"):
                self.status.setText(event["error"])
            else:
                self.show_speed(dub=f"voice ready in {event['ms']} ms ({event['voice']})")

        elif kind == "clear":
            area = event.get("area")
            for index, overlay in enumerate(self.overlays):
                if area is None or area == index:
                    overlay.hide()
            if area is None:
                self.screen_overlay.forget()
        elif kind == "status":
            if event["state"] == "error":
                self.set_running(False)
                self.status.setText(event["error"])
            elif event["state"] == "loading":
                self.status.setText("Loading the model…")
            elif event["state"] == "running":
                text = ("Translating the whole screen. Text under this window is left alone." if self.whole_screen()
                        else "Translating. The box over the text shows the translation.")
                if self.listening():
                    text += " Speech is subtitled at the bottom of the screen."
                self.status.setText(text)
        elif kind == "download":
            if event.get("error") and event["error"] != "Cancelled":
                self.status.setText(event["error"])
            if event.get("active"):
                self.progress.setMaximum(1000)
                self.progress.setValue(int(1000 * event["done"] / max(1, event["total"])))
            QTimer.singleShot(0, self.refresh_model)

    def copy_diagnostics(self) -> None:
        result = self.api.call("/api/log")
        if "text" not in result:
            self.status.setText(f"Could not read the log: {result.get('error', 'no answer')}")
            return
        QGuiApplication.clipboard().setText(result["text"])
        self.status.setText(f"Log copied, ready to paste. It is also saved in {result['path']}")

    def show_speed(self, **timings: str) -> None:
        self.timings.update(timings)
        lines = []
        model, ocr = self.devices.get("model"), self.devices.get("ocr")
        where = [f"Model on {model}" if model else "", f"text reading on {ocr}" if ocr else ""]
        if any(where):
            lines.append(" · ".join(w for w in where if w))
        steps = " · ".join(self.timings[k] for k in ("read", "translate", "hear", "speech", "dub") if k in self.timings)
        if steps:
            lines.append(steps[0].upper() + steps[1:])
        text = "<br>".join(lines)
        if model == "CPU" and self.store.settings.gpu_layers != 0:
            text += ("<br><span style='color:#d97706'>No usable graphics card was found, so the model runs on the "
                     "CPU, which is slow. Updating the graphics driver usually fixes this; Qwen3.5 2B is faster "
                     "meanwhile.</span>")
        self.speed.setText(text)

    def closeEvent(self, event) -> None:
        self.api.call("/api/stop", {})
        for widget in [*self.sources, *self.overlays, self.speech_overlay, self.screen_overlay, self.flash]:
            widget.close()
        super().closeEvent(event)


def native_rect(area: QRect) -> dict:
    """Map Qt's device-independent coordinates to the physical pixels screen capture uses on Windows/Linux."""
    screen = QGuiApplication.screenAt(area.center()) or QGuiApplication.primaryScreen()
    ratio = 1.0 if sys.platform == "darwin" else screen.devicePixelRatio()
    origin = screen.geometry().topLeft()
    return {
        "left": round(origin.x() + (area.x() - origin.x()) * ratio),
        "top": round(origin.y() + (area.y() - origin.y()) * ratio),
        "width": round(area.width() * ratio),
        "height": round(area.height() * ratio),
    }


def bring_to_front(window: QWidget) -> None:
    window.showNormal()
    window.raise_()
    window.activateWindow()
    if QGuiApplication.platformName() == "cocoa":
        import AppKit

        AppKit.NSApp.activateIgnoringOtherApps_(True)


def tray_icon() -> QIcon:
    pixmap = QPixmap(44, 44)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(QPen(Qt.black, 3))
    painter.drawRoundedRect(3, 5, 38, 34, 7, 7)
    painter.setFont(pixel_font(22))
    painter.drawText(QRect(3, 5, 38, 34), Qt.AlignCenter, "文A")
    painter.end()
    icon = QIcon(pixmap)
    icon.setIsMask(True)
    return icon


def make_tray(window: "MainWindow") -> QSystemTrayIcon:
    tray = QSystemTrayIcon(tray_icon(), window)
    tray.setToolTip("Screen Translator")
    menu = QMenu()
    show = QAction("Show Screen Translator", menu)
    toggle = QAction("Start", menu)
    quit_app = QAction("Quit", menu)
    check_updates = QAction("Check for updates at startup", menu)
    check_updates.setCheckable(True)
    check_updates.setChecked(window.store.settings.check_updates)
    check_updates.toggled.connect(lambda on: window.api.call("/api/settings", {"check_updates": on}))
    show.triggered.connect(lambda: bring_to_front(window))
    toggle.triggered.connect(window.toggle_running)
    quit_app.triggered.connect(window.close)
    menu.aboutToShow.connect(lambda: toggle.setText(f"{'Stop' if window.running else 'Start'}    {shortcut('T')}"))
    for action in (show, toggle, check_updates, quit_app):
        menu.addAction(action)
    tray.setContextMenu(menu)
    tray.show()
    window.tray_menu = menu
    return tray


def run(store: SettingsStore) -> None:
    # Qt's raise() also activates the app on macOS, which swipes away the full-screen Space a box pops up over.
    os.environ.setdefault("QT_MAC_SET_RAISE_PROCESS", "0")
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Screen Translator")
    app.setWindowIcon(QIcon(str(resources.files("screen_translator") / "web" / "icon.png")))
    bridge = Bridge()
    ready, stop = threading.Event(), threading.Event()
    server = threading.Thread(target=serve, args=(store, ready, stop, bridge.event.emit), name="server", daemon=True)
    server.start()
    if not ready.wait(15):
        raise SystemExit(f"Could not start the local server on port {store.settings.port}; is it already in use?")

    if QGuiApplication.platformName() == "cocoa":
        import AppKit

        # Only a background (accessory) app may put windows over another app's full-screen Space, e.g. full-screen
        # YouTube; it must be set before any window exists, and the menu-bar icon stands in for the Dock icon.
        AppKit.NSApp.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
    window = MainWindow(Api(f"http://127.0.0.1:{store.settings.port}"), store, bridge)
    tray = make_tray(window)
    window.enable_global_hotkeys()
    window.check_for_updates()
    window.show()
    bring_to_front(window)
    app.exec()
    tray.hide()
    stop.set()
    server.join(10)
