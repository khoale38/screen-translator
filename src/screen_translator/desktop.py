from __future__ import annotations

import json
import os
import sys
import threading
import urllib.error
import urllib.request
import webbrowser

from PySide6.QtCore import QObject, QPoint, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QAction, QColor, QFont, QFontMetrics, QGuiApplication, QIcon, QKeySequence, QPainter, QPen, QPixmap, QShortcut,
)
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QMenu, QProgressBar, QPushButton,
    QSystemTrayIcon, QVBoxLayout, QWidget,
)

from . import languages
from .capture import exclude_window_from_capture
from .config import SettingsStore
from .models import MODELS
from .server import serve

GREEN = QColor(46, 204, 113)
HEADER = 26
EDGE = 10
TAB = 330


class Bridge(QObject):
    event = Signal(dict)


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
        self.opacity = 0.92
        self.box = True
        self.anchor = QRect()
        self.window_number = 0

    def ensure_native(self) -> None:
        if not self.window_number:
            self.setGeometry(QRect(-10000, -10000, 10, 10))
            self.show()
            self.window_number = float_above_everything(self)
            hide_from_screen_capture(self)
            self.hide()
        exclude_window_from_capture(self.window_number)

    def show_text(self, text: str, final: bool, area: QRect, mode: str, line_height: float = 0.0) -> None:
        self.text, self.final = text, final
        if not text:
            return
        rect = QRect(area)
        if mode == "below":
            rect.moveTop(area.bottom() + 6)
        sizes = _text_sizes(line_height * area.height(), rect.height())
        size = self._fit(rect, sizes)
        if size is None:
            size = sizes[-1]
            needed = self._height_for(rect.width(), size)
            if mode == "cover":
                rect.setTop(rect.bottom() - needed)
            else:
                rect.setHeight(needed)
        self._font_size = size
        self.setGeometry(rect)
        self.show()
        self.raise_()
        self.update()

    def _font(self, size: int) -> QFont:
        font = _pixel_font(self.font(), size)
        font.setWeight(QFont.DemiBold)
        return font

    def _height_for(self, width: int, size: int) -> int:
        bounds = QFontMetrics(self._font(size)).boundingRect(QRect(0, 0, width - 24, 10_000), Qt.TextWordWrap, self.text)
        return bounds.height() + 16

    def _fit(self, rect: QRect, sizes: list[int]) -> int | None:
        return next((size for size in sizes if self._height_for(rect.width(), size) <= rect.height()), None)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        size = getattr(self, "_font_size", 20)
        painter.setFont(self._font(size))
        rect = self.rect().adjusted(12, 8, -12, -8)
        flags = Qt.AlignCenter | Qt.TextWordWrap
        if self.box:
            painter.setBrush(QColor(14, 20, 32, int(255 * self.opacity)))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(self.rect(), 6, 6)
        else:
            painter.setPen(QColor(0, 0, 0, 230 if self.final else 160))
            reach = max(1, round(size / 12))
            for dx in range(-reach, reach + 1):
                for dy in range(-reach, reach + 1):
                    if dx or dy:
                        painter.drawText(rect.translated(dx, dy), flags, self.text)
        painter.setPen(QColor(255, 255, 255, 255 if self.final else 190))
        painter.drawText(rect, flags, self.text)


class ScreenOverlay(QWidget):
    """Whole-screen mode: one click-through layer that repaints each translated block over its original text."""

    def __init__(self) -> None:
        super().__init__(None, window_flags() | Qt.WindowTransparentForInput | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.blocks: list[dict] = []
        self.translations: dict[str, tuple[str, bool]] = {}
        self.skip = QRect()
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
        sizes = _text_sizes(block["line_height"] * self.height(), rect.height())
        flags = Qt.TextWordWrap
        for size in sizes:
            needed = QFontMetrics(_pixel_font(self.font(), size)).boundingRect(
                rect.toRect().adjusted(2, 0, -2, 0), flags, text)
            if needed.height() <= rect.height() and needed.width() <= rect.width() - 4:
                return rect, size
        size = sizes[-1]
        needed = QFontMetrics(_pixel_font(self.font(), size)).boundingRect(
            QRect(0, 0, int(rect.width()) - 4, 10_000), flags, text)
        rect.setHeight(max(rect.height(), needed.height() + 4))
        return rect, size

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        skip = QRectF(self.skip.translated(-self.geometry().topLeft()))
        for i, block in enumerate(self.blocks):
            rect = self._rect(block)
            if not skip.isEmpty() and skip.intersects(rect):
                continue
            if i not in self._layouts:
                painter.setPen(QPen(QColor(46, 204, 113, 140), 1, Qt.DashLine))
                painter.setBrush(Qt.NoBrush)
                painter.drawRoundedRect(rect, 3, 3)
                continue
            rect, size = self._layouts[i]
            text, final = self.translations[block["text"]]
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(block["bg"]))
            painter.drawRoundedRect(rect, 3, 3)
            ink = QColor(block["fg"])
            ink.setAlpha(255 if final else 170)
            painter.setPen(ink)
            painter.setFont(_pixel_font(self.font(), size))
            painter.drawText(rect.adjusted(2, 0, -2, 0), Qt.AlignLeft | Qt.AlignVCenter | Qt.TextWordWrap, text)


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
        painter.setFont(_pixel_font(self.font(), 40))
        painter.drawText(badge, Qt.AlignCenter, self.label)


def _text_sizes(original_px: float, box_px: int) -> list[int]:
    """Pixel sizes to try, largest first: start from the original text's size and shrink at most to 60% of it."""
    start = round(original_px * 0.85) if original_px >= 6 else min(40, max(16, round(box_px * 0.55)))
    start = max(9, min(96, start))
    return list(range(start, max(8, round(start * 0.6)) - 1, -1))


def _pixel_font(base: QFont, size: int) -> QFont:
    font = QFont(base.family())
    font.setPixelSize(size)
    return font


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

        self.scope = QComboBox()
        self.scope.addItem("A selected area (subtitles, dialogue)", "area")
        self.scope.addItem("The whole screen", "screen")
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
        self.box = QCheckBox("Box behind text")
        self.box.setToolTip("Off: draw only the outlined translation, without the dark box.")
        self.mode = QComboBox()
        self.mode.addItem("Cover the original text", "cover")
        self.mode.addItem("Show below the original text", "below")

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
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.last = QLabel()
        self.last.setWordWrap(True)
        self.last.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.mode_row = QWidget()
        mode_layout = QHBoxLayout(self.mode_row)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.addWidget(self.mode, 1)
        mode_layout.addWidget(self.box)
        self.screen_label = QLabel("Screen")
        self.mode_label = QLabel("Overlay")
        form = QGridLayout()
        rows = ((QLabel("Translate"), self.scope), (self.screen_label, self.screen_pick),
                (QLabel("From"), self.source_lang), (QLabel("To"), self.target_lang),
                (QLabel("Model"), self.model), (self.mode_label, self.mode_row), (QLabel("Watching"), self.notes))
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
        layout.addLayout(form)
        layout.addLayout(download_row)
        layout.addLayout(tools)
        layout.addWidget(self.start)
        layout.addWidget(self.status)
        layout.addWidget(self.last)

        self.select.toggled.connect(self.toggle_source)
        self.add_area.clicked.connect(self.new_area)
        self.start.clicked.connect(self.toggle_running)
        self.forget.clicked.connect(lambda: self.api.call("/api/context/clear", {}))
        self.more.clicked.connect(lambda: webbrowser.open(f"{self.api.base_url}/panel"))
        self.download.clicked.connect(self.toggle_download)
        for combo, key in ((self.source_lang, "source_lang"), (self.target_lang, "target_lang"),
                           (self.model, "model"), (self.mode, "overlay_mode"), (self.scope, "scope"),
                           (self.screen_pick, "screen_name")):
            combo.currentIndexChanged.connect(lambda _i, c=combo, k=key: self.api.call("/api/settings", {k: c.currentData()}))
        self.scope.currentIndexChanged.connect(lambda _i: self.show_scope())
        self.box.toggled.connect(self.set_box)
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
                             (self.model, s.model), (self.mode, s.overlay_mode), (self.scope, s.scope),
                             (self.screen_pick, s.screen_name)):
            combo.blockSignals(True)
            combo.setCurrentIndex(max(0, combo.findData(value)))
            combo.blockSignals(False)
        self.notes.setText(s.notes)
        self.box.blockSignals(True)
        self.box.setChecked(s.overlay_box)
        self.box.blockSignals(False)
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

    def show_scope(self) -> None:
        screen = self.whole_screen()
        for widget in (self.screen_label, self.screen_pick):
            widget.setVisible(screen and self.screen_pick.count() > 1)
        for widget in (self.mode_label, self.mode_row, self.select):
            widget.setVisible(not screen)
        self.add_area.setVisible(not screen and self.select.isChecked())
        if screen:
            self.select.setChecked(False)
        if not self.running:
            self.set_running(False)

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
            self.api.call("/api/settings", {"screen_rect": native_rect(screen.geometry())})
            self.screen_overlay.cover(screen.geometry())

    def chosen_screen(self):
        name = self.screen_pick.currentData()
        return next((s for s in QGuiApplication.screens() if s.name() == name), None) or QGuiApplication.primaryScreen()

    def refresh_model(self) -> None:
        state = self.api.call("/api/state")
        model = next((m for m in state.get("models", []) if m["key"] == self.model.currentData()), None)
        download = state.get("download") or {}
        active = bool(download.get("active"))
        missing = bool(model) and not model["downloaded"] and not self.store.settings.custom_model_path
        self.download.setVisible(missing or active)
        self.progress.setVisible(active)
        self.download.setText("Cancel download" if active else f"Download model ({model['size'] / 1e9:.1f} GB)" if model else "")

    def toggle_download(self) -> None:
        state = self.api.call("/api/state")
        if (state.get("download") or {}).get("active"):
            self.api.call("/api/models/download/cancel", {})
        else:
            self.api.call(f"/api/models/{self.model.currentData()}/download", {})

    def add_source(self, area: QRect) -> SourceFrame:
        frame = SourceFrame()
        frame.set_area(area)
        frame.moved.connect(self.save_areas)
        frame.removed.connect(lambda f=frame: self.remove_source(f))
        self.sources.append(frame)
        self.overlays.append(TranslationOverlay())
        self.overlays[-1].opacity = self.store.settings.overlay_opacity
        self.overlays[-1].box = self.store.settings.overlay_box
        self.number_sources()
        return frame

    def set_box(self, on: bool) -> None:
        for overlay in self.overlays:
            overlay.box = on
            overlay.update()
        self.api.call("/api/settings", {"overlay_box": on})

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

    def toggle_running(self) -> None:
        if self.running:
            self.api.call("/api/stop", {})
            self.set_running(False)
            return
        self.select.setChecked(False)
        if self.whole_screen():
            geometry = self.chosen_screen().geometry()
            self.api.call("/api/settings", {"screen_rect": native_rect(geometry)})
            self.screen_overlay.skip = self.frameGeometry()
            self.screen_overlay.cover(geometry)
        else:
            self.save_areas()
            for overlay in self.overlays:
                overlay.ensure_native()
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
        if not running:
            for overlay in self.overlays:
                overlay.hide()
            self.screen_overlay.hide()
            if self.whole_screen():
                self.status.setText("Start (Ctrl+Enter) to translate every piece of text on the screen in place.")
            else:
                self.status.setText("Click “Select areas”, drag the green box over the text (add more boxes if the "
                                    "text is in several places), then Start (Ctrl+Enter).")

    def on_event(self, event: dict) -> None:
        kind = event["type"]
        if kind == "line" and self.running:
            area = event.get("area", 0)
            if area < len(self.overlays):
                self.overlays[area].show_text(event["translation"], event["final"], self.sources[area].area(),
                                              self.store.settings.overlay_mode, event.get("line_height", 0.0))
            if event["final"]:
                self.last.setText(f"<span style='color:gray'>{event['source']}</span><br>{event['translation']}"
                                  f"<br><span style='color:gray;font-size:11px'>{event['ms']} ms · "
                                  f"{event['context_lines']} lines of context</span>")
        elif kind == "screen" and self.running:
            self.screen_overlay.skip = self.frameGeometry()
            self.screen_overlay.set_blocks(event["blocks"])
        elif kind == "screen_text" and self.running:
            self.screen_overlay.set_translation(event["source"], event["translation"], event["final"])
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
            elif event["state"] == "running" and self.whole_screen():
                self.status.setText("Translating the whole screen. Text under this window is left alone.")
            elif event["state"] == "running":
                self.status.setText("Translating. The box over the text shows the translation.")
        elif kind == "download":
            if event.get("active"):
                self.progress.setMaximum(1000)
                self.progress.setValue(int(1000 * event["done"] / max(1, event["total"])))
            QTimer.singleShot(0, self.refresh_model)

    def closeEvent(self, event) -> None:
        self.api.call("/api/stop", {})
        for widget in [*self.sources, *self.overlays, self.screen_overlay, self.flash]:
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
    painter.setFont(_pixel_font(QFont(), 22))
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
    show.triggered.connect(lambda: bring_to_front(window))
    toggle.triggered.connect(window.toggle_running)
    quit_app.triggered.connect(window.close)
    menu.aboutToShow.connect(lambda: toggle.setText("Stop" if window.running else "Start"))
    for action in (show, toggle, quit_app):
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
    window.show()
    bring_to_front(window)
    app.exec()
    tray.hide()
    stop.set()
    server.join(10)
