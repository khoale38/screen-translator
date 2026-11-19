from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
import webbrowser

from PySide6.QtCore import QObject, QPoint, QRect, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QGuiApplication, QKeySequence, QPainter, QPen, QShortcut
from PySide6.QtWidgets import (
    QApplication, QComboBox, QGridLayout, QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

from . import languages
from .capture import exclude_window_from_capture
from .config import SettingsStore
from .models import MODELS
from .server import serve

GREEN = QColor(46, 204, 113)
HEADER = 26
EDGE = 10


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
    if sys.platform == "darwin":
        import objc

        window = objc.objc_object(c_void_p=int(widget.winId())).window()
        window.setLevel_(25)
        window.setCollectionBehavior_((1 << 0) | (1 << 4) | (1 << 8))
        window.setHidesOnDeactivate_(False)
        return int(window.windowNumber())
    return 0


def hide_from_screen_capture(widget: QWidget) -> None:
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.user32.SetWindowDisplayAffinity(int(widget.winId()), 0x11)


def window_flags() -> Qt.WindowType:
    flags = Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
    # A Tool window keeps the frames out of the Windows taskbar; on macOS it would vanish when the app loses focus.
    return flags | (Qt.Tool if sys.platform == "win32" else Qt.Window)


class SourceFrame(QWidget):
    """The green box you drag over the text. The header strip sits above the captured area so it is never OCR'd."""

    moved = Signal(QRect)

    def __init__(self) -> None:
        super().__init__(None, window_flags())
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setMinimumSize(80, HEADER + 30)
        self.setMouseTracking(True)
        self._drag: tuple[str, QPoint, QRect] | None = None

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
        tab = QRect(0, 0, min(self.width(), 300), HEADER - 2)
        painter.fillRect(tab, GREEN)
        painter.setPen(Qt.white)
        painter.setFont(QFont(self.font().family(), 11, QFont.DemiBold))
        painter.drawText(tab.adjusted(8, 0, 0, 0), Qt.AlignVCenter, "Source · drag over the text, resize from edges")

    def _zone(self, pos: QPoint) -> str:
        zone = ""
        if pos.y() >= self.height() - EDGE:
            zone += "b"
        elif pos.y() <= EDGE and pos.x() > 300:
            zone += "t"
        if pos.x() <= EDGE:
            zone += "l"
        elif pos.x() >= self.width() - EDGE:
            zone += "r"
        return zone or "move"

    def mousePressEvent(self, event) -> None:
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
        self.anchor = QRect()
        self._native_ready = False

    def ensure_native(self) -> None:
        if self._native_ready:
            return
        self.setGeometry(QRect(-10000, -10000, 10, 10))
        self.show()
        exclude_window_from_capture(float_above_everything(self))
        hide_from_screen_capture(self)
        self.hide()
        self._native_ready = True

    def show_text(self, text: str, final: bool, area: QRect, mode: str) -> None:
        self.text, self.final = text, final
        if not text:
            return
        rect = QRect(area)
        if mode == "below":
            rect.moveTop(area.bottom() + 6)
        size = self._fit(rect)
        if size is None:
            needed = self._height_for(rect.width(), 16)
            if mode == "cover":
                rect.setTop(rect.bottom() - needed)
            else:
                rect.setHeight(needed)
            size = 16
        self._font_size = size
        self.setGeometry(rect)
        self.show()
        self.raise_()
        self.update()

    def _font(self, size: int) -> QFont:
        font = QFont(self.font().family(), size)
        font.setWeight(QFont.DemiBold)
        return font

    def _height_for(self, width: int, size: int) -> int:
        bounds = QFontMetrics(self._font(size)).boundingRect(QRect(0, 0, width - 24, 10_000), Qt.TextWordWrap, self.text)
        return bounds.height() + 16

    def _fit(self, rect: QRect) -> int | None:
        for size in range(min(40, max(16, int(rect.height() * 0.55))), 15, -1):
            if self._height_for(rect.width(), size) <= rect.height():
                return size
        return None

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor(14, 20, 32, int(255 * self.opacity)))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(self.rect(), 6, 6)
        painter.setFont(self._font(getattr(self, "_font_size", 20)))
        painter.setPen(QColor(255, 255, 255, 255 if self.final else 190))
        painter.drawText(self.rect().adjusted(12, 8, -12, -8), Qt.AlignCenter | Qt.TextWordWrap, self.text)


class MainWindow(QWidget):
    def __init__(self, api: Api, store: SettingsStore, bridge: Bridge) -> None:
        super().__init__()
        self.api, self.store = api, store
        self.running = False
        self.setWindowTitle("Screen Translator")
        self.setMinimumWidth(460)

        self.source = SourceFrame()
        self.overlay = TranslationOverlay()
        self.source.moved.connect(self.save_area)
        bridge.event.connect(self.on_event)

        self.source_lang = QComboBox()
        self.target_lang = QComboBox()
        for code, label in languages.LANGUAGES.items():
            self.source_lang.addItem(label, code)
            if code != "auto":
                self.target_lang.addItem(label, code)
        self.model = QComboBox()
        for spec in MODELS.values():
            self.model.addItem(spec.label, spec.key)
        self.mode = QComboBox()
        self.mode.addItem("Cover the original text", "cover")
        self.mode.addItem("Show below the original text", "below")

        self.download = QPushButton("Download model")
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setMaximumHeight(8)
        self.select = QPushButton("Select area")
        self.select.setCheckable(True)
        self.forget = QPushButton("Forget context")
        self.more = QPushButton("Glossary && more…")
        self.start = QPushButton()
        self.start.setMinimumHeight(52)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.last = QLabel()
        self.last.setWordWrap(True)
        self.last.setTextInteractionFlags(Qt.TextSelectableByMouse)

        form = QGridLayout()
        form.addWidget(QLabel("From"), 0, 0)
        form.addWidget(self.source_lang, 0, 1)
        form.addWidget(QLabel("To"), 1, 0)
        form.addWidget(self.target_lang, 1, 1)
        form.addWidget(QLabel("Model"), 2, 0)
        form.addWidget(self.model, 2, 1)
        form.addWidget(QLabel("Overlay"), 3, 0)
        form.addWidget(self.mode, 3, 1)
        form.setColumnStretch(1, 1)
        download_row = QHBoxLayout()
        download_row.addWidget(self.download)
        download_row.addWidget(self.progress, 1)
        tools = QHBoxLayout()
        for button in (self.select, self.forget, self.more):
            tools.addWidget(button)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(download_row)
        layout.addLayout(tools)
        layout.addWidget(self.start)
        layout.addWidget(self.status)
        layout.addWidget(self.last)

        self.select.toggled.connect(self.toggle_source)
        self.start.clicked.connect(self.toggle_running)
        self.forget.clicked.connect(lambda: self.api.call("/api/context/clear", {}))
        self.more.clicked.connect(lambda: webbrowser.open(f"{self.api.base_url}/panel"))
        self.download.clicked.connect(self.toggle_download)
        for combo, key in ((self.source_lang, "source_lang"), (self.target_lang, "target_lang"),
                           (self.model, "model"), (self.mode, "overlay_mode")):
            combo.currentIndexChanged.connect(lambda _i, c=combo, k=key: self.api.call("/api/settings", {k: c.currentData()}))
        QShortcut(QKeySequence("Ctrl+Return"), self, self.toggle_running)

        self.load()

    def load(self) -> None:
        s = self.store.settings
        for combo, value in ((self.source_lang, s.source_lang), (self.target_lang, s.target_lang),
                             (self.model, s.model), (self.mode, s.overlay_mode)):
            combo.blockSignals(True)
            combo.setCurrentIndex(max(0, combo.findData(value)))
            combo.blockSignals(False)
        self.overlay.opacity = s.overlay_opacity
        if s.area:
            self.source.set_area(QRect(*s.area))
        else:
            screen = QGuiApplication.primaryScreen().availableGeometry()
            self.source.set_area(QRect(screen.center().x() - 400, screen.bottom() - 220, 800, 120))
        self.refresh_model()
        self.set_running(False)

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

    def toggle_source(self, visible: bool) -> None:
        self.source.setVisible(visible)
        if visible:
            float_above_everything(self.source)
            self.source.raise_()
        self.select.setText("Done" if visible else "Select area")

    def save_area(self, area: QRect) -> None:
        self.api.call("/api/settings", {"area": [area.x(), area.y(), area.width(), area.height()],
                                        "capture_rect": native_rect(area)})

    def toggle_running(self) -> None:
        if self.running:
            self.api.call("/api/stop", {})
            self.set_running(False)
            return
        self.select.setChecked(False)
        self.save_area(self.source.area())
        self.overlay.ensure_native()
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
        if not running:
            self.overlay.hide()
            self.status.setText("Click “Select area”, drag the green box over the text, then Start (Ctrl+Enter).")

    def on_event(self, event: dict) -> None:
        kind = event["type"]
        if kind == "line" and self.running:
            self.overlay.show_text(event["translation"], event["final"], self.source.area(),
                                   self.store.settings.overlay_mode)
            if event["final"]:
                self.last.setText(f"<span style='color:gray'>{event['source']}</span><br>{event['translation']}"
                                  f"<br><span style='color:gray;font-size:11px'>{event['ms']} ms · "
                                  f"{event['context_lines']} lines of context</span>")
        elif kind == "clear":
            self.overlay.hide()
        elif kind == "status":
            if event["state"] == "error":
                self.set_running(False)
                self.status.setText(event["error"])
            elif event["state"] == "loading":
                self.status.setText("Loading the model…")
            elif event["state"] == "running":
                self.status.setText("Translating. The box over the text shows the translation.")
        elif kind == "download":
            if event.get("active"):
                self.progress.setMaximum(1000)
                self.progress.setValue(int(1000 * event["done"] / max(1, event["total"])))
            QTimer.singleShot(0, self.refresh_model)

    def closeEvent(self, event) -> None:
        self.api.call("/api/stop", {})
        self.source.close()
        self.overlay.close()
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


def run(store: SettingsStore) -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Screen Translator")
    bridge = Bridge()
    ready, stop = threading.Event(), threading.Event()
    server = threading.Thread(target=serve, args=(store, ready, stop, bridge.event.emit), name="server", daemon=True)
    server.start()
    if not ready.wait(15):
        raise SystemExit(f"Could not start the local server on port {store.settings.port}; is it already in use?")

    window = MainWindow(Api(f"http://127.0.0.1:{store.settings.port}"), store, bridge)
    window.show()
    app.exec()
    stop.set()
    server.join(10)
