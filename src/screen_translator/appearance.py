"""How a translation looks on screen: shared by the real overlays and the live preview in the settings window."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QIcon, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton, QSlider,
    QToolButton, QWidget,
)

PAD_X = 10
PAD_Y = 4
BOX_COLOR = (14, 20, 32)

SAMPLES = {
    "vi": "Cắt đuôi cảnh sát.", "en": "Lose the cops.", "ja": "警察をまけ。", "zh": "甩掉警察。", "zh-Hant": "甩掉警察。",
    "ko": "경찰을 따돌려.", "fr": "Sème les flics.", "de": "Häng die Cops ab.", "es": "Despista a la policía.",
    "pt": "Despiste a polícia.", "it": "Semina la polizia.", "nl": "Schud de politie af.", "pl": "Zgub policję.",
    "tr": "Polisleri ekte.", "ru": "Оторвись от копов.", "uk": "Відірвися від копів.", "id": "Lolos dari polisi.",
    "ms": "Lepaskan diri daripada polis.", "th": "หนีตำรวจให้พ้น", "ar": "تخلّص من الشرطة.", "hi": "पुलिस से पीछा छुड़ाओ।",
}
SETTINGS_WORD = {
    "vi": "Cài đặt", "en": "Settings", "ja": "設定", "zh": "设置", "zh-Hant": "設定", "ko": "설정", "fr": "Paramètres",
    "de": "Einstellungen", "es": "Ajustes", "pt": "Configurações", "it": "Impostazioni", "ru": "Настройки",
}


@dataclass
class Style:
    box: bool = True
    opacity: float = 0.92
    mode: str = "cover"
    text_size: str = "x1"


def text_sizes(original_px: float, box_px: int, setting: str = "x1") -> list[int]:
    """Pixel sizes to try, largest first. "pxN" is a fixed size; "xS" scales the original text's size by S."""
    if setting.startswith("px"):
        return [int(setting[2:])]
    scale = float(setting[1:]) if setting.startswith("x") else 1.0
    base = original_px * 0.85 if original_px >= 6 else min(40, max(16, box_px * 0.55))
    start = max(9, min(120, round(base * scale)))
    return list(range(start, max(8, round(start * 0.6)) - 1, -1))


def pixel_font(size: int, bold: bool = False) -> QFont:
    font = QFont(QApplication.font().family())
    font.setPixelSize(size)
    if bold:
        font.setWeight(QFont.DemiBold)
    return font


def _wrapped(text: str, width: int, size: int, flags=Qt.TextWordWrap) -> QRect:
    return QFontMetrics(pixel_font(size, bold=True)).boundingRect(QRect(0, 0, max(1, width - 2 * PAD_X), 10_000),
                                                                  flags, text)


def place_translation(text: str, area: QRect, style: Style, line_height: float = 0.0) -> tuple[QRect, int]:
    """Where an area translation goes and its pixel size: sized like the original, fitted, and hugged by its box."""
    rect = QRect(area)
    if style.mode == "below":
        rect.moveTop(area.bottom() + 6)
    sizes = text_sizes(line_height * area.height(), rect.height(), style.text_size)
    size = next((s for s in sizes if _wrapped(text, rect.width(), s).height() + 2 * PAD_Y <= rect.height()), None)
    if size is None:
        size = sizes[-1]
        needed = _wrapped(text, rect.width(), size).height() + 2 * PAD_Y
        if style.mode == "cover":
            rect.setTop(rect.bottom() - needed)
        else:
            rect.setHeight(needed)
    if style.box:
        bounds = _wrapped(text, rect.width(), size, Qt.AlignCenter | Qt.TextWordWrap)
        snug = QRect(0, 0, min(rect.width(), bounds.width() + 2 * PAD_X),
                     min(rect.height(), bounds.height() + 2 * PAD_Y))
        snug.moveCenter(rect.center())
        if style.mode == "below":
            snug.moveTop(rect.top())
        rect = snug
    return rect, size


def paint_translation(painter: QPainter, rect: QRect, text: str, size: int, style: Style, final: bool = True) -> None:
    painter.save()
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setFont(pixel_font(size, bold=True))
    inner = rect.adjusted(PAD_X, PAD_Y, -PAD_X, -PAD_Y)
    flags = Qt.AlignCenter | Qt.TextWordWrap
    if style.box:
        painter.setBrush(QColor(*BOX_COLOR, int(255 * style.opacity)))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(rect, 6, 6)
    else:
        painter.setPen(QColor(0, 0, 0, 230 if final else 160))
        reach = max(1, round(size / 12))
        for dx in range(-reach, reach + 1):
            for dy in range(-reach, reach + 1):
                if dx or dy:
                    painter.drawText(inner.translated(dx, dy), flags, text)
    painter.setPen(QColor(255, 255, 255, 255 if final else 190))
    painter.drawText(inner, flags, text)
    painter.restore()


def paint_patch(painter: QPainter, rect: QRectF, text: str, size: int, bg: str, fg: str, final: bool = True) -> None:
    """Whole-screen style: the original's own colours, so the translation looks printed in place."""
    painter.save()
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(bg))
    painter.drawRoundedRect(rect, 3, 3)
    ink = QColor(fg)
    ink.setAlpha(255 if final else 170)
    painter.setPen(ink)
    painter.setFont(pixel_font(size))
    painter.drawText(rect.adjusted(2, 0, -2, 0), Qt.AlignLeft | Qt.AlignVCenter | Qt.TextWordWrap, text)
    painter.restore()


def _scene_background(painter: QPainter, rect: QRect) -> None:
    sky = QLinearGradient(QPointF(rect.topLeft()), QPointF(rect.bottomLeft()))
    sky.setColorAt(0, QColor(70, 92, 130))
    sky.setColorAt(0.55, QColor(196, 140, 110))
    sky.setColorAt(1, QColor(40, 36, 44))
    painter.fillRect(rect, sky)
    skyline = QPainterPath()
    base = rect.top() + rect.height() * 0.62
    skyline.moveTo(rect.left(), base)
    x = rect.left()
    for width, height in ((0.08, 0.18), (0.06, 0.3), (0.1, 0.12), (0.05, 0.36), (0.09, 0.2), (0.07, 0.27),
                          (0.11, 0.15), (0.06, 0.33), (0.08, 0.1), (0.1, 0.24), (0.2, 0.14)):
        top = base - rect.height() * height
        skyline.lineTo(x, top)
        x += rect.width() * width
        skyline.lineTo(x, top)
    skyline.lineTo(rect.right(), base)
    skyline.lineTo(rect.right(), rect.bottom())
    skyline.lineTo(rect.left(), rect.bottom())
    painter.fillPath(skyline, QColor(28, 30, 40))


def paint_area_scene(painter: QPainter, rect: QRect, style: Style, target: str, outline_area: bool = True) -> None:
    """A film frame with an English subtitle, its selected area, and the translation drawn as the overlay would."""
    _scene_background(painter, rect)
    source = "警察をまけ！" if target == "en" else "Lose the cops."
    original_px = max(10, round(rect.height() * 0.11))
    original = QRect(rect.left(), rect.bottom() - round(rect.height() * 0.42), rect.width(), round(original_px * 1.6))
    painter.save()
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setFont(pixel_font(original_px, bold=True))
    painter.setPen(QColor(0, 0, 0, 200))
    painter.drawText(original.translated(1, 1), Qt.AlignCenter, source)
    painter.setPen(QColor(250, 240, 120))
    painter.drawText(original, Qt.AlignCenter, source)
    area = original.adjusted(round(rect.width() * 0.2), -6, -round(rect.width() * 0.2), 6)
    if outline_area:
        painter.setPen(QPen(QColor(46, 204, 113), 1.5, Qt.DashLine))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(area)
    painter.restore()
    text = SAMPLES.get(target, "Lose the cops.")
    placed, size = place_translation(text, area, style, original_px / area.height())
    paint_translation(painter, placed, text, size, style)


def paint_screen_scene(painter: QPainter, rect: QRect, style: Style, target: str) -> None:
    """A game screen whose labels are repainted in their own colours, as whole-screen mode does."""
    _scene_background(painter, rect)
    tip = QRectF(rect.left() + rect.width() * 0.05, rect.top() + rect.height() * 0.12,
                 rect.width() * 0.5, rect.height() * 0.2)
    button = QRectF(rect.left() + rect.width() * 0.62, rect.top() + rect.height() * 0.68,
                    rect.width() * 0.3, rect.height() * 0.16)
    line_px = rect.height() * 0.09
    for box, bg, fg, translated in (
        (tip, "#2b2f45", "#f2f2f2", SAMPLES.get(target, "Lose the cops.")),
        (button, "#e8e8e8", "#1b1b1b", SETTINGS_WORD.get(target, "Settings")),
    ):
        size = text_sizes(line_px, round(box.height()), style.text_size)[0]
        paint_patch(painter, box.adjusted(-3, -2, 3, 2), translated, size, bg, fg)


class OverlayPreview(QWidget):
    """Live sample of the current appearance settings, drawn with the same code as the real overlay."""

    def __init__(self) -> None:
        super().__init__()
        self.style = Style()
        self.scope = "area"
        self.target = "vi"
        self.setMinimumHeight(150)
        self.setToolTip("A preview of how translations will look. It updates as you change the settings below.")

    def sizeHint(self) -> QSize:
        return QSize(440, 160)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        frame = self.rect().adjusted(0, 0, -1, -1)
        clip = QPainterPath()
        clip.addRoundedRect(QRectF(frame), 8, 8)
        painter.setClipPath(clip)
        if self.scope == "screen":
            paint_screen_scene(painter, frame, self.style, self.target)
        else:
            paint_area_scene(painter, frame, self.style, self.target)
        painter.setClipping(False)
        painter.setPen(QPen(QColor(0, 0, 0, 60), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(QRectF(frame), 8, 8)


ICON_SIZE = QSize(156, 54)


def style_icon(box: bool, target: str, size: QSize = ICON_SIZE) -> QIcon:
    pixmap = QPixmap(size * 2)
    pixmap.setDevicePixelRatio(2)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    rect = QRect(0, 0, size.width(), size.height())
    clip = QPainterPath()
    clip.addRoundedRect(QRectF(rect), 6, 6)
    painter.setClipPath(clip)
    _scene_background(painter, rect)
    text = SAMPLES.get(target, "Lose the cops.")
    style = Style(box=box, text_size="px12")
    placed, font_size = place_translation(text, rect.adjusted(4, 14, -4, -8), style)
    paint_translation(painter, placed, text, font_size, style)
    painter.end()
    return QIcon(pixmap)


class AppearancePanel(QGroupBox):
    """Visual controls for how translations look, with a live preview; emits settings patches."""

    changed = Signal(dict)

    def __init__(self) -> None:
        super().__init__("Appearance")
        self.preview = OverlayPreview()
        self.style_box = self._choice("Dark box")
        self.style_outline = self._choice("Outlined text")
        self.cover = self._segment("Over the original")
        self.below = self._segment("Below it")
        self.opacity = QSlider(Qt.Horizontal)
        self.opacity.setRange(40, 100)
        self.opacity.setSingleStep(5)
        self.opacity_value = QLabel()
        self.size = QSlider(Qt.Horizontal)
        self.size.setTickPosition(QSlider.TicksBelow)
        self.size_value = QLabel()
        self.size_value.setMinimumWidth(130)
        self.fixed = QCheckBox("Fixed size")
        self.fixed.setToolTip("Off: translations follow the size of the original text. On: always this many pixels.")

        styles = QButtonGroup(self)
        positions = QButtonGroup(self)
        for group, buttons in ((styles, (self.style_box, self.style_outline)), (positions, (self.cover, self.below))):
            group.setExclusive(True)
            for button in buttons:
                group.addButton(button)

        self.style_label = QLabel("Style")
        self.position_label = QLabel("Position")
        self.opacity_label = QLabel("Box opacity")
        style_row = QHBoxLayout()
        style_row.addWidget(self.style_box)
        style_row.addWidget(self.style_outline)
        style_row.addStretch(1)
        position_row = QHBoxLayout()
        position_row.setSpacing(0)
        position_row.addWidget(self.cover)
        position_row.addWidget(self.below)
        position_row.addStretch(1)
        opacity_row = QHBoxLayout()
        opacity_row.addWidget(self.opacity, 1)
        opacity_row.addWidget(self.opacity_value)
        size_row = QHBoxLayout()
        size_row.addWidget(self.size, 1)
        size_row.addWidget(self.size_value)
        size_row.addWidget(self.fixed)

        grid = QGridLayout(self)
        grid.addWidget(self.preview, 0, 0, 1, 2)
        for row, (label, content) in enumerate(((self.style_label, style_row), (self.position_label, position_row),
                                                (self.opacity_label, opacity_row), (QLabel("Text size"), size_row)), 1):
            grid.addWidget(label, row, 0)
            grid.addLayout(content, row, 1)
        grid.setColumnStretch(1, 1)

        self.style_box.toggled.connect(lambda on: self._emit({"overlay_box": on}))
        self.cover.toggled.connect(lambda on: self._emit({"overlay_mode": "cover" if on else "below"}))
        self.opacity.valueChanged.connect(lambda value: self._emit({"overlay_opacity": value / 100}))
        self.size.valueChanged.connect(lambda _v: self._emit({"text_size": self.text_size()}))
        self.fixed.toggled.connect(self._switch_size_mode)
        self.set_values(Style())

    @staticmethod
    def _choice(label: str) -> QToolButton:
        button = QToolButton()
        button.setText(label)
        button.setCheckable(True)
        button.setIconSize(ICON_SIZE)
        button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        button.setStyleSheet("QToolButton { padding: 4px; border: 2px solid transparent; border-radius: 8px; }"
                             "QToolButton:checked { border-color: #2ea44f; }")
        return button

    @staticmethod
    def _segment(label: str) -> QPushButton:
        button = QPushButton(label)
        button.setCheckable(True)
        button.setStyleSheet("QPushButton { padding: 5px 12px; }"
                             "QPushButton:checked { background: #2ea44f; color: white; }")
        return button

    def style(self) -> Style:
        return Style(self.style_box.isChecked(), self.opacity.value() / 100,
                     "cover" if self.cover.isChecked() else "below", self.text_size())

    def text_size(self) -> str:
        value = self.size.value()
        return f"px{value}" if self.fixed.isChecked() else f"x{value / 100:g}"

    def set_values(self, style: Style) -> None:
        widgets = (self.style_box, self.style_outline, self.cover, self.below, self.opacity, self.size, self.fixed)
        for widget in widgets:
            widget.blockSignals(True)
        (self.style_box if style.box else self.style_outline).setChecked(True)
        (self.cover if style.mode == "cover" else self.below).setChecked(True)
        self.opacity.setValue(round(style.opacity * 100))
        fixed = style.text_size.startswith("px")
        self.fixed.setChecked(fixed)
        self._size_range(fixed)
        self.size.setValue(int(style.text_size[2:]) if fixed else round(float(style.text_size[1:] or 1) * 100))
        for widget in widgets:
            widget.blockSignals(False)
        self._refresh()

    def set_scope(self, scope: str) -> None:
        self.preview.scope = scope
        for widget in (self.style_label, self.style_box, self.style_outline, self.position_label, self.cover,
                       self.below):
            widget.setVisible(scope != "screen")
        self._refresh()

    def set_target(self, target: str) -> None:
        self.preview.target = target
        self.style_box.setIcon(style_icon(True, target))
        self.style_outline.setIcon(style_icon(False, target))
        self._refresh()

    def _size_range(self, fixed: bool) -> None:
        if fixed:
            self.size.setRange(12, 64)
            self.size.setSingleStep(2)
            self.size.setTickInterval(8)
        else:
            self.size.setRange(50, 200)
            self.size.setSingleStep(5)
            self.size.setTickInterval(25)

    def _switch_size_mode(self, fixed: bool) -> None:
        self.size.blockSignals(True)
        self._size_range(fixed)
        self.size.setValue(24 if fixed else 100)
        self.size.blockSignals(False)
        self._emit({"text_size": self.text_size()})

    def _emit(self, patch: dict) -> None:
        self._refresh()
        self.changed.emit(patch)

    def _refresh(self) -> None:
        style = self.style()
        self.preview.style = style
        boxed = style.box and self.preview.scope != "screen"
        for widget in (self.opacity_label, self.opacity, self.opacity_value):
            widget.setVisible(boxed)
        self.opacity_value.setText(f"{self.opacity.value()}%")
        value = self.size.value()
        if self.fixed.isChecked():
            self.size_value.setText(f"{value} px")
        else:
            self.size_value.setText("Match the original" if value == 100 else f"{value}% of the original")
        self.preview.update()
