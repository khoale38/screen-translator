"""How an area translation is sized, placed and drawn: shared by the overlays."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QRect, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter
from PySide6.QtWidgets import QApplication

PAD_X = 10
PAD_Y = 4
BOX_COLOR = (14, 20, 32)

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


def place_translation(text: str, area: QRect, style: Style, line_height: float = 0.0,
                      bounds: QRect | None = None) -> tuple[QRect, int]:
    """Where an area translation goes and its pixel size: sized like the original, fitted, and hugged by its box.

    In "below" mode it flips above the area when the screen (`bounds`) has no room underneath.
    """
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
        wrapped = _wrapped(text, rect.width(), size, Qt.AlignCenter | Qt.TextWordWrap)
        snug = QRect(0, 0, min(rect.width(), wrapped.width() + 2 * PAD_X),
                     min(rect.height(), wrapped.height() + 2 * PAD_Y))
        snug.moveCenter(rect.center())
        if style.mode == "below":
            snug.moveTop(rect.top())
        rect = snug
    if style.mode == "below" and bounds is not None and rect.bottom() > bounds.bottom():
        rect.moveBottom(area.top() - 6)
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


def paint_patch(painter: QPainter, rect: QRectF, text: str, size: int, bg: str, fg: str, final: bool = True,
                align=Qt.AlignLeft) -> None:
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
    painter.drawText(rect.adjusted(2, 0, -2, 0), align | Qt.AlignVCenter | Qt.TextWordWrap, text)
    painter.restore()
