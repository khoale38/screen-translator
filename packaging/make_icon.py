"""Draws the app icon and writes it as PNG (window icon), ICNS (macOS) and ICO (Windows).

Run on macOS so the 文 glyph comes from a CJK system font: .venv/bin/python packaging/make_icon.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QLinearGradient, QPainter, QPainterPath

ROOT = Path(__file__).resolve().parent.parent
SIZE = 1024


def tile(painter: QPainter, rect: QRectF, fill: QColor, ink: QColor, glyph: str, family: str) -> None:
    shadow = QPainterPath()
    shadow.addRoundedRect(rect.translated(0, 14), 72, 72)
    painter.fillPath(shadow, QColor(0, 0, 0, 70))
    path = QPainterPath()
    path.addRoundedRect(rect, 72, 72)
    painter.fillPath(path, fill)
    font = QFont(family)
    font.setPixelSize(int(rect.height() * 0.68))
    font.setWeight(QFont.Bold)
    painter.setFont(font)
    painter.setPen(ink)
    painter.drawText(rect.adjusted(0, -8, 0, 0), Qt.AlignCenter, glyph)


def draw(size: int = SIZE, full_bleed: bool = False) -> QImage:
    image = QImage(size, size, QImage.Format_ARGB32_Premultiplied)
    image.fill(Qt.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setRenderHint(QPainter.TextAntialiasing)
    painter.scale(size / SIZE, size / SIZE)
    # macOS icons sit on an 824 px plate inside the 1024 canvas; Windows icons use the whole square.
    plate = QRectF(0, 0, SIZE, SIZE) if full_bleed else QRectF(100, 100, 824, 824)
    gradient = QLinearGradient(plate.topLeft(), plate.bottomRight())
    gradient.setColorAt(0, QColor("#3B82F6"))
    gradient.setColorAt(1, QColor("#1E3A8A"))
    background = QPainterPath()
    background.addRoundedRect(plate, plate.width() * 0.225, plate.width() * 0.225)
    painter.fillPath(background, gradient)
    scale = plate.width() / 824
    def at(x: float, y: float, w: float) -> QRectF:
        return QRectF(plate.left() + x * scale, plate.top() + y * scale, w * scale, w * scale)
    cjk = {"darwin": "PingFang SC", "win32": "Microsoft YaHei"}.get(sys.platform, "Noto Sans CJK SC")
    tile(painter, at(118, 118, 380), QColor("#F8FAFC"), QColor("#1E3A8A"), "文", cjk)
    tile(painter, at(326, 326, 380), QColor("#F59E0B"), QColor("#FFFFFF"), "A", "Helvetica Neue")
    painter.end()
    return image


def to_pil(image: QImage) -> Image.Image:
    image = image.convertToFormat(QImage.Format_RGBA8888)
    return Image.frombytes("RGBA", (image.width(), image.height()), bytes(image.constBits()))


def main() -> None:
    QGuiApplication.instance() or QGuiApplication([])
    out = ROOT / "packaging" / "icons"
    out.mkdir(exist_ok=True)
    mac = to_pil(draw())
    mac.save(out / "icon.icns", sizes=[(16, 16), (32, 32), (64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)])
    windows = to_pil(draw(full_bleed=True))
    windows.save(out / "icon.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    windows.resize((256, 256), Image.LANCZOS).save(ROOT / "src" / "screen_translator" / "web" / "icon.png")
    print(f"Wrote icons to {out}")


if __name__ == "__main__":
    main()
