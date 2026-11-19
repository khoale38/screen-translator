import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtCore = pytest.importorskip("PySide6.QtCore")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from screen_translator import desktop  # noqa: E402

QRect = QtCore.QRect


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_source_frame_area_excludes_its_label_strip(app):
    frame = desktop.SourceFrame()
    frame.set_area(QRect(200, 600, 700, 120))
    assert frame.area() == QRect(200, 600, 700, 120)
    assert frame.geometry().top() == 600 - desktop.HEADER


def test_overlay_covers_the_area_and_shrinks_text_to_fit(app):
    overlay = desktop.TranslationOverlay()
    area = QRect(100, 500, 800, 100)
    overlay.show_text("Tối qua bạn đã ở đâu vậy?", True, area, "cover")
    assert overlay.geometry() == area
    assert 16 <= overlay._font_size <= 40


def test_overlay_grows_upwards_when_the_translation_is_too_long(app):
    overlay = desktop.TranslationOverlay()
    area = QRect(100, 500, 300, 40)
    overlay.show_text("A much longer translation that cannot possibly fit on one short line " * 3, True, area, "cover")
    assert overlay.geometry().bottom() == area.bottom()
    assert overlay.geometry().height() > area.height()


def test_overlay_below_mode_sits_under_the_text(app):
    overlay = desktop.TranslationOverlay()
    area = QRect(100, 500, 800, 100)
    overlay.show_text("Hello", True, area, "below")
    assert overlay.geometry().top() > area.bottom()


def test_native_rect_matches_logical_rect_at_scale_one(app):
    assert desktop.native_rect(QRect(10, 20, 300, 40)) == {"left": 10, "top": 20, "width": 300, "height": 40}
