import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtCore = pytest.importorskip("PySide6.QtCore")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from screen_translator import desktop  # noqa: E402
from screen_translator.appearance import Style, text_sizes  # noqa: E402

QRect = QtCore.QRect


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_source_frame_area_excludes_its_label_strip(app):
    frame = desktop.SourceFrame()
    frame.set_area(QRect(200, 600, 700, 120))
    assert frame.area() == QRect(200, 600, 700, 120)
    assert frame.geometry().top() == 600 - desktop.HEADER


def test_overlay_stays_inside_the_area_and_shrinks_text_to_fit(app):
    overlay = desktop.TranslationOverlay()
    area = QRect(100, 500, 800, 100)
    overlay.show_text("Tối qua bạn đã ở đâu vậy?", True, area)
    assert area.contains(overlay.geometry())
    assert 16 <= overlay._font_size <= 40


def test_overlay_grows_upwards_when_the_translation_is_too_long(app):
    overlay = desktop.TranslationOverlay()
    area = QRect(100, 500, 300, 40)
    overlay.show_text("A much longer translation that cannot possibly fit on one short line " * 3, True, area)
    assert overlay.geometry().bottom() == area.bottom()
    assert overlay.geometry().height() > area.height()


def test_overlay_below_mode_sits_under_the_text(app):
    overlay = desktop.TranslationOverlay()
    overlay.style = Style(mode="below")
    area = QRect(100, 500, 800, 100)
    overlay.show_text("Hello", True, area)
    assert overlay.geometry().top() > area.bottom()


def test_native_rect_matches_logical_rect_at_scale_one(app):
    assert desktop.native_rect(QRect(10, 20, 300, 40)) == {"left": 10, "top": 20, "width": 300, "height": 40}


def test_screen_overlay_fits_translations_inside_their_blocks(app):
    overlay = desktop.ScreenOverlay()
    overlay.setGeometry(QRect(0, 0, 1000, 800))
    block = {"text": "こんにちは", "box": (0.1, 0.1, 0.4, 0.05), "line_height": 0.05, "bg": "#ffffff", "fg": "#000000"}
    overlay.set_blocks([block, dict(block, text="pending", box=(0.1, 0.5, 0.4, 0.05))])
    assert overlay._layouts == {}
    overlay.set_translation("こんにちは", "Hello", True)
    rect, size = overlay._layouts[0]
    assert rect.top() < 80 + 1 and rect.bottom() > 120 - 1
    assert 9 <= size <= 40
    assert 1 not in overlay._layouts


def test_screen_overlay_grows_a_block_down_when_the_translation_is_long(app):
    overlay = desktop.ScreenOverlay()
    overlay.setGeometry(QRect(0, 0, 1000, 800))
    block = {"text": "短い", "box": (0.1, 0.1, 0.1, 0.02), "line_height": 0.02, "bg": "#000000", "fg": "#ffffff",
             "translation": "A translation that is far longer than the original short label"}
    overlay.set_blocks([block])
    rect, _ = overlay._layouts[0]
    assert rect.height() > 0.02 * 800 + 4


class FakeApi:
    base_url = "http://127.0.0.1:1"

    def __init__(self):
        self.posts = []

    def call(self, path, body=None):
        if body is not None:
            self.posts.append((path, body))
        return {}


def make_window(tmp_path, **settings):
    from screen_translator.config import SettingsStore

    store = SettingsStore(tmp_path / "settings.json")
    store.update(settings)
    api = FakeApi()
    return desktop.MainWindow(api, store, desktop.Bridge()), api


def test_areas_can_be_added_removed_and_are_saved_together(app, tmp_path):
    window, api = make_window(tmp_path, areas=[[100, 600, 800, 100]])
    window.new_area()
    assert len(window.sources) == len(window.overlays) == 2
    assert [f.number for f in window.sources] == [1, 2]
    assert all(f.removable for f in window.sources)
    path, body = api.posts[-1]
    assert path == "/api/settings" and len(body["areas"]) == len(body["capture_rects"]) == 2
    window.remove_source(window.sources[0])
    assert window.sources[0].number == 1 and not window.sources[0].removable
    assert api.posts[-1][1]["areas"] == [[a.x(), a.y(), a.width(), a.height()] for a in [window.sources[0].area()]]


def test_each_line_lands_on_the_overlay_of_its_own_area(app, tmp_path):
    window, _ = make_window(tmp_path, areas=[[100, 100, 400, 60], [100, 600, 800, 100]])
    window.running = True
    window.on_event({"type": "line", "area": 1, "source": "犬", "translation": "Dog", "final": True,
                     "ms": 5, "context_lines": 0})
    assert window.overlays[1].isVisible() and window.overlays[1].text == "Dog"
    assert not window.overlays[0].isVisible()
    window.on_event({"type": "clear", "area": 1})
    assert not window.overlays[1].isVisible()


def test_screens_are_numbered_and_switching_moves_a_running_translation(app, tmp_path):
    window, api = make_window(tmp_path, scope="area")
    assert window.screen_pick.itemText(0).startswith("1 · ")
    window.choose_screen(1)
    assert window.whole_screen()
    assert window.flash.isVisible() and window.flash.label == "Screen 1"
    window.running = True
    api.posts.clear()
    window.choose_screen(1)
    geometry = window.chosen_screen().geometry()
    assert api.posts[-1] == ("/api/settings", {"screen_rect": desktop.native_rect(geometry),
                                               "screen_height": geometry.height()})
    assert window.screen_overlay.geometry() == geometry
    window.choose_screen(9)
    assert len(api.posts) == 1


def test_translation_text_is_sized_like_the_original(app):
    overlay = desktop.TranslationOverlay()
    area = QRect(100, 500, 900, 160)
    overlay.show_text("Chạy trốn khỏi cảnh sát.", True, area, line_height=0.15)
    assert 16 <= overlay._font_size <= round(0.15 * 160 * 0.85)
    assert text_sizes(30, 200)[0] == 26
    assert text_sizes(0, 100)[0] == 40


def test_what_you_are_watching_is_saved_as_notes(app, tmp_path):
    window, api = make_window(tmp_path, notes="GTA V gameplay")
    assert window.notes.text() == "GTA V gameplay"
    window.notes.setText("  A Korean drama ")
    window.notes.editingFinished.emit()
    assert api.posts[-1] == ("/api/settings", {"notes": "A Korean drama"})


def test_picking_a_style_updates_the_overlays_and_is_saved(app, tmp_path):
    window, api = make_window(tmp_path, overlay_box=False)
    panel = window.appearance
    assert panel.style_outline.isChecked() and not window.overlays[0].style.box
    assert not panel.opacity.isVisibleTo(panel)
    panel.style_box.click()
    assert all(overlay.style.box for overlay in window.overlays)
    assert panel.preview.style.box and panel.opacity.isVisibleTo(panel)
    assert api.posts[-1] == ("/api/settings", {"overlay_box": True})
    panel.below.click()
    assert window.overlays[0].style.mode == "below"
    assert api.posts[-1] == ("/api/settings", {"overlay_mode": "below"})
    panel.opacity.setValue(70)
    assert window.overlays[0].style.opacity == 0.7
    assert api.posts[-1] == ("/api/settings", {"overlay_opacity": 0.7})


def test_without_the_box_only_the_outlined_text_is_painted(app):
    overlay = desktop.TranslationOverlay()
    corners = {}
    for box in (True, False):
        overlay.style = Style(box=box)
        overlay.show_text("Chạy trốn khỏi cảnh sát.", True, QRect(100, 500, 800, 100), line_height=0.3)
        image = overlay.grab().toImage()
        corners[box] = image.pixelColor(3, image.height() // 2).alpha()
        assert any(image.pixelColor(x, image.height() // 2).alpha() > 200 for x in range(image.width()))
    assert corners[True] > 200 and corners[False] == 0


def test_the_box_hugs_the_text_instead_of_filling_the_area(app):
    overlay = desktop.TranslationOverlay()
    area = QRect(100, 500, 900, 200)
    overlay.show_text("Vào xe.", True, area, line_height=0.2)
    box = overlay.geometry()
    assert box.width() < area.width() / 2 and box.height() < area.height() / 2
    assert abs(box.center().x() - area.center().x()) <= 1 and abs(box.center().y() - area.center().y()) <= 1
    overlay.restyle(Style(mode="below"))
    assert overlay.geometry().top() == area.bottom() + 6


def test_text_that_comes_back_unchanged_is_left_uncovered(app):
    overlay = desktop.ScreenOverlay()
    overlay.setGeometry(QRect(0, 0, 1000, 800))
    block = {"text": "MIXIFOOD", "box": (0.8, 0.9, 0.1, 0.05), "line_height": 0.05, "bg": "#000000", "fg": "#ffffff"}
    overlay.set_blocks([block])
    overlay.set_translation("MIXIFOOD", "MIXIFOOD", True)
    assert overlay._unchanged(block)
    overlay.set_translation("MIXIFOOD", "Mixi Food", True)
    assert not overlay._unchanged(block)


def test_text_size_can_scale_the_original_or_be_fixed():
    assert text_sizes(40, 200, "x1")[0] == 34
    assert text_sizes(40, 200, "x1.5")[0] == 51
    assert text_sizes(40, 200, "x0.75")[0] == 26
    assert text_sizes(40, 200, "px24") == [24]


def test_changing_text_size_resizes_what_is_already_shown(app, tmp_path):
    window, api = make_window(tmp_path, areas=[[100, 500, 900, 200]], overlay_box=False)
    window.running = True
    window.on_event({"type": "line", "area": 0, "source": "Lose the cops", "translation": "Cắt đuôi cảnh sát",
                     "final": True, "ms": 1, "context_lines": 0, "line_height": 0.1})
    overlay = window.overlays[0]
    assert overlay._font_size <= 17
    panel = window.appearance
    panel.fixed.setChecked(True)
    assert panel.size_value.text() == "24 px" and api.posts[-1] == ("/api/settings", {"text_size": "px24"})
    panel.size.setValue(44)
    assert overlay._font_size == 44
    assert api.posts[-1] == ("/api/settings", {"text_size": "px44"})
    assert window.screen_overlay.text_size == "px44"
    panel.fixed.setChecked(False)
    panel.size.setValue(150)
    assert panel.size_value.text() == "150% of the original"
    assert api.posts[-1] == ("/api/settings", {"text_size": "x1.5"})


def test_saved_settings_load_into_the_appearance_controls(app, tmp_path):
    window, _ = make_window(tmp_path, overlay_box=True, overlay_opacity=0.6, overlay_mode="below", text_size="x1.25",
                            target_lang="ja")
    panel = window.appearance
    assert panel.style() == Style(True, 0.6, "below", "x1.25")
    assert panel.size_value.text() == "125% of the original"
    assert panel.preview.target == "ja" and not panel.style_box.icon().isNull()


def test_the_preview_follows_the_mode_and_hides_area_only_controls(app, tmp_path):
    window, _ = make_window(tmp_path, scope="area")
    panel = window.appearance
    assert panel.preview.scope == "area" and panel.style_box.isVisibleTo(panel)
    window.scope.setCurrentIndex(window.scope.findData("screen"))
    assert panel.preview.scope == "screen"
    assert not panel.style_box.isVisibleTo(panel) and not panel.cover.isVisibleTo(panel)
    assert panel.size.isVisibleTo(panel)
    image = panel.preview.grab().toImage()
    assert image.width() > 100 and image.pixelColor(image.width() // 2, 5).alpha() == 255


def test_the_preview_crop_keeps_the_area_in_view_and_stays_on_screen(app):
    screen = QRect(0, 0, 1920, 1080)
    area = QRect(560, 960, 800, 100)
    crop = desktop.preview_crop(area, screen)
    assert crop.contains(area) and screen.contains(crop)
    assert abs(crop.width() / crop.height() - 2.75) < 0.05


class PreviewCapture:
    def __init__(self):
        from screen_translator.capture import Frame

        self.frame = Frame(bytes([40, 40, 40, 255]) * (400 * 200), 400, 200)

    def grab_rect(self, rect):
        return self.frame

    def close(self):
        pass


def preview_ocr(lines):
    from screen_translator.ocr import OcrLine

    class Ocr:
        def recognize(self, frame, lang):
            return [OcrLine(text, 0.99, box) for text, box in lines]

    return Ocr


def test_the_area_preview_uses_the_real_screen_and_the_text_size_found_there(app, tmp_path, monkeypatch):
    screen = QtWidgets.QApplication.primaryScreen().geometry()
    area = QRect(screen.left() + 50, screen.bottom() - 150, screen.width() - 100, 100)
    window, _ = make_window(tmp_path, areas=[[area.x(), area.y(), area.width(), area.height()]], target_lang="vi")
    job = window.preview_job()
    seen = []
    monkeypatch.setattr(window, "preview_capture", PreviewCapture)
    monkeypatch.setattr(window, "preview_ocr_factory", preview_ocr([("Hold L1 to open your loadout", (0.1, 0.35, 0.8, 0.3))]))
    recognize = window.preview_ocr_factory.recognize
    monkeypatch.setattr(window.preview_ocr_factory, "recognize",
                        lambda self, frame, lang: seen.append((frame.width, frame.height)) or recognize(self, frame, lang))
    window.show_preview(window.build_preview(job))
    shot = window.appearance.preview.snapshot
    assert shot is not None and shot.area == area
    assert abs(shot.line_height - 0.3) < 0.01
    crop = job["crop"]
    assert seen == [(round(400 * area.width() / crop.width()), round(200 * area.height() / crop.height()))]
    assert "sized like the text" in window.appearance.caption.text()
    image = window.appearance.preview.grab().toImage()
    assert image.pixelColor(image.width() // 2, 3).red() < 60


def test_the_screen_preview_patches_real_text_and_skips_what_the_pipeline_would(app, tmp_path, monkeypatch):
    window, _ = make_window(tmp_path, scope="screen", target_lang="vi", min_text_px=12)
    lines = [("Collect the cash", (0.1, 0.1, 0.3, 0.05)), ("tiny chat", (0.1, 0.5, 0.1, 0.002)),
             ("cất súng luôn kia a", (0.1, 0.7, 0.3, 0.05))]
    monkeypatch.setattr(window, "preview_capture", PreviewCapture)
    monkeypatch.setattr(window, "preview_ocr_factory", preview_ocr(lines))
    window.show_preview(window.build_preview(window.preview_job()))
    blocks = window.appearance.preview.snapshot.blocks
    assert [b["text"] for b in blocks] == ["Collect the cash"]
    window.running = True
    window.on_event({"type": "screen_text", "source": "Collect the cash", "translation": "Lấy tiền", "final": True})
    assert window.appearance.preview.translations["Collect the cash"] == "Lấy tiền"
    window.appearance.preview.grab()


def test_without_screen_access_the_preview_falls_back_to_the_sample_scene(app, tmp_path, monkeypatch):
    window, _ = make_window(tmp_path)

    class Denied:
        def __init__(self):
            raise PermissionError("blocked")

    monkeypatch.setattr(window, "preview_capture", Denied)
    window.show_preview(window.build_preview(window.preview_job()))
    assert window.appearance.preview.snapshot is None
    assert "Allow screen recording" in window.appearance.caption.text()


def test_the_area_preview_shows_the_last_real_translation(app, tmp_path):
    window, _ = make_window(tmp_path, areas=[[100, 500, 900, 200]])
    window.running = True
    window.on_event({"type": "line", "area": 0, "source": "Lose the cops", "translation": "Cắt đuôi cảnh sát",
                     "final": True, "ms": 1, "context_lines": 0})
    assert window.appearance.preview.area_translation == "Cắt đuôi cảnh sát"


def test_below_mode_flips_above_the_area_at_the_bottom_of_the_screen(app):
    from screen_translator.appearance import place_translation

    screen = QRect(0, 0, 1920, 1080)
    low = QRect(600, 1010, 700, 50)
    rect, _ = place_translation("Giữ L1 để mở trang bị", low, Style(mode="below"), 0.6, screen)
    assert rect.bottom() < low.top() and screen.contains(rect)
    high = QRect(600, 400, 700, 50)
    rect, _ = place_translation("Giữ L1 để mở trang bị", high, Style(mode="below"), 0.6, screen)
    assert rect.top() > high.bottom()
