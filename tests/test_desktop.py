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
        self.state = {}
        self.responses = {}

    def call(self, path, body=None):
        if body is not None:
            self.posts.append((path, body))
        if path == "/api/state":
            return self.state
        return self.responses.get(path, {})


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


def test_below_mode_flips_above_the_area_at_the_bottom_of_the_screen(app):
    from screen_translator.appearance import place_translation

    screen = QRect(0, 0, 1920, 1080)
    low = QRect(600, 1010, 700, 50)
    rect, _ = place_translation("Giữ L1 để mở trang bị", low, Style(mode="below"), 0.6, screen)
    assert rect.bottom() < low.top() and screen.contains(rect)
    high = QRect(600, 400, 700, 50)
    rect, _ = place_translation("Giữ L1 để mở trang bị", high, Style(mode="below"), 0.6, screen)
    assert rect.top() > high.bottom()


def test_the_overlay_row_restyles_shown_translations_and_saves(app, tmp_path):
    window, api = make_window(tmp_path, areas=[[100, 500, 900, 200]], overlay_box=False, text_size="px44")
    assert not window.box.isChecked() and window.text_size.currentData() == "x1"
    window.running = True
    window.on_event({"type": "line", "area": 0, "source": "Lose the cops", "translation": "Cắt đuôi cảnh sát",
                     "final": True, "ms": 1, "context_lines": 0, "line_height": 0.1})
    overlay = window.overlays[0]
    small = overlay._font_size
    window.text_size.setCurrentIndex(window.text_size.findData("x1.5"))
    assert overlay._font_size > small
    assert api.posts[-1] == ("/api/settings", {"text_size": "x1.5"})
    assert window.screen_overlay.text_size == "x1.5"
    window.box.setChecked(True)
    assert overlay.style.box and api.posts[-1] == ("/api/settings", {"overlay_box": True})
    window.position.setCurrentIndex(window.position.findData("below"))
    assert overlay.style.mode == "below" and api.posts[-1] == ("/api/settings", {"overlay_mode": "below"})


def test_whole_screen_mode_only_shows_the_text_size(app, tmp_path):
    window, _ = make_window(tmp_path, scope="screen")
    assert window.text_size.isVisibleTo(window)
    assert not window.position.isVisibleTo(window) and not window.box.isVisibleTo(window)



def line_event(text, area=0):
    return {"type": "line", "area": area, "source": "x", "translation": text, "final": True, "ms": 1,
            "context_lines": 0, "line_height": 0.3}


def test_the_hide_shortcut_hides_translations_and_brings_back_the_latest(app, tmp_path):
    window, _ = make_window(tmp_path, areas=[[100, 500, 900, 200]])
    window.running = True
    window.on_event(line_event("Cắt đuôi cảnh sát"))
    overlay = window.overlays[0]
    assert overlay.isVisible()
    window.on_hotkey("hide")
    assert not overlay.isVisible() and "hidden" in window.status.text()
    window.on_event(line_event("Lấy tiền mặt"))
    assert not overlay.isVisible()
    window.on_hotkey("hide")
    assert overlay.isVisible() and overlay.text == "Lấy tiền mặt"


def test_the_start_shortcut_starts_and_stops(app, tmp_path):
    window, api = make_window(tmp_path)
    window.on_hotkey("toggle")
    assert window.running and api.posts[-1] == ("/api/start", {})
    window.on_hotkey("hide")
    window.on_hotkey("toggle")
    assert not window.running and api.posts[-1] == ("/api/stop", {})
    assert not window.overlays_hidden


def test_the_screen_shortcut_picks_a_screen(app, tmp_path):
    window, _ = make_window(tmp_path, scope="area")
    window.on_hotkey("screen1")
    assert window.whole_screen() and window.screen_pick.currentIndex() == 0


def test_the_window_lists_the_shortcuts_and_any_another_app_holds(app, tmp_path, monkeypatch):
    from screen_translator import hotkeys

    monkeypatch.setattr(hotkeys.GlobalHotkeys, "register", lambda self: [hotkeys.label("H")])
    window, _ = make_window(tmp_path)
    window.enable_global_hotkeys()
    text = window.shortcuts.text()
    assert hotkeys.label("T") in text and "Already used by another app: " + hotkeys.label("H") in text


def test_a_newer_release_shows_a_download_link_unless_the_check_is_off(app, tmp_path):
    from screen_translator.updates import Release

    window, _ = make_window(tmp_path)
    window.show_update(None)
    assert window.update_note.isHidden()
    window.show_update(Release("0.2.0", "https://github.com/khoale38/screen-translator/releases/tag/v0.2.0"))
    assert not window.update_note.isHidden()
    assert "0.2.0" in window.update_note.text() and "releases/tag/v0.2.0" in window.update_note.text()

    (tmp_path / "off").mkdir()
    quiet, _ = make_window(tmp_path / "off", check_updates=False)
    quiet.show_update(Release("0.2.0", "https://example.invalid"))
    assert quiet.update_note.isHidden()
    window.close()
    quiet.close()


def test_the_window_says_where_the_model_runs_and_how_long_each_step_takes(app, tmp_path):
    window, _ = make_window(tmp_path, scope="screen")
    window.set_running(True)
    window.on_event({"type": "devices", "model": "NVIDIA GeForce RTX 2060 (Vulkan)", "ocr": "GPU (DirectML)"})
    window.on_event({"type": "screen", "ms": 420, "blocks": [{"text": "犬", "box": [0.1, 0.1, 0.1, 0.05],
                                                               "line_height": 0.05, "bg": "#000", "fg": "#fff",
                                                               "translation": None}]})
    window.on_event({"type": "screen_text", "source": "犬", "translation": "Dog", "final": True, "ms": 1800,
                     "batch": 3})
    text = window.speed.text()
    assert "Model on NVIDIA GeForce RTX 2060 (Vulkan) · text reading on GPU (DirectML)" in text
    assert "Reading the screen 420 ms · translating 3 pieces 1.8 s" in text
    assert "CPU" not in text
    window.on_event({"type": "screen_text", "source": "犬", "translation": "Dog", "final": True, "ms": 900,
                     "batch": 9, "steps": {"list": 240, "steps": 31, "step": 21}})
    assert "translating 9 pieces 0.9 s (reading the list 240 ms, then 31 steps of 21 ms)" in window.speed.text()

    window.on_event({"type": "devices", "model": "CPU", "ocr": "CPU"})
    assert "Updating the graphics driver" in window.speed.text()


def test_speech_is_an_add_on_with_its_own_download_and_subtitle_box(app, tmp_path):
    window, api = make_window(tmp_path, scope="area", areas=[[100, 100, 400, 60]])
    api.state["speech"] = {"label": "Speech recognition", "size": 240_000_000, "downloaded": False}
    window.refresh_model()
    assert not window.download.isVisibleTo(window)
    window.listen.setChecked(True)
    assert ("/api/settings", {"listen": True}) in api.posts
    assert window.download.isVisibleTo(window) and "speech recognition (0.2 GB)" in window.download.text()

    window.running = True
    window.on_event({"type": "line", "id": 1, "area": 0, "source": "看板", "translation": "Sign", "final": True,
                     "ms": 300, "context_lines": 0})
    window.on_event({"type": "line", "id": "speech-1", "area": "speech", "source": "待って", "translation": "Wait",
                     "final": True, "ms": 200, "context_lines": 1})
    assert window.overlays[0].text == "Sign" and window.speech_overlay.text == "Wait"
    assert window.speech_overlay.geometry().center().x() == window.subtitle_area().center().x()
    assert "Speech translated 0.2 s after it ended" in window.speed.text()


def test_settings_from_the_speech_only_test_build_turn_into_the_add_on(tmp_path):
    from screen_translator.config import Settings

    settings = Settings.from_dict({"scope": "speech"})
    assert (settings.scope, settings.listen) == ("area", True)


def test_dubbing_appears_with_speech_and_offers_voices_accents_and_game_sound(app, tmp_path):
    window, api = make_window(tmp_path, scope="area", target_lang="en")
    api.responses["/api/voices"] = {"voices": [
        {"value": "", "label": "Match the speaker"},
        {"value": "accent:en-GB", "label": "Match the speaker · UK accent"},
        {"value": "voice:zira", "label": "Microsoft Zira · en-US · female"}]}
    assert not window.dub.isVisibleTo(window)
    window.listen.setChecked(True)
    assert window.dub.isVisibleTo(window) and not window.dub_row.isVisibleTo(window)
    window.dub.setChecked(True)
    assert ("/api/settings", {"dub": True}) in api.posts
    assert window.dub_row.isVisibleTo(window)
    assert [window.dub_voice.itemText(i) for i in range(window.dub_voice.count())][1] == "Match the speaker · UK accent"
    window.dub_original.setCurrentIndex(window.dub_original.findData("mute"))
    window.dub_original.activated.emit(window.dub_original.currentIndex())
    assert ("/api/settings", {"dub_original": "mute"}) in api.posts

    api.responses["/api/voices"] = {"voices": [{"value": "", "label": "Match the speaker"}]}
    window.refresh_voices()
    assert "No voice for" in window.status.text()


def test_copy_log_puts_the_report_on_the_clipboard(app, tmp_path):
    window, api = make_window(tmp_path)
    api.responses["/api/log"] = {"text": "Screen Translator 0.1.0\nline 0: 300 ms", "path": "C:/st/screen-translator.log"}
    window.copy_log.click()
    assert QtWidgets.QApplication.clipboard().text() == "Screen Translator 0.1.0\nline 0: 300 ms"
    assert "C:/st/screen-translator.log" in window.status.text()
