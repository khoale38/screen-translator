from screen_translator import diagnostics
from screen_translator.config import Settings


def test_only_finished_steps_are_logged_with_their_timings():
    assert diagnostics.describe({"type": "line", "final": False, "translation": "Hel"}) is None
    line = diagnostics.describe({"type": "line", "final": True, "area": 0, "ms": 640, "first_ms": 170,
                                 "context_lines": 3, "source": "カイ\nまさか", "translation": "Kai: No way"})
    assert line == "line 0: 640 ms, first word 170 ms, 3 lines of context: 'カイ / まさか' -> 'Kai: No way'"
    screen = diagnostics.describe({"type": "screen_text", "final": True, "ms": 900, "batch": 9, "source": "Settings",
                                   "translation": "Cài đặt", "steps": {"list": 400, "steps": 20, "step": 25}})
    assert "batch of 9 (list 400 ms, 20 steps of 25 ms)" in screen
    assert diagnostics.describe({"type": "status", "state": "error", "error": "boom"}) == "status error: boom"


def test_the_report_starts_with_the_machine_and_settings():
    events = []
    emit = diagnostics.recording(events.append)
    emit({"type": "devices", "model": "NVIDIA GeForce RTX 2060 (Vulkan)", "ocr": "DirectML"})
    text = diagnostics.report(Settings(target_lang="vi", notes="GTA V"), {"model": "NVIDIA GeForce RTX 2060 (Vulkan)"})
    first, devices, settings = text.splitlines()[:3]
    assert first.startswith("Screen Translator ")
    assert devices.startswith("Model on NVIDIA GeForce RTX 2060 (Vulkan)")
    assert "auto -> vi" in settings and "'GTA V'" in settings
    assert text.splitlines()[-1].endswith("model on NVIDIA GeForce RTX 2060 (Vulkan), text reading on DirectML")
    assert len(events) == 1
