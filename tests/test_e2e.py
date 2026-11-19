"""Real OCR + real model on a rendered typewriter dialogue. Opt in with ST_E2E=1 (slow, downloads a model)."""

import os
import tempfile
import threading
import time
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from screen_translator.capture import Frame
from screen_translator.config import SettingsStore
from screen_translator.models import MODELS, download
from screen_translator.ocr import default_engine
from screen_translator.pipeline import Pipeline

pytestmark = pytest.mark.skipif(os.environ.get("ST_E2E") != "1", reason="set ST_E2E=1 to run")

SCRIPT = [
    ("カイ", "ユキはまだ帰ってこないの？"),
    ("アオイ", "彼女、昨日から連絡がないんだ。"),
    ("カイ", "まさか、あの森に行ったんじゃ…"),
    ("アオイ", "…行ったよ。止めたけど聞かなかった。"),
]
CJK_FONTS = [
    "C:/Windows/Fonts/YuGothM.ttc", "C:/Windows/Fonts/meiryo.ttc", "C:/Windows/Fonts/msgothic.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]


class DialogueScreen:
    char_s, hold_s = 0.07, 3.0

    def __init__(self):
        self.font = ImageFont.truetype(next(f for f in CJK_FONTS if Path(f).exists()), 34)
        self.t0 = None

    def grab(self, monitor, region):
        self.t0 = self.t0 or time.monotonic()
        t = time.monotonic() - self.t0
        name, shown = "", ""
        for speaker, line in SCRIPT:
            duration = len(line) * self.char_s + self.hold_s
            if t < duration:
                name, shown = speaker, line[: int(t / self.char_s) + 1]
                break
            t -= duration
        image = Image.new("RGB", (900, 150), (12, 14, 20))
        draw = ImageDraw.Draw(image)
        draw.text((30, 20), name, font=self.font, fill=(255, 212, 121))
        draw.text((30, 80), shown, font=self.font, fill=(255, 255, 255))
        return Frame(image.convert("RGBA").tobytes("raw", "BGRA"), image.width, image.height)

    def close(self):
        pass


def test_translates_a_dialogue_with_context_end_to_end():
    spec = MODELS[os.environ.get("ST_E2E_MODEL", "qwen3.5-2b")]
    if not spec.downloaded:
        download(spec, lambda done, total: None, threading.Event())

    store = SettingsStore(Path(tempfile.mkdtemp()) / "settings.json")
    store.update({"model": spec.key, "source_lang": "ja", "target_lang": "en",
                  "glossary": {"ユキ": "Yuki", "カイ": "Kai", "アオイ": "Aoi"}})
    ocr = default_engine()
    ocr.recognize(DialogueScreen().grab(1, None), "ja")
    events = []
    pipeline = Pipeline(store, events.append, ocr_factory=lambda: ocr, capture_factory=DialogueScreen,
                        permission_check=lambda: None)
    pipeline.start()
    deadline = time.monotonic() + float(os.environ.get("ST_E2E_TIMEOUT", "240"))
    finals = lambda: [e for e in events if e["type"] == "line" and e["final"]]
    while len(finals()) < len(SCRIPT) and time.monotonic() < deadline:
        errors = [e for e in events if e["type"] == "status" and e["state"] == "error"]
        assert not errors, errors
        time.sleep(0.2)
    pipeline.shutdown()

    lines = finals()
    print(f"OCR: {ocr.name}")
    for line in lines:
        print(f"{line['ms']:5}ms ctx={line['context_lines']}  {line['source']!r} -> {line['translation']!r}")
    assert len(lines) == len(SCRIPT)
    assert [line["context_lines"] for line in lines] == [0, 1, 2, 3]
    for line, (speaker, _) in zip(lines, SCRIPT):
        assert line["source"].startswith(speaker + "\n")
        assert line["translation"].startswith(("Kai: ", "Aoi: ")), line["translation"]
        assert line["translation"].isascii() and len(line["translation"]) > 12
    assert "Yuki" in lines[0]["translation"]
