import asyncio
import json

import numpy as np
import pytest

from screen_translator.capture import Frame, block_colors
from screen_translator.ocr import OcrLine
from screen_translator.server import create_app
from screen_translator.text import group_blocks

from aiohttp.test_utils import TestClient, TestServer

from test_app import EchoTranslator, FakeCapture, make_pipeline_factory, store, wait_for  # noqa: F401


def px(text, x, y, w, h, width=1000, height=1000):
    return OcrLine(text, 0.99, (x / width, y / height, w / width, h / height))


def test_wrapped_rows_join_but_columns_and_headings_stay_apart():
    lines = [
        px("Settings", 100, 50, 200, 40),
        px("The quick brown fox jumps", 100, 120, 400, 20),
        px("over the lazy dog.", 100, 145, 250, 20),
        px("Sidebar item", 700, 120, 150, 20),
        px("Cancel", 100, 400, 80, 20),
    ]
    texts = [b.text for b in group_blocks(lines, 1000, 1000)]
    assert texts == ["Settings", "The quick brown fox jumps over the lazy dog.", "Sidebar item", "Cancel"]


def test_words_split_on_one_row_are_one_block():
    lines = [px("Hello", 100, 100, 60, 20), px("world", 170, 100, 60, 20)]
    [block] = group_blocks(lines, 1000, 1000)
    assert block.text == "Hello world"
    assert block.box == pytest.approx((0.1, 0.1, 0.13, 0.02))


def test_block_colors_find_the_background_and_the_ink():
    pixels = np.zeros((40, 100, 4), np.uint8)
    pixels[:, :, :3] = (250, 240, 230)
    pixels[15:25, 20:80, :3] = (30, 20, 10)
    frame = Frame(pixels.tobytes(), 100, 40)
    assert block_colors(frame, (0.1, 0.25, 0.8, 0.5)) == ("#e6f0fa", "#0a141e")


class FakeScreenCapture(FakeCapture):
    def grab_rect(self, rect):
        return self.grab(1, None)


class ScriptedScreenOcr:
    name = "fake"

    def __init__(self, script):
        self.script = list(script)

    def recognize(self, frame, lang):
        lines = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        return [OcrLine(text, 0.99, box) for text, box in lines]


async def test_whole_screen_translates_each_block_once_and_reuses_translations(store):
    menu = ("ファイル", (0.05, 0.02, 0.1, 0.03))
    first = ("こんにちは", (0.3, 0.4, 0.3, 0.05))
    second = ("さようなら", (0.3, 0.6, 0.3, 0.05))
    script = [[menu, first]] * 4 + [[menu, second]] * 4 + [[menu, first]] * 4
    store.update({"scope": "screen", "screen_rect": {"left": 0, "top": 0, "width": 16, "height": 16},
                  "screen_interval_s": 0.01})
    translator = EchoTranslator()

    def factory(store, emit):
        pipeline = make_pipeline_factory([""], translator)(store, emit)
        pipeline.ocr_factory = lambda: ScriptedScreenOcr(script)
        pipeline.capture_factory = FakeScreenCapture
        return pipeline

    app = create_app(store, factory, FakeScreenCapture)
    async with TestClient(TestServer(app, port=store.settings.port)) as client:
        events = []
        ws = await client.ws_connect("/ws")

        async def reader():
            async for msg in ws:
                events.append(json.loads(msg.data))

        task = asyncio.create_task(reader())
        await client.post("/api/start", json={})
        await wait_for(lambda: any(
            e["type"] == "screen" and [b["text"] for b in e["blocks"]] == ["ファイル", "こんにちは"]
            and all(b["translation"] for b in e["blocks"]) for e in events))
        await client.post("/api/stop", json={})
        await ws.close()
        task.cancel()

    translated = [call[-1]["content"] for call in translator.calls]
    assert translated == ["ファイル", "こんにちは", "さようなら"]
    finals = [e for e in events if e["type"] == "screen_text" and e["final"]]
    assert [e["translation"] for e in finals] == ["EN(ファイル)", "EN(こんにちは)", "EN(さようなら)"]
    assert any(e["type"] == "screen" and [b["text"] for b in e["blocks"]] == ["ファイル"] for e in events)


class TwoAreaCapture(FakeCapture):
    def grab_rect(self, rect):
        self.tick += 1
        return Frame(bytes([self.tick * 60 % 256]) * (rect["width"] * 16 * 4), rect["width"], 16)


class TwoAreaOcr:
    name = "fake"

    def recognize(self, frame, lang):
        return [OcrLine({16: "カイ", 20: "行こう"}[frame.width], 0.99, (0.1, 0.1, 0.8, 0.8))]


async def test_each_area_is_read_on_its_own_and_shares_one_context(store):
    store.update({"capture_rects": [{"left": 0, "top": 0, "width": 16, "height": 16},
                                    {"left": 0, "top": 50, "width": 20, "height": 16}]})
    translator = EchoTranslator()

    def factory(store, emit):
        pipeline = make_pipeline_factory([""], translator)(store, emit)
        pipeline.ocr_factory = TwoAreaOcr
        pipeline.capture_factory = TwoAreaCapture
        return pipeline

    events = []
    app = create_app(store, factory, TwoAreaCapture, on_event=events.append)
    async with TestClient(TestServer(app, port=store.settings.port)) as client:
        await client.post("/api/start", json={})
        await wait_for(lambda: sum(1 for e in events if e["type"] == "line" and e["final"]) >= 2)
        await client.post("/api/stop", json={})

    finals = {e["area"]: e for e in events if e["type"] == "line" and e["final"]}
    assert finals[0]["translation"] == "EN(カイ)" and finals[1]["translation"] == "EN(行こう)"
    assert sorted(e["context_lines"] for e in finals.values()) == [0, 1]
    assert finals[0]["line_height"] == pytest.approx(0.8)
