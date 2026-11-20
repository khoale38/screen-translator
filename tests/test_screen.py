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


class BatchEchoTranslator(EchoTranslator):
    """Answers numbered lists like the model does, but skips any line containing "skip me"."""

    def translate(self, messages, on_partial=None):
        content = messages[-1]["content"]
        rows = [line.split(". ", 1) for line in content.splitlines()]
        if len(rows) < 2 or not all(len(r) == 2 and r[0].isdigit() for r in rows):
            return super().translate(messages, on_partial)
        self.calls.append(messages)
        reply = "\n".join(f"{n}. EN({text})" for n, text in rows if "skip me" not in text)
        if on_partial:
            on_partial(reply.splitlines()[0] + "\n")
        return reply


async def test_whole_screen_translates_each_block_once_and_reuses_translations(store):
    menu = ("ファイル", (0.05, 0.02, 0.1, 0.03))
    first = ("こんにちは", (0.3, 0.4, 0.3, 0.05))
    second = ("さようなら", (0.3, 0.6, 0.3, 0.05))
    script = [[menu, first]] * 4 + [[menu, second]] * 4 + [[menu, first]] * 4
    store.update({"scope": "screen", "min_text_px": 0, "screen_rect": {"left": 0, "top": 0, "width": 16, "height": 16},
                  "screen_interval_s": 0.01})
    translator = BatchEchoTranslator()

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
    assert translated == ["1. ファイル\n2. こんにちは", "さようなら"]
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



def test_batch_replies_are_parsed_by_number_and_unfinished_lines_wait():
    from screen_translator.translator import parse_numbered

    raw = "1. Cài đặt\n2) Tiếp tục chơi\n3. Thoát"
    assert parse_numbered(raw, 3) == {1: "Cài đặt", 2: "Tiếp tục chơi", 3: "Thoát"}
    assert parse_numbered(raw, 3, complete=False) == {1: "Cài đặt", 2: "Tiếp tục chơi"}
    assert parse_numbered("1. A\n7. out of range\nchatter", 3) == {1: "A"}


def test_batches_are_capped_by_count_and_length():
    from screen_translator.pipeline import _take_batch

    assert len(_take_batch([f"label {i}" for i in range(30)])) == 12
    assert _take_batch(["x" * 1000, "y" * 300, "z"]) == ["x" * 1000]
    assert _take_batch(["x" * 5000]) == ["x" * 5000]


async def test_blocks_the_model_skips_in_a_batch_are_translated_one_by_one(store):
    lines = [(f"line {i}", (0.1, 0.1 * i, 0.5, 0.04)) for i in range(1, 4)] + [("skip me", (0.1, 0.6, 0.5, 0.04))]
    store.update({"scope": "screen", "min_text_px": 0, "screen_rect": {"left": 0, "top": 0, "width": 16, "height": 16},
                  "screen_interval_s": 0.01})
    translator = BatchEchoTranslator()

    def factory(store, emit):
        pipeline = make_pipeline_factory([""], translator)(store, emit)
        pipeline.ocr_factory = lambda: ScriptedScreenOcr([lines])
        pipeline.capture_factory = FakeScreenCapture
        return pipeline

    events = []
    app = create_app(store, factory, FakeScreenCapture, on_event=events.append)
    async with TestClient(TestServer(app, port=store.settings.port)) as client:
        await client.post("/api/start", json={})
        await wait_for(lambda: sum(1 for e in events if e["type"] == "screen_text" and e["final"]) >= 4)
        await client.post("/api/stop", json={})

    assert len(translator.calls) == 2
    assert translator.calls[1][-1]["content"] == "skip me"
    finals = {e["source"]: e["translation"] for e in events if e["type"] == "screen_text" and e["final"]}
    assert finals == {"line 1": "EN(line 1)", "line 2": "EN(line 2)", "line 3": "EN(line 3)", "skip me": "EN(skip me)"}
    assert any(e["type"] == "screen_text" and not e["final"] for e in events)


class StillCapture(FakeCapture):
    def grab_rect(self, rect):
        return Frame(bytes([90]) * (16 * 16 * 4), 16, 16)


class CountingOcr(ScriptedScreenOcr):
    reads = 0

    def recognize(self, frame, lang):
        CountingOcr.reads += 1
        return super().recognize(frame, lang)


async def test_a_still_screen_is_published_after_a_single_read(store):
    store.update({"scope": "screen", "min_text_px": 0, "screen_rect": {"left": 0, "top": 0, "width": 16, "height": 16},
                  "screen_interval_s": 0.3})
    CountingOcr.reads = 0

    def factory(store, emit):
        pipeline = make_pipeline_factory([""], BatchEchoTranslator())(store, emit)
        pipeline.ocr_factory = lambda: CountingOcr([[("Settings", (0.1, 0.1, 0.3, 0.05))]])
        pipeline.capture_factory = StillCapture
        return pipeline

    events = []
    app = create_app(store, factory, StillCapture, on_event=events.append)
    async with TestClient(TestServer(app, port=store.settings.port)) as client:
        await client.post("/api/start", json={})
        await wait_for(lambda: any(e["type"] == "screen" and e["blocks"] for e in events), timeout=0.25)
        await client.post("/api/stop", json={})
    assert CountingOcr.reads == 1


async def test_text_smaller_than_the_chosen_size_is_skipped(store):
    store.update({"scope": "screen", "min_text_px": 12, "screen_height": 1080, "screen_interval_s": 0.01,
                  "screen_rect": {"left": 0, "top": 0, "width": 16, "height": 16}})
    lines = [("Hold L1 to open your loadout", (0.4, 0.9, 0.2, 31 / 1080)), ("cat sina nia", (0.0, 0.6, 0.1, 9 / 1080))]

    def factory(store, emit):
        pipeline = make_pipeline_factory([""], BatchEchoTranslator())(store, emit)
        pipeline.ocr_factory = lambda: ScriptedScreenOcr([lines])
        pipeline.capture_factory = StillCapture
        return pipeline

    events = []
    app = create_app(store, factory, StillCapture, on_event=events.append)
    async with TestClient(TestServer(app, port=store.settings.port)) as client:
        await client.post("/api/start", json={})
        await wait_for(lambda: any(e["type"] == "screen" and e["blocks"] for e in events))
        await client.post("/api/stop", json={})
    shown = next(e for e in events if e["type"] == "screen" and e["blocks"])
    assert [b["text"] for b in shown["blocks"]] == ["Hold L1 to open your loadout"]


def test_an_overlay_that_connects_late_gets_the_screen_with_every_known_translation():
    from screen_translator.server import Hub

    hub = Hub()
    hub.emit({"type": "screen", "ms": 5, "blocks": [
        {"text": "メニュー", "box": (0, 0, 0.1, 0.1), "line_height": 0.02, "bg": "#000", "fg": "#fff",
         "translation": "Menu"},
        {"text": "こんにちは", "box": (0, 0.5, 0.3, 0.1), "line_height": 0.02, "bg": "#000", "fg": "#fff",
         "translation": None},
    ]})
    hub.emit({"type": "screen_text", "source": "こんにちは", "translation": "Hel", "final": False})
    assert [b["translation"] for b in hub.screen_snapshot()["blocks"]] == ["Menu", None]
    hub.emit({"type": "screen_text", "source": "こんにちは", "translation": "Hello", "final": True})
    assert [b["translation"] for b in hub.screen_snapshot()["blocks"]] == ["Menu", "Hello"]
    hub.emit({"type": "status", "state": "stopped", "error": ""})
    assert hub.screen_snapshot() is None
