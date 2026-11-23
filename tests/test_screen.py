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
    device = "fake GPU"

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
        def shown(texts):
            return [i for i, e in enumerate(events) if e["type"] == "screen"
                    and [b["text"] for b in e["blocks"]] == texts and all(b["translation"] for b in e["blocks"])]

        await wait_for(lambda: shown(["ファイル", "さようなら"])
                       and any(i > shown(["ファイル", "さようなら"])[0] for i in shown(["ファイル", "こんにちは"])))
        await client.post("/api/stop", json={})
        await ws.close()
        task.cancel()

    translated = [call[-1]["content"] for call in translator.calls]
    assert translated == ["1. ファイル\n2. こんにちは", "さようなら"]
    finals = [e for e in events if e["type"] == "screen_text" and e["final"]]
    assert [e["translation"] for e in finals] == ["EN(ファイル)", "EN(こんにちは)", "EN(さようなら)"]
    assert any(e["type"] == "screen" and [b["text"] for b in e["blocks"]] == ["ファイル"] for e in events)
    assert [e["ocr"] for e in events if e["type"] == "devices"] == ["fake GPU"]


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


def test_each_screen_item_gets_the_same_prompt_up_to_the_item_it_asks_for():
    from screen_translator.translator import build_screen_messages, strip_item_letter

    screen = ["Settings", "Lose the cops.\nNow"]
    first, second = (build_screen_messages(i, screen, [], "en", "vi") for i in range(2))
    assert first[0] == second[0]
    assert first[-1]["content"] == "A. Settings\nB. Lose the cops. / Now\n\nTranslate item A: Settings"
    assert second[-1]["content"].endswith("Translate item B: Lose the cops. / Now")
    assert strip_item_letter("B. Cắt đuôi cảnh sát.", 1) == "Cắt đuôi cảnh sát."
    assert strip_item_letter("C. Cắt đuôi cảnh sát.", 1) == "Cắt đuôi cảnh sát."
    assert strip_item_letter("Translate item B: Cài đặt", 1) == "Cài đặt"
    assert strip_item_letter("Bắt đầu", 1) == "Bắt đầu"
    assert strip_item_letter("Tiêu đề J: Phụ đề", 9) == "Phụ đề"
    assert strip_item_letter("Tỷ lệ 3:2", 1) == "Tỷ lệ 3:2"


class SideBySideTranslator(EchoTranslator):
    """Translates screen items side by side; echoes the item letter like small models do."""

    def __init__(self):
        super().__init__()
        self.parallel_calls = []

    def translate_each(self, conversations, lengths, on_piece):
        self.parallel_calls.append(len(conversations))
        results = []
        for index, messages in enumerate(conversations):
            content = messages[-1]["content"]
            letter = content.rsplit("Translate item ", 1)[1][0]
            item = next(line[3:] for line in content.splitlines() if line.startswith(f"{letter}. "))
            on_piece(index, f"{letter}. EN(", False)
            on_piece(index, f"{letter}. EN({item})", True)
            results.append(f"EN({item})")
        return results


async def test_whole_screen_translates_its_pieces_side_by_side_when_the_model_can(store):
    menu = ("ファイル", (0.05, 0.02, 0.1, 0.03))
    first = ("こんにちは", (0.3, 0.4, 0.3, 0.05))
    store.update({"scope": "screen", "min_text_px": 0, "screen_rect": {"left": 0, "top": 0, "width": 16, "height": 16},
                  "screen_interval_s": 0.01})
    translator = SideBySideTranslator()

    def factory(store, emit):
        pipeline = make_pipeline_factory([""], translator)(store, emit)
        pipeline.ocr_factory = lambda: ScriptedScreenOcr([[menu, first]])
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
        await wait_for(lambda: sum(e["type"] == "screen_text" and e["final"] for e in events) >= 2)
        await client.post("/api/stop", json={})
        await ws.close()
        task.cancel()

    finals = {e["source"]: e["translation"] for e in events if e["type"] == "screen_text" and e["final"]}
    assert finals == {"ファイル": "EN(ファイル)", "こんにちは": "EN(こんにちは)"}
    assert translator.parallel_calls == [2, 2] and translator.calls == []


def _px(x, y, w, h, width=1280, height=900):
    return (x / width, y / height, w / width, h / height)


def test_manga_columns_make_one_bubble_read_right_to_left():
    lines = [
        OcrLine("やる！", 0.96, _px(340, 149, 42, 101)),
        OcrLine("ぶっ倒して", 1.0, _px(382, 147, 44, 159)),
        OcrLine("あいつを", 0.99, _px(423, 148, 44, 132)),
        OcrLine("今日こそ", 1.0, _px(466, 150, 42, 128)),
        OcrLine("どこに行くんだ？", 0.98, _px(1009, 118, 43, 259)),
        OcrLine("おい、待てよ！", 0.96, _px(1054, 118, 43, 229)),
        OcrLine("第3話 約束の場所", 0.96, _px(695, 816, 270, 40)),
    ]
    blocks = {b.text: b for b in group_blocks(lines, 1280, 900)}
    assert set(blocks) == {"今日こそあいつをぶっ倒してやる!", "おい、待てよ!どこに行くんだ?", "第3話 約束の場所"}
    bubble = blocks["今日こそあいつをぶっ倒してやる!"]
    assert bubble.vertical and round(bubble.line_height * 900) == 43
    assert not blocks["第3話 約束の場所"].vertical


def test_centred_columns_furigana_and_a_short_last_column_stay_in_one_bubble():
    lines = [
        OcrLine("今日は本当に", 0.99, _px(500, 100, 42, 250)),
        OcrLine("ほんとう", 0.99, _px(543, 160, 18, 80)),
        OcrLine("楽しかった", 0.99, _px(452, 120, 42, 210)),
        OcrLine("よ。", 0.99, _px(404, 200, 42, 60)),
        OcrLine("だ", 0.99, _px(356, 205, 42, 42)),
    ]
    [bubble] = group_blocks(lines, 1280, 900)
    assert bubble.text == "今日は本当に楽しかったよ。だ" and bubble.vertical
    assert bubble.box[0] + bubble.box[2] >= (543 + 18) / 1280 - 1e-9


def test_manga_bubbles_are_listed_right_to_left():
    lines = [OcrLine("右の吹き出し", 0.99, _px(900, 100, 42, 250)), OcrLine("左の吹き出し", 0.99, _px(300, 105, 42, 250))]
    assert [b.text for b in group_blocks(lines, 1280, 900)] == ["右の吹き出し", "左の吹き出し"]


def test_menu_items_and_a_button_icon_gap_do_not_merge():
    menu = [px("New Game", 100, 100, 160, 30), px("Load Game", 100, 150, 170, 30), px("Options", 100, 200, 130, 30)]
    assert [b.text for b in group_blocks(menu, 1000, 1000)] == ["New Game", "Load Game", "Options"]
    prompt = [px("Press", 100, 500, 80, 30), px("to interact", 220, 500, 150, 30)]
    assert len(group_blocks(prompt, 1000, 1000)) == 2


def test_a_short_row_that_stops_mid_phrase_still_wraps():
    lines = [px("北の洞窟で", 100, 100, 150, 30), px("賢者に会う", 100, 140, 150, 30)]
    assert [b.text for b in group_blocks(lines, 1000, 1000)] == ["北の洞窟で賢者に会う"]
    lines = [px("Secure the", 100, 100, 150, 30), px("comms tower", 100, 140, 170, 30)]
    assert [b.text for b in group_blocks(lines, 1000, 1000)] == ["Secure the comms tower"]


def test_short_lines_of_one_speech_bubble_stay_together():
    bubble = [px("WAIT!", 100, 100, 90, 30), px("WHERE ARE", 90, 132, 120, 30), px("YOU GOING?", 85, 164, 130, 30)]
    assert [b.text for b in group_blocks(bubble, 1000, 1000)] == ["WAIT! WHERE ARE YOU GOING?"]
    korean = [px("오빠", 100, 100, 60, 30), px("어디 가?", 90, 133, 90, 30)]
    assert len(group_blocks(korean, 1000, 1000)) == 1


def test_manga_rows_are_read_right_to_left_even_when_bubbles_start_at_different_heights():
    lines = [OcrLine("左の吹き出しです", 0.99, _px(300, 100, 42, 300)),
             OcrLine("右の吹き出しです", 0.99, _px(900, 160, 42, 300)),
             OcrLine("下の吹き出しです", 0.99, _px(600, 600, 42, 250))]
    assert [b.text for b in group_blocks(lines, 1280, 900)] == ["右の吹き出しです", "左の吹き出しです", "下の吹き出しです"]


def test_hud_stacks_chat_lines_and_toggles_stay_apart():
    hud = [px("HP 1250/1500", 100, 100, 160, 30), px("MP 320/400", 100, 134, 140, 30)]
    assert len(group_blocks(hud, 1000, 1000)) == 2
    chat = [px("[Party] Minsu: heal me please", 100, 500, 300, 30), px("[Guild] Haneul: raid at 9", 100, 534, 260, 30)]
    assert len(group_blocks(chat, 1000, 1000)) == 2
    toggle = [px("On", 100, 300, 40, 30), px("Off", 100, 350, 45, 30)]
    assert len(group_blocks(toggle, 1000, 1000)) == 2
    tilted = [px("THIS IS A LONG", 100, 100, 200, 30), px("SPEECH BUBBLE", 100, 118, 200, 30)]
    assert len(group_blocks(tilted, 1000, 1000)) == 1


def test_a_name_tab_joins_its_dialogue_and_furigana_and_split_bubbles_behave():
    tab = [px("ミサキ", 100, 700, 90, 30), px("また遅刻？これで三回目だよ。", 100, 745, 420, 30)]
    assert [b.text for b in group_blocks(tab, 1000, 1000)] == ["ミサキ\nまた遅刻?これで三回目だよ。"]
    ruby = [px("ほんとう", 140, 300, 60, 14), px("本当に楽しかった", 120, 316, 220, 30)]
    assert [b.text for b in group_blocks(ruby, 1000, 1000)] == ["本当に楽しかった"]
    bubble = [px("Only 3", 100, 100, 90, 30), px("tickets left?!", 90, 133, 140, 30)]
    assert [b.text for b in group_blocks(bubble, 1000, 1000)] == ["Only 3 tickets left?!"]


def test_name_tabs_join_only_one_clear_pair_and_furigana_needs_kanji_under_it():
    headers = [px("Inventory", 100, 100, 120, 30), px("A sword forged in fire, sharp and old.", 100, 140, 420, 30),
               px("Quests", 600, 100, 90, 30), px("Find the lost blade in the northern cave.", 600, 140, 420, 30)]
    assert len(group_blocks(headers, 1000, 1000)) == 4
    lead_in = [px("いいかげんに", 120, 300, 80, 14), px("しろ!!", 110, 316, 120, 30)]
    assert "いいかげんに" in group_blocks(lead_in, 1000, 1000)[0].text
    label = [px("アオイ", 100, 700, 60, 18), px("ねえ、タクミ。明日の花火大会、一緒に行かない？", 100, 720, 520, 34)]
    assert group_blocks(label, 1000, 1000)[0].text.startswith("アオイ")


def test_one_dialogue_name_tab_joins_even_with_a_quest_panel_on_screen():
    blocks = [px("Quest", 700, 100, 80, 30), px("Find the lost blade in the northern caves", 700, 140, 280, 30),
              px("Mira", 100, 720, 60, 30), px("You came back! I knew you would.", 100, 760, 420, 30)]
    texts = [b.text for b in group_blocks(blocks, 1000, 1000)]
    assert "Mira\nYou came back! I knew you would." in texts and "Quest" in texts


def test_a_key_tag_mid_sentence_and_a_number_line_in_a_bubble_stay_joined():
    tutorial = [px("宝箱を開けるには", 100, 100, 200, 30), px("[E] を押してください", 100, 134, 220, 30)]
    assert len(group_blocks(tutorial, 1000, 1000)) == 1
    bubble = [px("Ready? Only", 100, 300, 140, 30), px("30!", 120, 333, 50, 30)]
    assert [b.text for b in group_blocks(bubble, 1000, 1000)] == ["Ready? Only 30!"]


def test_round_8_cutscene_names_hp_readouts_and_quest_titles_stay_apart():
    cutscene = [px("Aria", 920, 860, 80, 49, 1920, 1080), px("We can't stay here. The bridge won't hold much longer.", 470, 917, 982, 56, 1920, 1080)]
    assert [b.text for b in group_blocks(cutscene, 1920, 1080)] == [
        "Aria\nWe can't stay here. The bridge won't hold much longer."]
    yui = [px("ユイ", 920, 860, 72, 41, 1920, 1080), px("お兄ちゃん、待って！一人で行かないで！", 570, 908, 766, 65, 1920, 1080)]
    assert [b.text for b in group_blocks(yui, 1920, 1080)] == ["ユイ\nお兄ちゃん、待って!一人で行かないで!"]
    hud = [px("Bram", 60, 900, 90, 32, 1920, 1080), px("1,430 / 1,600", 60, 936, 150, 26, 1920, 1080)]
    assert "Bram" in [b.text for b in group_blocks(hud, 1920, 1080)]
    toast = [px("クエスト更新", 66, 315, 163, 36, 1920, 1080), px("粉屋の消えた娘", 64, 360, 235, 43, 1920, 1080),
             px("夜に古い水車小屋を調べる", 65, 411, 294, 33, 1920, 1080)]
    assert len(group_blocks(toast, 1920, 1080)) == 3
    bubble = [px("哥，你明天", 253, 169, 155, 38, 1920, 1080), px("陪我去买书好不好？", 194, 209, 260, 33, 1920, 1080)]
    assert len(group_blocks(bubble, 1920, 1080)) == 1


def test_round_9_short_cutscene_lines_and_hud_buttons_keep_the_name_apart():
    from screen_translator.text import assemble

    for name, line in (("星璃", "旅行者，快跟上！"), ("세라", "괜찮아, 내가 있잖아."), ("Aria", "Stay close to me.")):
        rows = [px(name, 930, 860, 60, 34, 1920, 1080), px(line, 800, 902, 320, 40, 1920, 1080)]
        blocks = [b.text for b in group_blocks(rows, 1920, 1080)]
        assert len(blocks) == 1 and blocks[0].split("\n")[0] == name, blocks
    window = [px("オート", 1595, 46, 85, 34, 1920, 1080), px("スキップ", 1754, 42, 117, 41, 1920, 1080),
              px("ユイ", 920, 860, 72, 41, 1920, 1080), px("お兄ちゃん、待って！", 760, 908, 400, 65, 1920, 1080)]
    assert assemble(window, width=1920, height=1080).split("\n")[0] == "ユイ"
    bubble = [px("형", 240, 107, 38, 40), px("어디 가?", 201, 145, 118, 44)]
    blocks = [b.text for b in group_blocks(bubble, 1000, 1000)]
    assert len(blocks) == 1 and "어디 가?" in blocks[0]


def test_round_12_brand_names_stray_think_tags_and_glued_speech():
    from screen_translator.pipeline import _punctuated
    from screen_translator.translator import clean_output

    assert clean_output("Kỹ thuật nhẹ\n</think>\n\nKỹ thuật nhẹ") == "Kỹ thuật nhẹ"
    assert [_punctuated(t) for t in ("네마마.", "爹女儿回来了。")] == ["네, 마마.", "爹，女儿回来了。"]


def test_round_13_capitals_jamo_and_ocr_degree_signs():
    import pathlib
    import tempfile

    from screen_translator.config import SettingsStore
    from screen_translator.pipeline import Pipeline
    from screen_translator.text import normalize

    store = SettingsStore(pathlib.Path(tempfile.mkdtemp()) / "settings.json")
    store.update({"target_lang": "vi"})
    pipeline = Pipeline(store, lambda e: None)
    assert [pipeline._capitalised(t) for t in ("x3 Bình máu", "sinh nhật: 12/5", "iPhone mới", "ㅋㅋㅋ vui quá")] == [
        "x3 Bình máu", "Sinh nhật: 12/5", "iPhone mới", "Haha vui quá"]
    assert normalize("ハル°") == "ハル。" and normalize("180°C") == "180°C"


def test_round_13_one_syllable_lines_are_not_name_tabs():
    from screen_translator.text import assemble

    rows = [px("응", 100, 560, 30, 30, 1280, 720), px("그래, 알았어. 내일 보자.", 100, 600, 600, 32, 1280, 720)]
    assert assemble(rows, width=1280, height=720) == "응, 그래, 알았어. 내일 보자."
    rows = [px("哥", 620, 560, 30, 30, 1280, 720), px("等等我！", 580, 600, 120, 32, 1280, 720)]
    assert assemble(rows, width=1280, height=720) == "哥，等等我！"
    rows = [px("等等我！", 580, 560, 120, 32, 1280, 720), px("嗯。", 610, 600, 45, 30, 1280, 720)]
    assert assemble(rows, width=1280, height=720) == "等等我！嗯。"
    rows = [px("姫", 100, 560, 30, 30, 1280, 720), px("お待ちください！", 100, 600, 600, 32, 1280, 720)]
    assert assemble(rows, width=1280, height=720).split("\n")[0] == "姫"


def test_round_15_loops_markdown_jamo_and_katakana_name_ends():
    import pathlib
    import tempfile

    from screen_translator.config import SettingsStore
    from screen_translator.pipeline import Pipeline
    from screen_translator.text import normalize
    from screen_translator.translator import clean_output

    from screen_translator.pipeline import _collapsed

    assert _collapsed("Ông ấy nhờ anh ba ba ba ba ba ba ba") == "Ông ấy nhờ anh ba"
    assert _collapsed("Không, không, không, không, không!") == "Không, không, không, không, không!"
    assert clean_output("**Trận chiến Lượt thứ 3!**") == "Trận chiến Lượt thứ 3!"
    assert normalize("【ミ力】行くよ") == "【ミカ】行くよ" and normalize("体力") == "体力"
    store = SettingsStore(pathlib.Path(tempfile.mkdtemp()) / "settings.json")
    store.update({"target_lang": "vi"})
    assert Pipeline(store, lambda e: None)._capitalised("ㅎㅎ vui ㅎ") == "Hehe vui"


def test_round_15_centred_one_character_calls_and_orphan_rows():
    from screen_translator.text import assemble

    rows = [px("妈", 595, 560, 79, 67, 1280, 720), px("我回来了。", 537, 610, 187, 67, 1280, 720)]
    assert assemble(rows, width=1280, height=720) == "妈，我回来了。"
    rows = [px("李明", 100, 500, 80, 40, 1280, 720), px("天色已晚，我们就在这破庙里歇一晚", 100, 550, 700, 40, 1280, 720),
            px("吧。", 100, 595, 60, 40, 1280, 720), px("按空格键继续", 1000, 680, 200, 30, 1280, 720)]
    assert assemble(rows, width=1280, height=720).split("\n")[-1] == "按空格键继续"


def test_round_16_area_furigana_buttons_and_choice_menus():
    from screen_translator.text import assemble

    W, H = 1280, 720
    furigana = [px("あした", 100, 500, 60, 18, W, H), px("えき", 190, 500, 40, 18, W, H),
                px("明日、駅の前で待ってるから。", 100, 520, 560, 36, W, H)]
    assert assemble(furigana, width=W, height=H, menus=True) == "明日、駅の前で待ってるから。"
    buttons = [px("白露", 100, 500, 80, 40, W, H), px("自动 跳过 记录", 900, 500, 300, 30, W, H),
               px("师兄，这次的任务就交给我吧！", 100, 550, 600, 40, W, H)]
    assert assemble(buttons, width=W, height=H, menus=True) == "白露\n师兄，这次的任务就交给我吧！"
    choices = [px("你真的要离开这个村子吗？", 400, 500, 480, 40, W, H), px("是", 400, 550, 40, 40, W, H),
               px("否", 400, 600, 40, 40, W, H)]
    assert assemble(choices, width=W, height=H, menus=True) == "你真的要离开这个村子吗？\n是\n否"
    call = [px("师父", 600, 560, 60, 30, W, H), px("弟子知错了。", 560, 600, 160, 32, W, H)]
    assert assemble(call, width=W, height=H, menus=True) == "师父，弟子知错了。"


def test_round_17_short_two_line_subtitles_stay_one_line_and_choices_follow_questions():
    from screen_translator.text import assemble

    W, H = 1280, 720
    subtitle = [px("我真的不知道", 520, 560, 240, 40, W, H), px("你在说什么", 530, 605, 200, 40, W, H)]
    assert assemble(subtitle, width=W, height=H, menus=True) == "我真的不知道你在说什么"
    choices = [px("정말 떠나시겠습니까?", 400, 500, 400, 40, W, H), px("예", 400, 550, 40, 40, W, H),
               px("아니요", 400, 600, 120, 40, W, H)]
    assert assemble(choices, width=W, height=H, menus=True) == "정말 떠나시겠습니까?\n예\n아니요"


def test_round_17_tightly_set_furigana_at_several_sizes():
    from screen_translator.text import assemble

    W, H = 1280, 720
    for size in (28, 36, 44):
        half = size // 2
        rows = [px("ハルカ", 100, 440, size * 3, size, W, H), px("せんせい ほんとう", 100, 500, size * 4, half + 6, W, H),
                px("先生、本当にありがとうございました。", 100, 500 + half - 6, size * 18, size + 8, W, H)]
        assert assemble(rows, width=W, height=H, menus=True) == "ハルカ\n先生、本当にありがとうございました。"


def test_round_18_menus_stay_apart_in_both_modes_and_family_titles_call():
    from screen_translator.text import assemble
    from screen_translator.vi_numbers import spoken_numbers

    W, H = 1920, 1080
    pause = [px("ポーズ", 860, 200, 200, 60, W, H), px("ゲームに戻る", 840, 400, 240, 40, W, H), px("設定", 920, 468, 80, 40, W, H),
             px("操作方法", 880, 536, 160, 40, W, H), px("タイトルに戻る", 820, 604, 280, 40, W, H)]
    assert [b.text for b in group_blocks(pause, W, H)] == ["ポーズ", "ゲームに戻る", "設定", "操作方法", "タイトルに戻る"]
    W, H = 1280, 720
    menu = [px("What will you do?", 60, 400, 320, 36, W, H), px("Attack", 100, 448, 120, 36, W, H),
            px("Defend", 100, 496, 120, 36, W, H), px("Use item", 100, 544, 150, 36, W, H)]
    assert assemble(menu, width=W, height=H, menus=True) == "What will you do?\nAttack\nDefend\nUse item"
    call = [px("Mom", 600, 560, 80, 40, W, H), px("I'm home.", 560, 610, 160, 44, W, H)]
    assert assemble(call, width=W, height=H, menus=True) == "Mom, I'm home."
    assert spoken_numbers("3:45 PM, 12 AM") == "ba giờ bốn mươi lăm chiều, mười hai giờ đêm"
