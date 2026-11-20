import asyncio
import json
import socket

import pytest
from aiohttp.test_utils import TestClient, TestServer

from screen_translator import pipeline as pipeline_module
from screen_translator.capture import Frame
from screen_translator.config import SettingsStore
from screen_translator.ocr import OcrLine
from screen_translator.pipeline import Pipeline
from screen_translator.server import create_app


class FakeCapture:
    def __init__(self):
        self.tick = 0

    def grab(self, monitor, region):
        self.tick += 1
        return Frame(bytes([self.tick * 60 % 256]) * (16 * 16 * 4), 16, 16)

    def monitors(self):
        return [{"index": 1, "left": 0, "top": 0, "width": 1920, "height": 1080}]

    def close(self):
        pass


class ScriptedOcr:
    name = "fake"

    def __init__(self, script):
        self.script = list(script)

    def recognize(self, frame, lang):
        text = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        return [OcrLine(text, 0.99, (0.1, 0.8, 0.8, 0.1))] if text else []


class EchoTranslator:
    def __init__(self):
        self.calls = []

    def translate(self, messages, on_partial=None):
        self.calls.append(messages)
        result = f"EN({messages[-1]['content']})"
        if on_partial:
            on_partial(result[:3])
        return result


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_module, "resolve_model", lambda settings: "fake.gguf")
    s = SettingsStore(tmp_path / "settings.json")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    s.update({"capture_interval_s": 0.01, "stable_ticks": 2, "port": port})
    return s


def make_pipeline_factory(script, translator):
    def factory(store, emit):
        return Pipeline(
            store, emit,
            ocr_factory=lambda: ScriptedOcr(script),
            translator_factory=lambda settings: translator,
            capture_factory=FakeCapture,
            permission_check=lambda: None,
        )
    return factory


async def wait_for(predicate, timeout=5.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.02)


async def test_pipeline_translates_each_stable_line_once_with_growing_context(store):
    translator = EchoTranslator()
    script = ["一行目", "一行目", "一行目", "二行目", "二行目", "二行目", "二行目"]
    app = create_app(store, make_pipeline_factory(script, translator), FakeCapture)
    async with TestClient(TestServer(app, port=store.settings.port)) as client:
        events = []
        ws = await client.ws_connect("/ws")

        async def reader():
            async for msg in ws:
                events.append(json.loads(msg.data))

        task = asyncio.create_task(reader())
        assert (await client.post("/api/start", json={})).status == 200
        await wait_for(lambda: sum(1 for e in events if e["type"] == "line" and e["final"]) >= 2)
        await client.post("/api/stop", json={})
        await ws.close()
        task.cancel()

    finals = [e for e in events if e["type"] == "line" and e["final"]]
    assert [e["translation"] for e in finals] == ["EN(一行目)", "EN(二行目)"]
    assert [e["context_lines"] for e in finals] == [0, 1]
    assert len(translator.calls) == 2
    assert translator.calls[1][1:3] == [
        {"role": "user", "content": "一行目"}, {"role": "assistant", "content": "EN(一行目)"},
    ]
    assert any(e["type"] == "line" and not e["final"] for e in events)


async def test_pipeline_ignores_its_own_subtitles_when_they_are_captured(store):
    translator = EchoTranslator()
    script = ["猫だ", "猫だ", "猫だ", "EN(猫だ)", "EN(猫だ)", "EN(猫だ)", "EN(猫だ)"]
    app = create_app(store, make_pipeline_factory(script, translator), FakeCapture)
    async with TestClient(TestServer(app, port=store.settings.port)) as client:
        await client.post("/api/start", json={})
        await asyncio.sleep(0.4)
        await client.post("/api/stop", json={})
    assert len(translator.calls) == 1


async def test_settings_update_persists_and_ignores_port(store):
    app = create_app(store, make_pipeline_factory([""], EchoTranslator()), FakeCapture)
    async with TestClient(TestServer(app, port=store.settings.port)) as client:
        resp = await client.post("/api/settings", json={
            "target_lang": "vi", "port": 1, "region": {"left": 0.1, "top": 0.7, "width": 0.8, "height": 0.2},
        })
        body = await resp.json()
    assert body["settings"]["target_lang"] == "vi"
    assert body["settings"]["port"] == store.settings.port
    assert SettingsStore(store.path).settings.region.top == 0.7


async def test_rejects_other_websites_and_foreign_hosts(store):
    app = create_app(store, make_pipeline_factory([""], EchoTranslator()), FakeCapture)
    async with TestClient(TestServer(app, port=store.settings.port)) as client:
        evil = await client.post("/api/start", json={}, headers={"Origin": "https://evil.example"})
        rebind = await client.get("/api/state", headers={"Host": "evil.example:8765"})
        ok = await client.get("/api/state", headers={"Origin": f"http://127.0.0.1:{store.settings.port}"})
    assert evil.status == 403
    assert rebind.status == 403
    assert ok.status == 200


async def test_pages_are_served(store):
    app = create_app(store, make_pipeline_factory([""], EchoTranslator()), FakeCapture)
    async with TestClient(TestServer(app, port=store.settings.port)) as client:
        for path in ("/panel", "/overlay", "/static/panel.js", "/static/overlay.js", "/static/app.css",
                     "/static/icon.png"):
            assert (await client.get(path)).status == 200, path
        assert (await client.get("/static/../config.py")).status == 404


async def test_a_line_that_comes_back_reuses_its_translation(store):
    translator = EchoTranslator()
    script = ["猫だ"] * 3 + [""] * 3 + ["猫だ"] * 3
    events = []
    app = create_app(store, make_pipeline_factory(script, translator), FakeCapture, on_event=events.append)
    async with TestClient(TestServer(app, port=store.settings.port)) as client:
        await client.post("/api/start", json={})
        await wait_for(lambda: sum(1 for e in events if e["type"] == "line" and e["final"]) >= 2)
        await client.post("/api/stop", json={})
    finals = [e for e in events if e["type"] == "line" and e["final"]]
    assert [e["translation"] for e in finals] == ["EN(猫だ)", "EN(猫だ)"]
    assert len(translator.calls) == 1


async def test_lines_already_in_the_target_language_are_left_alone(store):
    store.update({"target_lang": "vi"})
    translator = EchoTranslator()
    events = []
    script = ["cất súng luôn kia a"] * 4
    app = create_app(store, make_pipeline_factory(script, translator), FakeCapture, on_event=events.append)
    async with TestClient(TestServer(app, port=store.settings.port)) as client:
        await client.post("/api/start", json={})
        await wait_for(lambda: any(e["type"] == "clear" and e.get("area") == 0 for e in events))
        await client.post("/api/stop", json={})
    assert translator.calls == []
    assert not any(e["type"] == "line" for e in events)


async def test_the_server_refuses_a_download_the_disk_cannot_hold(store, monkeypatch):
    from collections import namedtuple

    from screen_translator import models

    monkeypatch.setattr(models.shutil, "disk_usage", lambda path: namedtuple("U", "total used free")(1, 1, 1000))
    events = []
    app = create_app(store, make_pipeline_factory([""], EchoTranslator()), FakeCapture, on_event=events.append)
    async with TestClient(TestServer(app, port=store.settings.port)) as client:
        resp = await client.post("/api/models/qwen3.5-9b/download", json={})
        body = await resp.json()
    assert resp.status == 409 and "Not enough disk space" in body["error"]
    downloads = [e for e in events if e["type"] == "download"]
    assert downloads and not downloads[-1]["active"] and "Not enough disk space" in downloads[-1]["error"]


async def test_an_area_picked_in_the_browser_panel_replaces_the_app_windows_areas(store):
    store.update({"scope": "screen", "capture_rects": [{"left": 0, "top": 0, "width": 10, "height": 10}]})
    app = create_app(store, make_pipeline_factory([""], EchoTranslator()), FakeCapture)
    async with TestClient(TestServer(app, port=store.settings.port)) as client:
        await client.post("/api/settings", json={"region": {"left": 0.1, "top": 0.7, "width": 0.8, "height": 0.2}})
    assert store.settings.capture_rects == [] and store.settings.capture_rect is None
    assert store.settings.scope == "area" and store.settings.region.top == 0.7
