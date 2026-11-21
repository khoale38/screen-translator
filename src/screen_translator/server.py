from __future__ import annotations

import asyncio
import json
import sys
import threading
from collections import deque
from importlib import resources
from typing import Callable

from aiohttp import WSMsgType, web

from . import languages
from .capture import ScreenCapture, ensure_screen_permission
from .config import SettingsStore
from .models import DOWNLOADS, MODELS, SPEECH, DownloadCancelled, NotEnoughSpace, check_space, download_all
from .ocr import RapidOcr, default_engine
from .pipeline import Pipeline

WEB = resources.files("screen_translator") / "web"


class Hub:
    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop | None = None
        self.sockets: set[web.WebSocketResponse] = set()
        self.lines: deque[dict] = deque(maxlen=30)
        self.status: dict = {"type": "status", "state": "stopped", "error": ""}
        self.download: dict | None = None
        self.screen: dict | None = None
        self.screen_texts: dict[str, str] = {}
        self.listeners: list[Callable[[dict], None]] = []

    def emit(self, event: dict) -> None:
        for listener in self.listeners:
            listener(event)
        if event["type"] == "line":
            if self.lines and self.lines[-1]["id"] == event["id"]:
                self.lines[-1] = event
            else:
                self.lines.append(event)
        elif event["type"] == "status":
            self.status = event
        elif event["type"] == "download":
            self.download = event
        elif event["type"] == "screen":
            self.screen = event
        elif event["type"] == "screen_text" and event["final"]:
            if len(self.screen_texts) >= 500:
                self.screen_texts.clear()
            self.screen_texts[event["source"]] = event["translation"]
        cleared = event["type"] == "clear" and event.get("area") is None
        if cleared or (event["type"] == "status" and event["state"] == "stopped"):
            self.screen, self.screen_texts = None, {}
        if self.loop:
            self.loop.call_soon_threadsafe(self._schedule, json.dumps(event, ensure_ascii=False))

    def screen_snapshot(self) -> dict | None:
        """The current whole-screen layout with every translation known so far, for a newly connected overlay."""
        if not self.screen:
            return None
        blocks = [{**b, "translation": b.get("translation") or self.screen_texts.get(b["text"])}
                  for b in self.screen["blocks"]]
        return {**self.screen, "blocks": blocks}

    def _schedule(self, payload: str) -> None:
        for ws in list(self.sockets):
            asyncio.ensure_future(self._send(ws, payload))

    async def _send(self, ws: web.WebSocketResponse, payload: str) -> None:
        try:
            await ws.send_str(payload)
        except ConnectionError:
            self.sockets.discard(ws)


def create_app(
    store: SettingsStore,
    pipeline_factory: Callable[[SettingsStore, Callable[[dict], None]], Pipeline] = Pipeline,
    capture_factory: Callable[[], ScreenCapture] = ScreenCapture,
    on_event: Callable[[dict], None] | None = None,
    voices_factory: Callable[[], object] | None = None,
) -> web.Application:
    hub = Hub()
    installed_voices: list = []
    if on_event:
        hub.listeners.append(on_event)
    pipeline = pipeline_factory(store, hub.emit)
    cancel_download = threading.Event()
    ocr_hint = "" if sys.platform == "darwin" else RapidOcr.hint
    allowed_hosts = {f"127.0.0.1:{store.settings.port}", f"localhost:{store.settings.port}"}

    @web.middleware
    async def local_only(request: web.Request, handler):
        if request.host not in allowed_hosts:
            raise web.HTTPForbidden(text="Host not allowed")
        origin = request.headers.get("Origin")
        if origin and origin.split("://", 1)[-1] not in allowed_hosts:
            raise web.HTTPForbidden(text="Origin not allowed")
        return await handler(request)

    app = web.Application(middlewares=[local_only])

    async def on_startup(_: web.Application) -> None:
        hub.loop = asyncio.get_running_loop()

    async def on_shutdown(_: web.Application) -> None:
        cancel_download.set()
        await asyncio.gather(*(ws.close() for ws in list(hub.sockets)), return_exceptions=True)
        await asyncio.to_thread(pipeline.shutdown)

    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)

    def page(name: str):
        async def handler(_: web.Request) -> web.Response:
            return web.Response(text=(WEB / name).read_text(encoding="utf-8"), content_type="text/html")
        return handler

    async def static(request: web.Request) -> web.Response:
        name = request.match_info["name"]
        kinds = {"app.css": "text/css", "panel.js": "application/javascript",
                 "overlay.js": "application/javascript", "icon.png": "image/png"}
        if name not in kinds:
            raise web.HTTPNotFound()
        return web.Response(body=(WEB / name).read_bytes(), content_type=kinds[name])

    def state() -> dict:
        return {
            "settings": store.settings.to_dict(),
            "status": hub.status,
            "running": pipeline.running,
            "languages": languages.LANGUAGES,
            "ocr_hint": ocr_hint,
            "models": [
                {"key": m.key, "label": m.label, "size": m.size_bytes, "downloaded": m.downloaded}
                for m in MODELS.values()
            ],
            "speech": {"label": SPEECH.label, "size": SPEECH.size_bytes, "downloaded": SPEECH.downloaded},
            "download": hub.download,
            "lines": list(hub.lines),
        }

    async def get_state(_: web.Request) -> web.Response:
        return web.json_response(state())

    async def update_settings(request: web.Request) -> web.Response:
        patch = await request.json()
        patch.pop("port", None)
        if "region" in patch:
            # The browser panel picks one area as a fraction of a monitor; it replaces the app window's areas.
            patch.update(capture_rects=[], capture_rect=None, scope="area")
        settings = store.update(patch)
        pipeline.apply_settings(settings)
        return web.json_response(state())

    async def start(_: web.Request) -> web.Response:
        try:
            await asyncio.to_thread(pipeline.start)
        except (FileNotFoundError, PermissionError) as exc:
            return web.json_response({"error": str(exc)}, status=409)
        return web.json_response(state())

    async def stop(_: web.Request) -> web.Response:
        await asyncio.to_thread(pipeline.stop)
        return web.json_response(state())

    async def clear_context(_: web.Request) -> web.Response:
        pipeline.clear_context()
        return web.json_response({"ok": True})

    def with_capture(fn):
        capture = capture_factory()
        try:
            return fn(capture)
        finally:
            capture.close()

    async def voices(_: web.Request) -> web.Response:
        from .dub import default_voices, voice_choices

        def listing() -> list[dict]:
            if not installed_voices:
                installed_voices.extend((voices_factory or default_voices)().voices)
            return [{"value": value, "label": label}
                    for value, label in voice_choices(installed_voices, store.settings.target_lang)]

        try:
            return web.json_response({"voices": await asyncio.to_thread(listing)})
        except Exception as exc:
            return web.json_response({"voices": [], "error": f"Could not list voices: {exc}"})

    async def monitors(_: web.Request) -> web.Response:
        return web.json_response(await asyncio.to_thread(with_capture, lambda c: c.monitors()))

    async def screenshot(request: web.Request) -> web.Response:
        try:
            ensure_screen_permission()
        except PermissionError as exc:
            raise web.HTTPConflict(text=str(exc))
        monitor = int(request.query.get("monitor", store.settings.monitor))
        body = await asyncio.to_thread(with_capture, lambda c: c.jpeg(monitor))
        return web.Response(body=body, content_type="image/jpeg", headers={"Cache-Control": "no-store"})

    async def test_ocr(_: web.Request) -> web.Response:
        def run() -> dict:
            from .text import assemble

            ensure_screen_permission()
            s = store.settings
            engine = default_engine()
            frame = with_capture(lambda c: c.grab(s.monitor, s.region))
            return {"engine": engine.name, "text": assemble(engine.recognize(frame, s.source_lang))}

        try:
            return web.json_response(await asyncio.to_thread(run))
        except PermissionError as exc:
            return web.json_response({"error": str(exc)}, status=409)

    async def start_download(request: web.Request) -> web.Response:
        spec = DOWNLOADS.get(request.match_info["key"])
        if spec is None:
            raise web.HTTPNotFound()
        if hub.download and hub.download.get("active"):
            return web.json_response({"error": "A download is already running."}, status=409)
        try:
            check_space(spec)
        except NotEnoughSpace as exc:
            hub.emit({"type": "download", "key": spec.key, "active": False, "error": str(exc)})
            return web.json_response({"error": str(exc)}, status=409)
        cancel_download.clear()

        def run() -> None:
            def progress(done: int, total: int) -> None:
                hub.emit({"type": "download", "key": spec.key, "done": done, "total": total, "active": True})

            try:
                download_all(spec, progress, cancel_download)
                hub.emit({"type": "download", "key": spec.key, "done": spec.size_bytes,
                          "total": spec.size_bytes, "active": False, "finished": True})
            except DownloadCancelled:
                hub.emit({"type": "download", "key": spec.key, "active": False, "error": "Cancelled"})
            except Exception as exc:
                hub.emit({"type": "download", "key": spec.key, "active": False, "error": str(exc)})

        hub.emit({"type": "download", "key": spec.key, "done": 0, "total": spec.size_bytes, "active": True})
        threading.Thread(target=run, name="download", daemon=True).start()
        return web.json_response({"ok": True})

    async def stop_download(_: web.Request) -> web.Response:
        cancel_download.set()
        return web.json_response({"ok": True})

    async def index(_: web.Request) -> web.Response:
        raise web.HTTPFound("/panel")

    async def websocket(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=20, timeout=2)
        await ws.prepare(request)
        hub.sockets.add(ws)
        await ws.send_str(json.dumps(hub.status))
        for line in hub.lines:
            await ws.send_str(json.dumps(line, ensure_ascii=False))
        if screen := hub.screen_snapshot():
            await ws.send_str(json.dumps(screen, ensure_ascii=False))
        try:
            async for msg in ws:
                if msg.type == WSMsgType.ERROR:
                    break
        finally:
            hub.sockets.discard(ws)
        return ws

    app.add_routes([
        web.get("/", index),
        web.get("/panel", page("panel.html")),
        web.get("/overlay", page("overlay.html")),
        web.get("/static/{name}", static),
        web.get("/ws", websocket),
        web.get("/api/state", get_state),
        web.post("/api/settings", update_settings),
        web.post("/api/start", start),
        web.post("/api/stop", stop),
        web.post("/api/context/clear", clear_context),
        web.get("/api/monitors", monitors),
        web.get("/api/voices", voices),
        web.get("/api/screenshot", screenshot),
        web.post("/api/ocr/test", test_ocr),
        web.post("/api/models/{key}/download", start_download),
        web.post("/api/models/download/cancel", stop_download),
    ])
    return app


def serve(
    store: SettingsStore,
    ready: threading.Event | None = None,
    stop: threading.Event | None = None,
    on_event: Callable[[dict], None] | None = None,
) -> None:
    async def main() -> None:
        runner = web.AppRunner(create_app(store, on_event=on_event))
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", store.settings.port).start()
        if ready:
            ready.set()
        try:
            while not (stop and stop.is_set()):
                await asyncio.sleep(0.25)
        finally:
            await runner.cleanup()

    asyncio.run(main())
