from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

from .config import SettingsStore
from .server import serve


class DesktopApi:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url
        self._floating = None

    def open_floating(self) -> None:
        import webview

        if self._floating is not None:
            return
        transparent = sys.platform != "win32"
        url = f"{self._base_url}/overlay?lines=2&size=28" + ("" if transparent else "&pagebg=101218")
        self._floating = webview.create_window(
            "Subtitles", url, width=900, height=180, frameless=True, easy_drag=True, on_top=True,
            transparent=transparent, background_color="#101218",
        )
        self._floating.events.closed += lambda: setattr(self, "_floating", None)


def run_desktop(base_url: str, stop: threading.Event) -> bool:
    try:
        import webview
    except ImportError:
        return False
    webview.create_window(
        "Screen Translator", f"{base_url}/panel", js_api=DesktopApi(base_url), width=520, height=900,
    )
    webview.start()
    stop.set()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(prog="screen-translator", description="Real-time on-screen translation.")
    parser.add_argument("--port", type=int, help="port for the control panel and OBS overlay (default 8765)")
    parser.add_argument("--headless", action="store_true", help="run only the server, e.g. when started from OBS")
    parser.add_argument("--browser", action="store_true", help="open the control panel in your web browser")
    parser.add_argument("--prefetch-ocr", metavar="DIR", type=Path, help="download the default RapidOCR models into DIR and exit")
    args = parser.parse_args()

    if args.prefetch_ocr:
        from .ocr import prefetch_rapid_models

        args.prefetch_ocr.mkdir(parents=True, exist_ok=True)
        print("\n".join(prefetch_rapid_models(args.prefetch_ocr)))
        return

    store = SettingsStore()
    if args.port:
        store.update({"port": args.port})
    base_url = f"http://127.0.0.1:{store.settings.port}"

    ready, stop = threading.Event(), threading.Event()
    server = threading.Thread(target=serve, args=(store, ready, stop), name="server", daemon=True)
    server.start()
    if not ready.wait(15):
        raise SystemExit(f"Could not start the server on {base_url}; is the port already in use?")
    print(f"Control panel: {base_url}/panel\nOBS overlay:   {base_url}/overlay", flush=True)

    if not args.headless and not args.browser and run_desktop(base_url, stop):
        server.join(5)
        return
    if args.browser:
        webbrowser.open(f"{base_url}/panel")
    try:
        server.join()
    except KeyboardInterrupt:
        stop.set()
        server.join(5)


if __name__ == "__main__":
    main()
