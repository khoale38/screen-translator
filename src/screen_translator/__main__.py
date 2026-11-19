from __future__ import annotations

import argparse
import threading
import webbrowser
from pathlib import Path

from .config import SettingsStore
from .server import serve


def main() -> None:
    parser = argparse.ArgumentParser(prog="screen-translator", description="Real-time on-screen translation.")
    parser.add_argument("--port", type=int, help="port for the control panel and OBS overlay (default 8765)")
    parser.add_argument("--headless", action="store_true", help="run only the server, e.g. when started from OBS")
    parser.add_argument("--browser", action="store_true", help="use the web control panel instead of the desktop app")
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

    if not args.headless and not args.browser:
        try:
            from .desktop import run
        except ImportError:
            pass
        else:
            run(store)
            return

    base_url = f"http://127.0.0.1:{store.settings.port}"
    ready, stop = threading.Event(), threading.Event()
    server = threading.Thread(target=serve, args=(store, ready, stop), name="server", daemon=True)
    server.start()
    if not ready.wait(15):
        raise SystemExit(f"Could not start the server on {base_url}; is the port already in use?")
    print(f"Control panel: {base_url}/panel\nOBS overlay:   {base_url}/overlay", flush=True)
    if args.browser:
        webbrowser.open(f"{base_url}/panel")
    try:
        server.join()
    except KeyboardInterrupt:
        stop.set()
        server.join(5)


if __name__ == "__main__":
    main()
