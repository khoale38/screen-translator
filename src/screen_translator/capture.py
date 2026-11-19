from __future__ import annotations

import io
import sys
from dataclasses import dataclass

import mss
import numpy as np
from PIL import Image

from .config import Region


@dataclass
class Frame:
    bgra: bytes
    width: int
    height: int

    def array(self) -> np.ndarray:
        return np.frombuffer(self.bgra, dtype=np.uint8).reshape(self.height, self.width, 4)


class ScreenCapture:
    """Wraps one mss instance; create it on the thread that grabs, since mss handles are not thread-safe."""

    def __init__(self) -> None:
        self._sct = mss.mss()

    def monitors(self) -> list[dict]:
        return [dict(m, index=i) for i, m in enumerate(self._sct.monitors) if i > 0]

    def _bounds(self, monitor: int, region: Region | None) -> dict:
        monitors = self._sct.monitors
        mon = monitors[monitor] if 0 < monitor < len(monitors) else monitors[1]
        if region is None:
            return {k: mon[k] for k in ("left", "top", "width", "height")}
        return {
            "left": mon["left"] + round(region.left * mon["width"]),
            "top": mon["top"] + round(region.top * mon["height"]),
            "width": max(8, round(region.width * mon["width"])),
            "height": max(8, round(region.height * mon["height"])),
        }

    def grab(self, monitor: int, region: Region | None) -> Frame:
        shot = self._sct.grab(self._bounds(monitor, region))
        return Frame(bytes(shot.bgra), shot.width, shot.height)

    def jpeg(self, monitor: int, max_width: int = 1600) -> bytes:
        frame = self.grab(monitor, None)
        image = Image.frombuffer("RGB", (frame.width, frame.height), frame.bgra, "raw", "BGRX")
        if image.width > max_width:
            image = image.resize((max_width, round(image.height * max_width / image.width)))
        buf = io.BytesIO()
        image.save(buf, "JPEG", quality=80)
        return buf.getvalue()

    def close(self) -> None:
        self._sct.close()


SCREEN_PERMISSION_HELP = (
    "macOS is blocking screen capture, so only the wallpaper is visible. Allow this app (or OBS / your terminal) in "
    "System Settings → Privacy & Security → Screen & System Audio Recording, then restart it."
)


def ensure_screen_permission() -> None:
    if sys.platform != "darwin":
        return
    import Quartz

    if not Quartz.CGPreflightScreenCaptureAccess():
        Quartz.CGRequestScreenCaptureAccess()
        raise PermissionError(SCREEN_PERMISSION_HELP)


def thumbnail(frame: Frame) -> np.ndarray:
    return frame.array()[::4, ::4, 1].astype(np.int16)


def changed(previous: np.ndarray | None, current: np.ndarray, min_fraction: float = 0.0005) -> bool:
    if previous is None or previous.shape != current.shape:
        return True
    moved = np.count_nonzero(np.abs(previous - current) > 24)
    return moved / current.size > min_fraction
