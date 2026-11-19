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


def sub_frame(frame: Frame, left: int, top: int, right: int, bottom: int) -> Frame:
    pixels = frame.array()[max(0, top):max(0, bottom), max(0, left):max(0, right)]
    return Frame(pixels.tobytes(), pixels.shape[1], pixels.shape[0])


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

    def grab_rect(self, rect: dict) -> Frame:
        if sys.platform == "darwin" and _excluded_windows:
            frame = _grab_below_our_windows(rect)
            if frame is not None:
                return frame
        shot = self._sct.grab({k: int(rect[k]) for k in ("left", "top", "width", "height")})
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


_excluded_windows: set[int] = set()


def exclude_window_from_capture(window_number: int) -> None:
    """macOS only: capture what is underneath our overlays, so they never read their own text."""
    if window_number:
        _excluded_windows.add(window_number)


def _grab_below_our_windows(rect: dict) -> Frame | None:
    import Quartz

    # Our overlays all float at the same level, so capturing below the lowest visible one skips every one of them.
    on_screen = Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID)
    ours = [int(w["kCGWindowNumber"]) for w in on_screen or [] if int(w["kCGWindowNumber"]) in _excluded_windows]
    if not ours:
        return None
    window_number = ours[-1]
    bounds = Quartz.CGRectMake(rect["left"], rect["top"], rect["width"], rect["height"])
    image = Quartz.CGWindowListCreateImage(
        bounds, Quartz.kCGWindowListOptionOnScreenBelowWindow, window_number, Quartz.kCGWindowImageDefault,
    )
    if image is None:
        return None
    width, height = Quartz.CGImageGetWidth(image), Quartz.CGImageGetHeight(image)
    stride = Quartz.CGImageGetBytesPerRow(image)
    raw = bytes(Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(image)))
    if stride != width * 4:
        raw = np.frombuffer(raw, np.uint8).reshape(height, stride)[:, : width * 4].tobytes()
    return Frame(raw, width, height)


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


def block_colors(frame: Frame, box: tuple[float, float, float, float]) -> tuple[str, str]:
    """Background and text colour of a text box, so the translation can be painted to look like the original."""
    pixels = frame.array()
    x0 = max(0, int(box[0] * frame.width) - 2)
    y0 = max(0, int(box[1] * frame.height) - 2)
    x1 = min(frame.width, int((box[0] + box[2]) * frame.width) + 3)
    y1 = min(frame.height, int((box[1] + box[3]) * frame.height) + 3)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return "#000000", "#ffffff"
    patch = pixels[y0:y1, x0:x1, 2::-1]
    region = patch.reshape(-1, 3).astype(np.int32)
    border = np.concatenate([patch[0], patch[-1], patch[:, 0], patch[:, -1]]).astype(np.int32)
    background = np.median(border, axis=0)
    distance = np.abs(region - background).sum(axis=1)
    ink = region[distance > 0.5 * distance.max()] if distance.max() > 0 else region
    text = np.median(ink, axis=0)
    if np.abs(text - background).sum() < 120:
        text = np.array([20, 20, 20]) if background @ [0.299, 0.587, 0.114] > 140 else np.array([245, 245, 245])
    to_hex = lambda rgb: "#" + "".join(f"{int(c):02x}" for c in rgb)
    return to_hex(background), to_hex(text)
