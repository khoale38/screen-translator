from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

from .ocr import OcrLine

_SPACELESS = re.compile(r"[぀-ヿ㐀-鿿豈-﫿ｦ-ﾟ฀-๿　-〿！-｠]")
_SPACES = re.compile(r"[^\S\n]+")
_BREAKS = re.compile(r" ?\n[\s]*")


def normalize(text: str) -> str:
    return _BREAKS.sub("\n", _SPACES.sub(" ", unicodedata.normalize("NFKC", text))).strip()


def similar(a: str, b: str, threshold: float = 0.9) -> bool:
    if not a or not b:
        return a == b
    return SequenceMatcher(None, a, b, autojunk=False).ratio() >= threshold


def _join(left: str, right: str) -> str:
    if not left:
        return right
    if _SPACELESS.match(left[-1]) and _SPACELESS.match(right[0]):
        return left + right
    return f"{left} {right}"


def assemble(lines: list[OcrLine], min_confidence: float = 0.3) -> str:
    kept = [line for line in lines if line.confidence >= min_confidence and line.text.strip()]
    if not kept:
        return ""

    kept.sort(key=lambda line: line.box[1] + line.box[3] / 2)
    rows: list[list[OcrLine]] = []
    for line in kept:
        center = line.box[1] + line.box[3] / 2
        if rows:
            anchor = rows[-1][0]
            if abs(center - (anchor.box[1] + anchor.box[3] / 2)) < anchor.box[3] / 2:
                rows[-1].append(line)
                continue
        rows.append([line])

    paragraphs: list[str] = []
    previous: tuple[str, OcrLine, float] | None = None
    for row in rows:
        ordered = sorted(row, key=lambda item: item.box[0])
        row_text = ""
        for line in ordered:
            row_text = _join(row_text, line.text.strip())
        width = ordered[-1].box[0] + ordered[-1].box[2] - ordered[0].box[0]
        if previous is not None and not _starts_new_paragraph(*previous, row[0], width):
            paragraphs[-1] = _join(paragraphs[-1], row_text)
        else:
            paragraphs.append(row_text)
        previous = (row_text, row[0], width)
    return "\n".join(paragraphs)


def _starts_new_paragraph(above_text: str, above: OcrLine, above_width: float, below: OcrLine, below_width: float) -> bool:
    """Wrapped text fills its first row, so a wide gap, a size change or a short row over a wide one is a new line."""
    height = max(above.box[3], below.box[3])
    gap = below.box[1] - (above.box[1] + above.box[3])
    if gap > 0.6 * height or min(above.box[3], below.box[3]) / height < 0.75:
        return True
    is_label = len(above_text) <= 12 and above_width < 0.4 * below_width
    return is_label and gap > 0.15 * height


@dataclass
class TextEvent:
    kind: str
    text: str = ""
    extends_previous: bool = False


class Stabilizer:
    """Emits text only after it stops changing, so typewriter-style dialogue is translated once, complete."""

    def __init__(self, stable_ticks: int = 2, min_chars: int = 2) -> None:
        self.stable_ticks = stable_ticks
        self.min_chars = min_chars
        self._candidate = ""
        self._count = 0
        self._empty = 0
        self._emitted = ""

    def reset(self) -> None:
        self._candidate = ""
        self._count = 0
        self._empty = 0
        self._emitted = ""

    def feed(self, text: str) -> TextEvent | None:
        norm = normalize(text)
        if len(norm) < self.min_chars:
            self._candidate, self._count = "", 0
            self._empty += 1
            if self._emitted and self._empty >= self.stable_ticks:
                self._emitted = ""
                return TextEvent("clear")
            return None

        self._empty = 0
        if norm == self._candidate:
            self._count += 1
        else:
            self._candidate, self._count = norm, 1

        if self._count < self.stable_ticks or similar(norm, self._emitted, 0.92):
            return None

        extends = bool(self._emitted) and norm.startswith(self._emitted)
        self._emitted = norm
        return TextEvent("text", norm, extends)
