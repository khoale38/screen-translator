from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from statistics import median
from difflib import SequenceMatcher

from .ocr import OcrLine

_SPACELESS = re.compile(r"[぀-ヿ㐀-鿿豈-﫿ｦ-ﾟ฀-๿　-〿！-｠]")
_SPACES = re.compile(r"[^\S\n]+")
_BREAKS = re.compile(r" ?\n[\s]*")
_DECORATION = "•·●○◦◆◇■□▪▫►▶◀◄▲▼★☆♦♥♠♣※→←↑↓|_=~"
# The L1/R1 button icon comes out of OCR as "LI", "Lt", "L|" or "L!"; "Lt." stays, as that is lieutenant.
_BUTTON_MISREAD = re.compile(r"(?<![A-Za-z])([LR])[Il|t!](?![A-Za-z0-9.])")


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


def clean_lines(lines: list[OcrLine], min_confidence: float = 0.3) -> list[OcrLine]:
    """Drop HUD icons and dot rows that OCR reads as text: keep lines with at least two letters, minus edge symbols."""
    kept = []
    for line in lines:
        text = _BUTTON_MISREAD.sub(r"\g<1>1", line.text.strip().strip(_DECORATION + " "))
        if line.confidence >= min_confidence and sum(ch.isalpha() for ch in text) >= 2:
            kept.append(OcrLine(text, line.confidence, line.box))
    return kept


def assemble(lines: list[OcrLine], min_confidence: float = 0.3) -> str:
    kept = clean_lines(lines, min_confidence)
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
    ratio = min(above.box[3], below.box[3]) / height
    # OCR boxes vary in height for the same font, so a size change alone only counts when it is large or spaced out.
    if gap > 0.6 * height or ratio < 0.6 or (ratio < 0.75 and gap > 0.3 * height):
        return True
    is_label = len(above_text) <= 12 and above_width < 0.4 * below_width
    return is_label and gap > 0.15 * height


@dataclass
class TextEvent:
    kind: str
    text: str = ""
    extends_previous: bool = False
    area: int = 0
    line_height: float = 0.0


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


@dataclass
class Block:
    """A paragraph or label found anywhere on screen; box and line_height are fractions of the frame."""

    text: str
    box: tuple[float, float, float, float]
    line_height: float
    bg: str = "#000000"
    fg: str = "#ffffff"


def group_blocks(lines: list[OcrLine], width: int = 1, height: int = 1, min_confidence: float = 0.3) -> list[Block]:
    kept = clean_lines(lines, min_confidence)
    pixels = [(l.box[0] * width, l.box[1] * height, l.box[2] * width, l.box[3] * height) for l in kept]
    parent = list(range(len(kept)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(kept)):
        for j in range(i + 1, len(kept)):
            if _same_block(pixels[i], pixels[j]):
                parent[root(i)] = root(j)

    groups: dict[int, list[int]] = {}
    for i in range(len(kept)):
        groups.setdefault(root(i), []).append(i)

    blocks = []
    for members in groups.values():
        group = [kept[i] for i in members]
        left = min(l.box[0] for l in group)
        top = min(l.box[1] for l in group)
        right = max(l.box[0] + l.box[2] for l in group)
        bottom = max(l.box[1] + l.box[3] for l in group)
        blocks.append(Block(normalize(assemble(group, min_confidence)), (left, top, right - left, bottom - top),
                            median(l.box[3] for l in group)))
    blocks.sort(key=lambda b: (round(b.box[1] * 50), b.box[0]))
    return blocks


def _same_block(a: tuple, b: tuple) -> bool:
    """Words split on one row, or wrapped rows of one paragraph; never side-by-side columns or differently sized text."""
    height = max(a[3], b[3])
    ratio = min(a[3], b[3]) / height
    if abs((a[1] + a[3] / 2) - (b[1] + b[3] / 2)) < height / 2:
        gap = max(a[0], b[0]) - min(a[0] + a[2], b[0] + b[2])
        return ratio >= 0.7 and gap < 0.8 * height
    upper, lower = (a, b) if a[1] < b[1] else (b, a)
    gap = lower[1] - (upper[1] + upper[3])
    overlap = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    same_size = ratio >= 0.75 or (ratio >= 0.6 and gap <= 0.3 * height)
    return same_size and -0.2 * height <= gap <= 0.6 * height and overlap >= 0.3 * min(a[2], b[2])
