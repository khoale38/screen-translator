from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from statistics import median
from difflib import SequenceMatcher

from .ocr import BUTTON_MISREAD, OcrLine

_SPACELESS = re.compile(r"[぀-ヿ㐀-鿿豈-﫿ｦ-ﾟ฀-๿　-〿！-｠]")
_SPACES = re.compile(r"[^\S\n]+")
_BREAKS = re.compile(r" ?\n[\s]*")
_DECORATION = "•·●○◦◆◇■□▪▫►▶◀◄▲▼★☆♦♥♠♣※→←↑↓|_=~"


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


def clean_lines(lines: list[OcrLine], min_confidence: float = 0.3, lone_cjk: bool = False) -> list[OcrLine]:
    """Drop HUD icons and dot rows that OCR reads as text: keep lines with at least two letters, minus edge symbols.

    With `lone_cjk`, a single CJK character is kept too, such as the short last column of a manga bubble (だ。).
    """
    kept = []
    for line in lines:
        text = BUTTON_MISREAD.sub(r"\g<1>1", line.text.strip().strip(_DECORATION + " "))
        letters = sum(ch.isalpha() for ch in text)
        if line.confidence >= min_confidence and (letters >= 2 or (lone_cjk and letters and _SPACELESS.search(text))):
            kept.append(OcrLine(text, line.confidence, line.box))
    return kept


def is_vertical(line: OcrLine, width: int = 1, height: int = 1) -> bool:
    """A column of CJK text, as in manga and light novels: much taller than wide, with more than one character."""
    return len(line.text.strip()) >= 2 and line.box[3] * height >= 1.5 * line.box[2] * width


def char_size(line: OcrLine, width: int = 1, height: int = 1) -> float:
    """The size of one character as a fraction of the frame's height: a row's height, or a column's width."""
    return line.box[2] * width / height if is_vertical(line, width, height) else line.box[3]


def assemble(lines: list[OcrLine], min_confidence: float = 0.3, width: int = 1, height: int = 1,
             lone_cjk: bool = False, vertical: bool | None = None) -> str:
    kept = clean_lines(lines, min_confidence, lone_cjk)
    if not kept:
        return ""
    if vertical is None:
        vertical = sum(is_vertical(line, width, height) for line in kept) * 2 > len(kept)
    if vertical:
        text = ""
        for line in sorted(kept, key=lambda line: -(line.box[0] + line.box[2])):
            text = _join(text, line.text.strip())
        return text

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
    if _ENTRY_START.match(below.text):
        return True
    # A name box over its dialogue: OCR pads boxes, so the name often overlaps the line below it.
    return _name_like(above_text) and above_width < 0.6 * below_width and gap > -0.5 * height


def _name_like(text: str) -> bool:
    """A speaker's name box rather than the short first line of a speech bubble ("WHERE ARE", "ねえ、")."""
    text = text.strip()
    words = text.split()
    if not 0 < len(text) <= 20 or len(words) > 3 or text.endswith(_ENDS) or _MID_PHRASE.search(text):
        return False
    if re.search("[\u3041-\u309f]", text):
        return False
    capitalised = all(word[0].isupper() or not word[0].isascii() for word in words)
    return capitalised and not (text.isupper() and len(words) > 1)


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
        typing = norm != self._candidate and norm.startswith(self._candidate)
        # Over a moving game scene the same subtitle reads a character or two differently each time.
        if self._candidate and not typing and similar(norm, self._candidate, 0.85):
            self._count += 1
        else:
            self._count = 1
        self._candidate = norm

        if self._count < self.stable_ticks or similar(norm, self._emitted, 0.92):
            if self._count == 1 and not similar(norm, self._emitted, 0.92):
                # Worth translating already; it only counts once a later read confirms it.
                return TextEvent("preview", norm)
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
    vertical: bool = False


def _orientation(line: OcrLine, width: int, height: int) -> str:
    """"v" for a column, "h" for a row, "?" for one or two CJK characters that could belong to either."""
    if is_vertical(line, width, height):
        return "v"
    text = line.text.strip()
    aspect = line.box[3] * height / max(1e-9, line.box[2] * width)
    return "?" if len(text) <= 2 and _SPACELESS.search(text) and 0.6 <= aspect <= 1.6 else "h"


def _ruby(pixels: list[tuple]) -> dict[int, int]:
    """Furigana: a thin column hugging a wider one, whose reading it spells out; maps each to the column it reads."""
    found = {}
    for i, a in enumerate(pixels):
        if a[4] != "v":
            continue
        for j, b in enumerate(pixels):
            if i == j or b[4] != "v" or a[2] >= 0.6 * b[2]:
                continue
            gap = max(a[0], b[0]) - min(a[0] + a[2], b[0] + b[2])
            if gap <= 0.5 * b[2] and a[1] >= b[1] - 0.5 * b[2] and a[1] + a[3] <= b[1] + b[3] + 0.5 * b[2]:
                found[i] = j
                break
    return found


def group_blocks(lines: list[OcrLine], width: int = 1, height: int = 1, min_confidence: float = 0.3) -> list[Block]:
    kept = clean_lines(lines, min_confidence, lone_cjk=True)
    pixels = [(l.box[0] * width, l.box[1] * height, l.box[2] * width, l.box[3] * height,
               _orientation(l, width, height), l.text.strip()) for l in kept]
    ruby = _ruby(pixels)
    parent = list(range(len(kept)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(kept)):
        for j in range(i + 1, len(kept)):
            if i not in ruby and j not in ruby and _same_block(pixels[i], pixels[j]):
                parent[root(i)] = root(j)
    for small, base in ruby.items():
        parent[root(small)] = root(base)

    groups: dict[int, list[int]] = {}
    for i in range(len(kept)):
        groups.setdefault(root(i), []).append(i)

    blocks = []
    for members in groups.values():
        group = [kept[i] for i in members]
        words = [kept[i] for i in members if i not in ruby]
        left = min(l.box[0] for l in group)
        top = min(l.box[1] for l in group)
        right = max(l.box[0] + l.box[2] for l in group)
        bottom = max(l.box[1] + l.box[3] for l in group)
        vertical = any(pixels[i][4] == "v" for i in members)
        text = normalize(assemble(words, min_confidence, width, height, lone_cjk=True, vertical=vertical))
        if sum(ch.isalpha() for ch in text) < 2:
            continue
        columns = [l for l in words if is_vertical(l, width, height)] or words
        blocks.append(Block(text, (left, top, right - left, bottom - top),
                            median(char_size(l, width, height) for l in columns), vertical=vertical))
    if sum(b.vertical for b in blocks) * 2 > len(blocks):
        return _manga_order(blocks)
    blocks.sort(key=lambda b: (round(b.box[1] * 50), b.box[0]))
    return blocks


def _manga_order(blocks: list[Block]) -> list[Block]:
    """Rows of bubbles that overlap in height, top to bottom, each read right to left, as manga is."""
    rows: list[list[Block]] = []
    for block in sorted(blocks, key=lambda b: b.box[1]):
        top, bottom = block.box[1], block.box[1] + block.box[3]
        row = rows[-1] if rows else None
        if row:
            row_top, row_bottom = min(b.box[1] for b in row), max(b.box[1] + b.box[3] for b in row)
            shared = min(bottom, row_bottom) - max(top, row_top)
            if shared >= 0.4 * min(block.box[3], row_bottom - row_top):
                row.append(block)
                continue
        rows.append([block])
    return [block for row in rows for block in sorted(row, key=lambda b: -b.box[0])]


_MID_PHRASE = re.compile(r"[、，,;:\-]$|[でにをはがのとへもや]$|\b(the|a|an|to|of|and|with|in|on|for|or|your|my)$", re.I)
_ENDS = (".", "!", "?", "。", "！", "？", "…", ":", "：")
_STAT_END = re.compile(r"\d[\d,./:%+\-]*\)?$")
_ENTRY_START = re.compile(r"^\s*(?:[-–－]\s|\[[^\]]{1,20}\]|[^\s:：\[\]]{1,24}:\s|[^\s:：\[\]]{1,12}：)")


def _cjk(text: str) -> bool:
    return bool(_SPACELESS.search(text))


def _same_block(a: tuple, b: tuple) -> bool:
    """Words split on one row, or wrapped rows of one paragraph; never side-by-side columns or differently sized text.

    Vertical text is the other way round: neighbouring columns of one speech bubble belong together.
    """
    if "v" in (a[4], b[4]):
        return a[4] in "v?" and b[4] in "v?" and _same_bubble(a, b)
    height = max(a[3], b[3])
    ratio = min(a[3], b[3]) / height
    if abs((a[1] + a[3] / 2) - (b[1] + b[3] / 2)) < height / 2:
        gap = max(a[0], b[0]) - min(a[0] + a[2], b[0] + b[2])
        # A wider hole on one row is usually a button icon or a gap between separate labels.
        return ratio >= 0.7 and gap < 0.6 * height
    upper, lower = (a, b) if a[1] < b[1] else (b, a)
    gap = lower[1] - (upper[1] + upper[3])
    overlap = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    same_size = ratio >= 0.75 or (ratio >= 0.6 and gap <= 0.3 * height)
    # Lines of one paragraph or speech bubble sit close together; more widely spaced rows join only when the upper
    # one stops mid-phrase, so menu items, list entries and chat lines stay apart.
    # A stat ending the upper row, or a tag, name, or dash starting the lower one, marks separate entries: HUD stacks,
    # chat lines, two speakers in one subtitle.
    text = upper[5].rstrip()
    if _STAT_END.search(text) or _ENTRY_START.match(lower[5]):
        return False
    mid_phrase = _MID_PHRASE.search(text) and not re.fullmatch(r"[A-Za-z]+", text)
    wraps = (gap <= 0.25 * height or mid_phrase or lower[5][:1].islower()
             or (_cjk(text) and len(text.replace(" ", "")) >= 8 and not text.endswith(_ENDS) and gap <= 0.35 * height))
    # Padded OCR boxes of tightly set or slightly rotated lines overlap by up to half a line.
    return wraps and same_size and -0.5 * height <= gap <= 0.6 * height and overlap >= 0.3 * min(a[2], b[2])


def _same_bubble(a: tuple, b: tuple) -> bool:
    size = max(a[2], b[2])
    gap = max(a[0], b[0]) - min(a[0] + a[2], b[0] + b[2])
    overlap = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return min(a[2], b[2]) >= 0.6 * size and -0.5 * size <= gap <= 0.8 * size and overlap >= 0.3 * min(a[3], b[3])
