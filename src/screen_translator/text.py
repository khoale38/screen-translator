from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from statistics import median
from difflib import SequenceMatcher

from .ocr import BUTTON_MISREAD, OcrLine, _with_dashes
from .romanize import CHOICE_WORDS, ROLES_VI

_SPACELESS = re.compile(r"[぀-ヿ㐀-鿿豈-﫿ｦ-ﾟ฀-๿　-〿！-｠]")
_KANA_FIX = {"八": "ハ", "口": "ロ", "工": "エ", "力": "カ", "夕": "タ"}
_KANA_LOOKALIKE = re.compile("(?<=[ァ-ヴー])[八口工力夕](?=[ァ-ヴー])"
                             "|(?:(?<=^)|(?<=[【「『\\[(（\\s]))[八口工力夕](?=[ァ-ヴー]{1,5}(?:$|[】」』\\]):：)）\\s]))",
                             re.MULTILINE)
# "ミ力" is ミカ: a look-alike closing a short katakana name on its own or in a label.
_KANA_NAME_END = re.compile("(?:(?<=^)|(?<=[【「『\\[(（\\s]))([ァ-ヴー]{1,5})([八口工力夕])(?=$|[】」』\\]):：)）\\s])",
                            re.MULTILINE)
_SPACES = re.compile(r"[^\S\n]+")
_BREAKS = re.compile(r" ?\n[\s]*")
_DECORATION = "•·●○◦◆◇■□▪▫►▶◀◄▲▼★☆♦♥♠♣※→←↑↓|_=~"


def normalize(text: str) -> str:
    # NFKC turns Korean chat laughter (ㅋㅋ, ㅠㅠ) into conjoining jamo, which then render broken; those keep their form.
    parts = re.split("([\u3131-\u318e]+)", text)
    text = "".join(part if i % 2 else unicodedata.normalize("NFKC", part) for i, part in enumerate(parts))
    # RapidOCR reads katakana ハ, ロ, エ, カ, タ as look-alike kanji inside a katakana word or a short katakana name.
    text = _KANA_LOOKALIKE.sub(lambda m: _KANA_FIX[m.group(0)], text)
    text = _KANA_NAME_END.sub(lambda m: m.group(1) + _KANA_FIX[m.group(2)], text)
    # OCR reads a vertical "。" as a degree sign after Japanese and Chinese text ("から°").
    text = re.sub("(?<=[\u3040-\u30ff\u4e00-\u9fff])°(?![CF])", "。", text)
    return _BREAKS.sub("\n", _SPACES.sub(" ", text)).strip()


def similar(a: str, b: str, threshold: float = 0.9) -> bool:
    if not a or not b:
        return a == b
    return SequenceMatcher(None, a, b, autojunk=False).ratio() >= threshold


def _join(left: str, right: str) -> str:
    if not left:
        return right
    if _SPACELESS.match(left[-1]) and _SPACELESS.match(right[0]):
        return left + right
    if (_SPACELESS.match(left[-1]) and right[0].isdigit()) or (left[-1].isdigit() and _SPACELESS.match(right[0])):
        return left + right
    broken = re.search(r"([A-Za-z]{2,})-$", left)
    if broken and right[0].isalpha():
        word = broken.group(1).lower()
        if word in _COMPOUND_PREFIXES:
            return left + right
        # A cut-off sentence ("you-" / "Don't", "just-" / "never mind"), not a word broken across lines ("tomor-" / "row").
        same_case = right[0].islower() or (broken.group(1).isupper() and right[0].isupper())
        if word not in _DASH_WORDS and same_case:
            return left[:-1] + right
    return f"{left} {right}"


_COMPOUND_PREFIXES = {"ex", "self", "non", "co", "anti", "pre", "post", "semi", "half", "well", "cross", "mid", "multi",
                      "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"}
_DASH_WORDS = {"just", "was", "but", "and", "if", "so", "what", "wait", "it", "the", "you", "we", "no", "not", "is", "are",
               "to", "my", "he", "she", "they", "this", "that", "then", "or", "me", "him", "her", "us", "them"}


def clean_lines(lines: list[OcrLine], min_confidence: float = 0.3, lone_cjk: bool = False) -> list[OcrLine]:
    """Drop HUD icons and dot rows that OCR reads as text: keep lines with at least two letters, minus edge symbols.

    With `lone_cjk`, a single CJK character is kept too, such as the short last column of a manga bubble (だ。).
    """
    kept = []
    for line in lines:
        text = BUTTON_MISREAD.sub(r"\g<1>1", line.text.strip().strip(_DECORATION + " "))
        # OCR reads a trailing ellipsis as a run of zeros and o's ("0oo00o").
        if re.fullmatch("[0oO]{3,}", text) and "0" in text and re.search("[oO]", text):
            continue
        letters = sum(ch.isalpha() for ch in text)
        # In bubbles a lone CJK character or Hangul syllable, or a line of just a number ("30!"), is real text too.
        lone = lone_cjk and ((letters and re.search("[\u3040-\u9fff\uac00-\ud7a3]", text)) or re.search(r"\d", text))
        # "爸，" read as a box of its own beside "我回来了。" is a call, not a stray glyph.
        beside = letters == 1 and re.search("[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7a3]", text) and any(
            other is not line and sum(ch.isalpha() for ch in other.text) >= 2
            and (_same_row(line, other) or _stacked(line, other)) for other in lines)
        if line.confidence >= min_confidence and (letters >= 2 or lone or beside):
            kept.append(OcrLine(text, line.confidence, line.box))
    # A second one-character choice ("否") stacked under the first ("是") belongs as much as the first does.
    for line in lines:
        text = line.text.strip()
        if line.confidence >= min_confidence and len(text) == 1 and re.search("[\u3400-\u9fff\uac00-\ud7a3]", text) \
                and all(k.box != line.box for k in kept) and any(len(k.text) == 1 and _stacked(line, k) for k in kept):
            kept.append(OcrLine(text, line.confidence, line.box))
    return kept


# A one-character name tab is a hanzi or kanji role (姫, 爹, 王); 응, 네, 嗯, 好 open a subtitle instead.
_ONE_SYLLABLE = re.compile("(?:[\u3400-\u9fff\uac00-\ud7a3]|师父|师兄|师姐|妈妈|爸爸|哥哥|姐姐|奶奶|爷爷|老师|陛下|殿下|娘娘)"
                           "[，,。.！!？?、…~]?")
_TITLE_WORDS = {word for word in ROLES_VI if not re.search("[\u3400-\u9fff]{3}", word)} | {
    "Mom", "Mum", "Dad", "Mother", "Father", "Grandpa", "Grandma", "Granny", "Sis", "Bro", "MOM", "DAD", "GRANDMA",
    "GRANDPA", "SIS", "BRO", "先輩", "せんぱい", "선배", "선배님", "学长", "学姐", "Senpai", "姉ちゃん", "兄ちゃん", "母さん", "父さん",
    "姉さん", "兄さん", "兄貴", "누나", "오빠", "형",
    "언니", "엄마", "아빠", "할머니", "할아버지", "お兄ちゃん", "お姉ちゃん", "先生", "哥哥", "姐姐", "哥", "姐",
    "前辈", "前輩", "師匠", "Captain", "ママ", "パパ", "じいちゃん", "ばあちゃん"}
_CONTROLS = re.compile(r"(?:(?:AUTO|SKIP|LOG|MENU|SAVE|LOAD|Q\.?SAVE|Q\.?LOAD|HIDE|CONFIG|オート|スキップ|ログ|メニュー|"
                       r"セーブ|ロード|自动|跳过|记录|回顾|菜单|隐藏|存档|读档|자동|스킵|로그|메뉴)\s*)+", re.I)
_ONE_CHARACTER_NAME = re.compile("(?![嗯好啊哦是对喂嘿哈呃唉呀嘛吧呢哎噢欸行何])[\u3400-\u9fff]")


def _stacked(short: OcrLine, line: OcrLine) -> bool:
    """A one-character row right above or below another: a name tab ("姫"), a call ("哥！") or a reply ("嗯。")."""
    height = max(short.box[3], line.box[3])
    gap = max(line.box[1] - (short.box[1] + short.box[3]), short.box[1] - (line.box[1] + line.box[3]))
    centre = abs((short.box[0] + short.box[2] / 2) - (line.box[0] + line.box[2] / 2))
    return -0.5 * height <= gap <= 1.2 * height and (abs(short.box[0] - line.box[0]) < 2 * height or centre < 2 * height)


def _same_row(a: OcrLine, b: OcrLine, reach: float = 1.5) -> bool:
    shared = min(a.box[1] + a.box[3], b.box[1] + b.box[3]) - max(a.box[1], b.box[1])
    gap = max(a.box[0], b.box[0]) - min(a.box[0] + a.box[2], b.box[0] + b.box[2])
    return shared > 0.5 * min(a.box[3], b.box[3]) and gap < reach * max(a.box[3], b.box[3])


def is_vertical(line: OcrLine, width: int = 1, height: int = 1) -> bool:
    """A column of CJK text, as in manga and light novels: much taller than wide, with more than one character."""
    return len(line.text.strip()) >= 2 and line.box[3] * height >= 1.5 * line.box[2] * width


def char_size(line: OcrLine, width: int = 1, height: int = 1) -> float:
    """The size of one character as a fraction of the frame's height: a row's height, or a column's width."""
    return line.box[2] * width / height if is_vertical(line, width, height) else line.box[3]


def assemble(lines: list[OcrLine], min_confidence: float = 0.3, width: int = 1, height: int = 1,
             lone_cjk: bool = False, vertical: bool | None = None, menus: bool = False) -> str:
    kept = clean_lines(_with_dashes(lines, width, height), min_confidence, lone_cjk)
    if not kept:
        return ""
    if vertical is None:
        vertical = sum(is_vertical(line, width, height) for line in kept) * 2 > len(kept)
    if vertical:
        text = ""
        columns = sorted(kept, key=lambda line: -(line.box[0] + line.box[2]))
        for i, line in enumerate(columns):
            # A title alone in the first column ("ママ" / "猫がいない！") calls the listener.
            call = i == 1 and columns[0].text.strip() in _TITLE_WORDS
            text = (text + "、" + line.text.strip()) if call else _join(text, line.text.strip())
        return text

    kept = [line for i, line in enumerate(kept)
            if _ruby_row_base([(l.box[0] * width, l.box[1] * height, l.box[2] * width, l.box[3] * height, "h", l.text)
                               for l in kept], i) is None]
    if menus:
        kept = _without_buttons(kept)
        kept = _without_edge_labels(kept)
        listed_lines = {id(kept[i]) for i in _list_rows([(l.box[0] * width, l.box[1] * height, l.box[2] * width,
                                                          l.box[3] * height, "h", l.text.strip()) for l in kept],
                                                        width)}
        clusters = _columns(kept, width, height)
    else:
        listed_lines = set()
        clusters = [kept]
    paragraphs: list[str] = []
    spans: list[list[float]] = []
    first_row: OcrLine | None = None
    for cluster in clusters:
        found, found_spans, row = _paragraphs(cluster, width, height, menus, listed_lines)
        paragraphs += found
        spans += found_spans
        first_row = first_row or row
    # Buttons in a far corner above the dialogue ("Auto  Skip") go last, so the name over the line is read as its tab.
    if len(paragraphs) > 2:
        first, second = spans[0], spans[1]
        apart = second[3] - first[2] > 2 * first_row.box[3] and (first[0] > second[1] or first[1] < second[0])
        if apart and _name_like(paragraphs[1]):
            paragraphs.append(paragraphs.pop(0))
    return "\n".join(paragraphs)


def _without_buttons(kept: list[OcrLine]) -> list[OcrLine]:
    """Visual-novel and gacha buttons on the name row ("AUTO SKIP LOG") are not part of anyone's line."""
    def button(line: OcrLine) -> bool:
        return bool(_CONTROLS.fullmatch(line.text.strip()))

    def button_row(line: OcrLine) -> bool:
        text = line.text.strip()
        beside = [other for other in kept if other is not line and button(other) and _same_row(line, other, 4)]
        if not button(line):
            # A misread button ("스컵") between two real ones on their row is one of them.
            return (len(text.replace(" ", "")) <= 4 and len(beside) >= 2
                    and min(o.box[0] for o in beside) < line.box[0] < max(o.box[0] for o in beside))
        # Several buttons in one box or side by side; a lone セーブ in a stacked menu is an item.
        return len(text.split()) > 1 or bool(beside)
    return [line for line in kept if not button_row(line)]


def _without_edge_labels(kept: list[OcrLine]) -> list[OcrLine]:
    """Corner tags (S2 E5, 第12集), danmaku along the top and a bottom-corner "Press A" hint: small, far, at the edge."""
    main = [line for line in kept if line.box[3] > 0.75 * max(other.box[3] for other in kept)]
    if len(main) == len(kept):
        return kept

    def edge_label(line: OcrLine) -> bool:
        if line in main:
            return False
        gaps = [max(other.box[1] - line.box[1] - line.box[3], line.box[1] - other.box[1] - other.box[3])
                for other in main]
        if min(gaps) <= 1.5 * max(other.box[3] for other in main):
            return False
        top = line.box[1] < 0.2 and all(other.box[1] > line.box[1] for other in main)
        corner = line.box[1] + line.box[3] > 0.8 and line.box[0] + line.box[2] > 0.9
        return top or corner
    return [line for line in kept if not edge_label(line)]


def _columns(kept: list[OcrLine], width: float, height: float) -> list[list[OcrLine]]:
    """Text side by side with no horizontal overlap, such as a choice menu beside the dialogue, read one after another."""
    parent = list(range(len(kept)))

    def root(i: int) -> int:
        while parent[i] != i:
            i = parent[i]
        return i

    for i, a in enumerate(kept):
        for j, b in enumerate(kept[:i]):
            margin = 1.5 * max(a.box[3], b.box[3]) * height / width
            if min(a.box[0] + a.box[2], b.box[0] + b.box[2]) - max(a.box[0], b.box[0]) > -margin:
                parent[root(i)] = root(j)
    groups: dict[int, list[OcrLine]] = {}
    for i, line in enumerate(kept):
        groups.setdefault(root(i), []).append(line)
    return sorted(groups.values(), key=lambda group: min(line.box[1] for line in group))


def _paragraphs(kept: list[OcrLine], frame_w: float, frame_h: float, menus: bool,
                listed_lines: set[int]) -> tuple[list[str], list[list[float]], OcrLine | None]:
    kept = sorted(kept, key=lambda line: line.box[1] + line.box[3] / 2)
    rows: list[list[OcrLine]] = []
    for line in kept:
        center = line.box[1] + line.box[3] / 2
        if rows:
            anchor = rows[-1][0]
            if abs(center - (anchor.box[1] + anchor.box[3] / 2)) < anchor.box[3] / 2:
                rows[-1].append(line)
                continue
        rows.append([line])
    # Text far apart on one baseline is separate, such as a choice menu beside the dialogue.
    rows = [segment for row in rows for segment in _split_wide_gaps(row)]

    menu_left: float | None = None
    paragraphs: list[str] = []
    spans: list[list[float]] = []
    previous: tuple[str, OcrLine, float] | None = None
    for row in rows:
        ordered = sorted(row, key=lambda item: item.box[0])
        row_text = ""
        for line in ordered:
            row_text = _join(row_text, line.text.strip())
        width = ordered[-1].box[0] + ordered[-1].box[2] - ordered[0].box[0]
        same_baseline = previous is not None and abs(
            (row[0].box[1] + row[0].box[3] / 2) - (previous[1].box[1] + previous[1].box[3] / 2)) < row[0].box[3] / 2
        left, right, bottom = ordered[0].box[0], ordered[-1].box[0] + ordered[-1].box[2], row[0].box[1] + row[0].box[3]
        after_question = len(paragraphs) >= 2 and paragraphs[-2].rstrip()[-1:] in "?？"
        lone = (previous is not None and _ONE_SYLLABLE.fullmatch(previous[0].strip()) and not (menus and after_question)
                and not _subject_of(row_text)
                and previous[0].strip() not in ("나", "저", "我", "私", "俺", "僕")
                and paragraphs[-1] == previous[0] and _stacked(previous[1], row[0])
                and (len(row_text.strip()) > 2 or re.search("[\uac00-\ud7a3]", previous[0])))
        if lone and not same_baseline:
            # "哥" centred over "等等我！" calls the listener; only a role set flush left over its line names the speaker.
            left = abs(previous[1].box[0] - row[0].box[0])
            centre = abs((previous[1].box[0] + previous[1].box[2] / 2) - (row[0].box[0] + row[0].box[2] / 2))
            flush = left * frame_w < row[0].box[3] * frame_h and left < centre
            tab = _ONE_CHARACTER_NAME.fullmatch(previous[0].strip()) or len(previous[0].strip().strip("，,。.！!？?、…~")) > 1
            if not (flush and tab):
                ends = previous[0].strip()[-1:] in "，,。.！!？?、…~"
                comma = "" if ends else ", " if re.search("[\uac00-\ud7a3]", previous[0]) else "，"
                paragraphs[-1] = paragraphs[-1] + comma + row_text if not ends else _join(paragraphs[-1], row_text)
                spans[-1] = [min(spans[-1][0], left), max(spans[-1][1], right), bottom, spans[-1][3]]
                previous = (paragraphs[-1], row[0], width)
                continue
        asked = len(paragraphs) >= 2 and paragraphs[-2].rstrip()[-1:] in "?？" and paragraphs[-1] == previous[0]
        menu = menus and previous is not None and not same_baseline and (_choice_rows(
            previous, row[0], row_text, frame_w, frame_h, asked, menu_left) or id(row[0]) in listed_lines)
        menu_left = row[0].box[0] if menu else None
        # "DAD / NEVER CAME HOME THAT NIGHT": a family word as the subject of a narration line, not a speaker's tab
        # set apart over a dialogue box.
        subject_row = (previous is not None and previous[0].strip() in _TITLE_WORDS and _subject_of(row_text)
                       and row[0].box[1] - previous[1].box[1] - previous[1].box[3] < 0.5 * row[0].box[3])
        if subject_row and not same_baseline:
            paragraphs[-1] = _join(paragraphs[-1], row_text)
            previous = (paragraphs[-1], row[0], width)
            continue
        if previous is not None and not same_baseline and not menu and not _starts_new_paragraph(
                *previous, row[0], width, frame_w, frame_h, menus, paragraphs[-1] == previous[0]):
            # Not "DAD / WILL KILL US", where the title is the subject of the next row.
            # Only before words that address the listener ("MOM / I'M HOME!"), not "DAD / BROKE THE VASE", and not a
            # name tab set flush left over a game's dialogue box.
            addresses = (_ADDRESSES.match(row_text) or _ASKED.match(row_text)
                         or (not row_text[:1].isascii() and not _subject_of(row_text)))
            # A tab sits flush left over a wider line; a centred call over a line of similar width shares its centre too.
            left_gap = abs(previous[1].box[0] - row[0].box[0]) * frame_w
            centre_gap = abs((previous[1].box[0] + previous[1].box[2] / 2) - (row[0].box[0] + row[0].box[2] / 2)) * frame_w
            flush_tab = left_gap < 0.5 * row[0].box[3] * frame_h and centre_gap > row[0].box[3] * frame_h
            called = paragraphs[-1].strip() in _TITLE_WORDS and (addresses or row_text[:1].islower())
            if (called or paragraphs[-1].strip().upper() in _INTERJECTIONS) and not flush_tab:
                paragraphs[-1] = paragraphs[-1].strip() + ("，" if _cjk(paragraphs[-1]) else ", ") + row_text
            else:
                paragraphs[-1] = _join(paragraphs[-1], row_text)
            spans[-1] = [min(spans[-1][0], left), max(spans[-1][1], right), bottom, spans[-1][3]]
        else:
            paragraphs.append(row_text)
            spans.append([left, right, bottom, row[0].box[1]])
        previous = (row_text, row[0], width)
    return paragraphs, spans, rows[0][0] if rows else None


_INTERJECTIONS = {"WAIT", "HEY", "LOOK", "LISTEN", "STOP", "PLEASE", "COME ON"}
_ADDRESSES = re.compile(r"(?:I|I'M|I'LL|I'VE|YOU|WE|WHERE|WHAT|WHY|HOW|CAN|COULD|LOOK|WAIT|COME|HELP|PLEASE|HEY|ARE|DID|"
                        r"DO|STOP|DON'T|LET'S|THANKS?|TELL|GIVE|SHOW|LET|TAKE|GET|GO|SIT|EAT|READ|PLAY|BUY|LISTEN|HURRY|"
                        r"RUN|KEEP|TRY|WAKE|OPEN|GUESS|SORRY|GOOD|THEY|THEY'RE)\b", re.I)
_ASKED = re.compile(r"(?:CAN|COULD|WOULD|WILL|SHOULD|DID|DO|ARE|IS|WAS|HAVE|HAS)\b(?: YOU\b|.*\?$)", re.I)
# Narrative verbs only: "Dad / Could you help me?" or "Mom / Was that you?" is a call or a tab.
_SUBJECT_VERB = re.compile(r"(?:NEVER|ALWAYS|CAME|WENT|SAID|SAYS|TOLD|LEFT|DIED|WORKED|USED TO|ONCE|LOVED|HATED|KNEW|"
                           r"BROKE|LOST|FOUND|GAVE|TOOK|MADE|TAUGHT|WAS|IS|WILL|WON'T|GOT|HAS|HAD)\b"
                           "|(?:总是|總是|知道|说过|說過|已经|已經|从来|從來|一直|曾经|曾經|告诉|告訴|去世)", re.I)


def _subject_of(row_text: str) -> bool:
    """The row goes on with a verb whose subject is the title above it, rather than addressing the listener."""
    return bool(_SUBJECT_VERB.match(row_text)) and not _ASKED.match(row_text)


def _choice_rows(previous: tuple, below: OcrLine, below_text: str, frame_w: float, frame_h: float,
                 asked: bool, menu_left: float | None) -> bool:
    """Short choices stacked under a prompt or each other; a centred two-row subtitle is one line.

    """
    above_text, above = previous[0].strip(), previous[1]
    height = max(above.box[3], below.box[3]) * frame_h
    short = lambda text: len(text.replace(" ", "")) <= 8
    left = abs(above.box[0] - below.box[0]) * frame_w
    centre = abs((above.box[0] + above.box[2] / 2) - (below.box[0] + below.box[2] / 2)) * frame_w
    # Items under a marked choice ("> Yes" / "No") line up with the text after the marker.
    anchor = menu_left if menu_left is not None else above.box[0] if _MENU_MARK.match(above_text) else None
    if anchor is not None and -0.5 * height <= (below.box[0] - anchor) * frame_w <= 2 * height:
        return True
    if not short(below_text):
        return False
    indented = (below.box[0] - above.box[0]) * frame_w > 0.5 * height and above_text[-1:] in "。.!！:：?？"
    if left >= 1.5 * height and not indented:
        return False
    # はい / いいえ, Yes / No, 接受 / 拒绝: single words flush with each other, not a wrapped sentence.
    words = left < 0.3 * height and above_text.strip() in CHOICE_WORDS and below_text.strip() in CHOICE_WORDS
    return above_text[-1:] in "?？" or asked and short(above_text) or indented or words


def _split_wide_gaps(row: list[OcrLine]) -> list[list[OcrLine]]:
    ordered = sorted(row, key=lambda item: item.box[0])
    segments = [[ordered[0]]]
    for line in ordered[1:]:
        last = segments[-1][-1]
        if line.box[0] - (last.box[0] + last.box[2]) > 3 * max(line.box[3], last.box[3]) or _MENU_MARK.match(line.text):
            segments.append([line])
        else:
            segments[-1].append(line)
    return segments


def _starts_new_paragraph(above_text: str, above: OcrLine, above_width: float, below: OcrLine, below_width: float,
                          frame_w: float = 1, frame_h: float = 1, menus: bool = False, opening: bool = True) -> bool:
    """Wrapped text fills its first row, so a wide gap, a size change or a short row over a wide one is a new line."""
    height = max(above.box[3], below.box[3])
    gap = below.box[1] - (above.box[1] + above.box[3])
    ratio = min(above.box[3], below.box[3]) / height
    if _lead_in((above.box[0] * frame_w, above.box[1] * frame_h, above_width * frame_w, above.box[3] * frame_h, "h", above_text),
                (below.box[0] * frame_w, below.box[1] * frame_h, below_width * frame_w, below.box[3] * frame_h, "h", below.text)):
        return False
    # OCR boxes vary in height for the same font, so a size change alone only counts when it is large or spaced out.
    if gap > 0.6 * height or ratio < 0.6 or (ratio < 0.75 and gap > 0.3 * height):
        return True
    if _separate_entries(above_text, below.text):
        return True
    # Only the first row of a paragraph can be a speaker's tab ("KNIGHT"), not a short row inside it ("CASTLE").
    if not opening:
        return False
    # A name box over its dialogue: OCR pads boxes, so the name often overlaps the line below it.
    left_aligned = abs(above.box[0] - below.box[0]) < height
    # A family title alone over the line ("Mom" / "I'm home.") calls the listener unless set clearly smaller as a tab.
    titled = above_text.strip().strip(",，、") in _TITLE_WORDS or above_text.strip().upper() in _INTERJECTIONS
    if titled:
        left_gap = abs(above.box[0] - below.box[0])
        centre_gap = abs((above.box[0] + above_width / 2) - (below.box[0] + below_width / 2))
        left_aligned = left_gap * frame_w < 0.5 * height * frame_h and left_gap < centre_gap
    # A title centred over its line calls the listener (policy 9); only a name is a centred cutscene tab.
    centred = (not titled and abs((above.box[0] + above_width / 2) - (below.box[0] + below_width / 2)) < height
               and gap > 0.08 * height and above.box[3] < 0.92 * below.box[3])
    # A visual-novel name tab over a short reply ("小林" / "好吧"): both short, set apart a little.
    # An all-caps speaker tab ("OLD MINER", "SGT. HALE") flush left over a much wider line.
    caps_shape = (above_text.isupper() and len(above_text.split()) <= 3 and len(above_text) <= 14 and left_aligned
                  and above_width < 0.45 * below_width and abs(above.box[0] - below.box[0]) * frame_w < 0.5 * height * frame_h)
    # Set right on a caption's first line, it is the caption's head ("MEANWHILE" / "AT THE HARBOR, …").
    if caps_shape:
        return gap > 0.25 * height
    short_reply = (menus and left_aligned and len(above_text.strip()) <= 4 and len(below.text.strip()) <= 6 and gap > 0.15 * height
                   and not above_text.strip()[-1:] in "，,、")
    return (_name_like(above_text) and (left_aligned or centred) and (above_width < 0.6 * below_width or short_reply)
            and gap > -0.5 * height)


def _name_like(text: str) -> bool:
    """A speaker's name box rather than the short first line of a speech bubble ("WHERE ARE", "ねえ、")."""
    text = text.strip()
    if text in CHOICE_WORDS:
        return False
    words = text.split()
    if not 0 < len(text) <= 20 or len(words) > 3 or text.endswith(_ENDS) or _MID_PHRASE.search(text):
        return False
    # A chat row or label carries its own speaker or tag ("【队伍】老刘：…").
    if re.search("[【】\\[\\]「」]", text):
        return False
    if re.search("[\u3041-\u309f]", text) and text not in ROLES_VI:
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
    # One item of a menu or choice list, never a speaker's name tab or a wrapped line.
    listed: bool = False


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
        if a[4] == "h" or a[4] == "?":
            below = _ruby_row_base(pixels, i)
            if below is not None:
                found[i] = below
            # A one- or two-kana reading is too short to show which way it runs.
            if below is not None or a[4] == "h" or not _KANA_ONLY.fullmatch(a[5]):
                continue
        for j, b in enumerate(pixels):
            if i == j or b[4] != "v" or a[2] >= 0.6 * b[2]:
                continue
            gap = max(a[0], b[0]) - min(a[0] + a[2], b[0] + b[2])
            if gap <= 0.5 * b[2] and a[1] >= b[1] - 0.5 * b[2] and a[1] + a[3] <= b[1] + b[3] + 0.5 * b[2]:
                found[i] = j
                break
    return found


def _ruby_row_base(pixels: list[tuple], i: int) -> int | None:
    """The row a small kana-only row sits right on top of, as furigana over horizontal text does."""
    a = pixels[i]
    if not _KANA_ONLY.fullmatch(a[5]):
        return None
    for j, b in enumerate(pixels):
        if j == i or b[4] == "v" or a[3] >= 0.6 * b[3]:
            continue
        gap = b[1] - (a[1] + a[3])
        inside = a[0] >= b[0] - 0.5 * b[3] and a[0] + a[2] <= b[0] + b[2] + 0.5 * b[3]
        # A reading spells out kanji; a small kana line over kana or a short lead-in over a shout is part of the text.
        width = b[2] / max(1, len(b[5]))
        start, end = int((a[0] - b[0]) / width), math.ceil((a[0] + a[2] - b[0]) / width)
        kanji = len(re.findall("[\u4e00-\u9fff々]", b[5][max(0, start):max(0, end)]))
        # A name label sits flush with the left edge of its line; a reading sits over the kanji somewhere along it.
        flush = abs(a[0] - b[0]) < 0.5 * b[3]
        # So does a cutscene name centred over its subtitle line, which is not much smaller than the line.
        centred = abs((a[0] + a[2] / 2) - (b[0] + b[2] / 2)) < b[3] and a[3] >= 0.55 * b[3]
        # A hiragana reading over the kanji that open a line ("あした" over "明日") is flush with it too.
        hiragana = re.fullmatch("[ぁ-ゖー ]+", a[5])
        # OCR pads boxes, so a tightly set reading can overlap its kanji by most of its own height.
        above = a[1] + a[3] / 2 < b[1] + 0.25 * b[3]
        if -0.75 * a[3] <= gap <= 0.4 * b[3] and above and inside and kanji and len(a[5].replace(" ", "")) <= 3 * kanji and (
                hiragana or not (flush or centred)):
            return j
    return None


def group_blocks(lines: list[OcrLine], width: int = 1, height: int = 1, min_confidence: float = 0.3) -> list[Block]:
    kept = clean_lines(lines, min_confidence, lone_cjk=True)
    pixels = [(l.box[0] * width, l.box[1] * height, l.box[2] * width, l.box[3] * height,
               _orientation(l, width, height), l.text.strip()) for l in kept]
    ruby = _ruby(pixels)
    listed = _list_rows(pixels, width, screen=True)
    cells = _cells(pixels)
    parent = list(range(len(kept)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(kept)):
        for j in range(i + 1, len(kept)):
            # A menu row stays apart from the rows above and below it, not from text beside it on its own row.
            stacked = abs((pixels[i][1] + pixels[i][3] / 2) - (pixels[j][1] + pixels[j][3] / 2)) >= 0.5 * max(
                pixels[i][3], pixels[j][3])
            stat = not re.search(r"[^\W\d_]", pixels[i][5]) or not re.search(r"[^\W\d_]", pixels[j][5])
            if i not in ruby and j not in ruby and not ((i in listed or j in listed) and stacked and not stat) \
                    and not ((i in cells or j in cells) and stacked) and _same_block(pixels[i], pixels[j]):
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
        letters = sum(ch.isalpha() for ch in text)
        lone = letters == 1 and re.search(r"[?!。！？…]|[\u3040-\u9fff\uac00-\ud7a3]", text)
        if letters < 2 and not lone:
            continue
        columns = [l for l in words if is_vertical(l, width, height)] or words
        blocks.append(Block(text, (left, top, right - left, bottom - top),
                            median(char_size(l, width, height) for l in columns), vertical=vertical,
                            listed=any(i in listed for i in members)))
    if sum(b.vertical for b in blocks) * 2 > len(blocks):
        # A short "！" column OCR reads as 一 or ー is not a bubble of its own.
        return _manga_order([block for block in blocks if block.text.strip() not in ("一", "ー", "口", "|", "丨")])
    blocks.sort(key=lambda b: (round(b.box[1] * 50), b.box[0]))
    return _with_name_boxes(blocks, width, height)


def _cells(pixels: list[tuple]) -> set[int]:
    """Labels in a row of separate items (a button bar, a label and its value), which never wrap onto the next row."""
    found = set()
    for i, a in enumerate(pixels):
        for b in pixels:
            if a is b or "v" in (a[4], b[4]):
                continue
            height = max(a[3], b[3])
            gap = max(a[0], b[0]) - min(a[0] + a[2], b[0] + b[2])
            if abs((a[1] + a[3] / 2) - (b[1] + b[3] / 2)) < 0.5 * height and 0.6 * height < gap < 4 * height:
                found.add(i)
    return found


def _with_name_boxes(blocks: list[Block], width: float = 1, height: float = 1) -> list[Block]:
    """A visual novel's name tab joined to the dialogue right under it, as "Name\nline", so the speaker is known.

    A visual novel shows one name tab at a time. With several header-over-text pairs on one screen, only the one
    that reads like dialogue (a sentence, lowest on screen, where dialogue boxes sit) is joined, unless the pairs sit
    side by side as skit bubbles; if none stands out, they are panel headings, chat tabs or separate bubbles, and
    nothing is joined. A backlog lists each name in a column beside its line, and every row is joined.
    """
    backlog = _backlog_pairs(blocks, width, height)
    if backlog:
        return _join_pairs(blocks, backlog)
    pairs = _name_pairs(blocks, width, height) or _chat_header(blocks)
    if len(pairs) > 1:
        names = [blocks[i] for i, _j in pairs]
        row = max(n.box[1] for n in names) - min(n.box[1] for n in names) < max(n.line_height for n in names)
        # A skit's name boxes sit apart from their bubbles; panel headings sit right on their text.
        row = row and all(blocks[j].box[1] - blocks[i].box[1] - blocks[i].box[3] > 0.5 * blocks[j].line_height
                          for i, j in pairs)
        apart = all(a.box[0] + a.box[2] < b.box[0] or b.box[0] + b.box[2] < a.box[0]
                    for a, b in zip(sorted(names, key=lambda n: n.box[0]), sorted(names, key=lambda n: n.box[0])[1:]))
        # Item names set right on their descriptions, one after another, are a list rather than dialogue.
        lefts = [n.box[0] * width for n in names]
        listing = max(lefts) - min(lefts) < max(n.line_height for n in names) * height and all(
            blocks[j].box[1] - blocks[i].box[1] - blocks[i].box[3] < 0.5 * blocks[j].line_height for i, j in pairs)
        if listing:
            pairs = []
        elif not (row and apart):
            spoken = [(i, j) for i, j in pairs if re.search(r"[.!?。！？…]", blocks[j].text)]
            lowest = max(spoken, key=lambda pair: blocks[pair[1]].box[1], default=None)
            pairs = [lowest] if lowest and blocks[lowest[1]].box[1] > 0.55 else []
    return _join_pairs(blocks, pairs)


def _chat_header(blocks: list[Block]) -> list[tuple[int, int]]:
    """A messenger screen: the contact's name in the title bar is the speaker of the one message shown."""
    texts = [i for i, block in enumerate(blocks) if not _CHAT_META.fullmatch(block.text.strip())]
    if len(texts) != 2:
        return []
    name, line = sorted(texts, key=lambda i: blocks[i].box[1])
    header = blocks[name]
    title_bar = header.box[1] < 0.05 and abs(header.box[0] + header.box[2] / 2 - 0.5) < 0.15
    return [(name, line)] if title_bar and _name_like(header.text) and not blocks[line].vertical else []


def _backlog_pairs(blocks: list[Block], width: float, height: float) -> list[tuple[int, int]]:
    """Two or more names in one column, each on the baseline of a line of dialogue set well to its right."""
    pairs = []
    for i, name in enumerate(blocks):
        if not _name_like(name.text) or name.vertical:
            continue
        lh = name.line_height * height
        for j, line in enumerate(blocks):
            gap = (line.box[0] - name.box[0] - name.box[2]) * width
            level = abs((line.box[1] + line.box[3] / 2) - (name.box[1] + name.box[3] / 2)) * height < 0.5 * lh
            if j != i and level and 2 * lh < gap < 10 * lh and re.search(r"[.!?。！？…]$", line.text.strip()):
                pairs.append((i, j))
                break
    lefts = [blocks[i].box[0] * width for i, _j in pairs]
    if len(pairs) < 2 or max(lefts) - min(lefts) > blocks[pairs[0][0]].line_height * height:
        return []
    return pairs


def _name_over_line(upper: tuple, lower: tuple) -> bool:
    """A cutscene speaker's name set over a long subtitle line ("Aria" / "We can't stay here..."), not a wrapped row."""
    name, line = upper[5].strip(), lower[5].strip()
    if not _name_like(name) or name.isupper() or upper[2] >= lower[2] / 3 or name in _TITLE_WORDS:
        return False
    eastern = _cjk(line) or re.search("[\uac00-\ud7a3]", line)
    if eastern and len(name.replace(" ", "")) <= 4 and len(line.replace(" ", "")) >= 16 or not eastern and len(line) >= 30:
        return True
    # Set smaller, centred, a little apart over a line of its own: "ユイ" over "お兄ちゃん、待って！". A bubble's rows touch.
    centred = abs((upper[0] + upper[2] / 2) - (lower[0] + lower[2] / 2)) < lower[3]
    apart = lower[1] - (upper[1] + upper[3]) > 0.08 * lower[3] or upper[3] < 0.85 * lower[3]
    enough = len(line.replace(" ", "")) >= 4 if eastern else len(line.split()) >= 2 and not line[:1].islower()
    return (centred and apart and upper[3] < 0.92 * lower[3] and enough
            and len(name.replace(" ", "")) <= (6 if eastern else 20))


def _list_rows(pixels: list[tuple], frame_width: float = 0, screen: bool = False) -> set[int]:
    """Rows of a menu: three or more short rows of one size, lined up, at an even pitch (pause menus, title screens)."""
    def short(text: str) -> bool:
        if _cjk(text) or re.search("[\uac00-\ud7a3]", text):
            return len(text.replace(" ", "")) <= 12
        return len(text.split()) <= 4 and len(text) <= 24

    # A row starting in lower case carries on the sentence above it ("So where have / you been all").
    # A stat readout ("H 45", "经验 12000") is a value under a label, not a menu item.
    rows = sorted((i for i, p in enumerate(pixels) if p[4] != "v" and short(p[5]) and not p[5][:1].islower()
                   and not re.search("[。.!！?？,，、:：]", p[5]) and not _READOUT.fullmatch(p[5])),
                  key=lambda i: pixels[i][1])
    listed: set[int] = set()
    for start in rows:
        chain = [start]
        for i in rows:
            a, b = pixels[chain[-1]], pixels[i]
            if b[1] <= a[1]:
                continue
            height = max(a[3], b[3])
            pitch = (b[1] + b[3] / 2) - (a[1] + a[3] / 2)
            lined = abs(a[0] - b[0]) < 0.5 * height or abs((a[0] + a[2] / 2) - (b[0] + b[2] / 2)) < 0.5 * height
            if min(a[3], b[3]) >= 0.8 * height and lined and 0.85 * height <= pitch <= 4 * height:
                chain.append(i)
        capitals = all(re.fullmatch(r"[A-Z0-9 '’,.!?\-]+", pixels[i][5]) for i in chain)
        centres = [pixels[i][0] + pixels[i][2] / 2 for i in chain]
        tallest = max(pixels[i][3] for i in chain)
        on_axis = frame_width and abs(median(centres) - frame_width / 2) < 0.1 * tallest
        # Two capitalised items on the middle of the screen (CONTINUE / NEW GAME) are a title menu.
        if len(chain) < 3 and not (screen and len(chain) == 2 and capitals and on_axis):
            continue
        # A speech bubble's unpunctuated rows run on into a row that ends the sentence ("DON'T TELL / GRANDMA / …!").
        last = pixels[chain[-1]]
        # A smaller description line under a menu is not the end of a sentence.
        ends = any(re.search("[。.!！?？…]$", p[5].strip()) and 0 < (p[1] - last[1]) <= 4 * last[3] and p[3] >= 0.85 * last[3]
                   and (abs(p[0] - last[0]) < 0.5 * last[3] or abs((p[0] + p[2] / 2) - (last[0] + last[2] / 2)) < 0.5 * last[3])
                   for p in pixels)
        # All-caps comic lettering: centred rows of varying width, closer than 1.6 line heights; a left-aligned all-caps
        # column (RESUME / SETTINGS / CONTROLS) is a menu.
        ragged = any(abs(pixels[a][0] - pixels[b][0]) > 0.3 * pixels[a][3] for a, b in zip(chain, chain[1:]))
        close = max(pixels[b][1] - pixels[a][1] for a, b in zip(chain, chain[1:])) < 1.6 * tallest
        if ends or (capitals and ragged and close):
            continue
        pitches = [(pixels[b][1] + pixels[b][3] / 2) - (pixels[a][1] + pixels[a][3] / 2) for a, b in zip(chain, chain[1:])]
        middle = median(pitches)
        if any(abs(p - middle) > 0.15 * middle for p in pitches):
            continue
        lefts = [pixels[i][0] for i in chain]
        if max(centres) - min(centres) < max(lefts) - min(lefts):
            # Centred rows: a menu sits on the middle of the screen; elsewhere, rows that only widen or only narrow
            # are the shape of a speech bubble, and a bubble's leading is tighter than a menu's.
            widths = [pixels[i][2] for i in chain]
            steps = [b - a for a, b in zip(widths, widths[1:])]
            shaped = all(step > 0.25 * tallest for step in steps) or all(step < -0.25 * tallest for step in steps)
            if (not on_axis or capitals and len(chain) > 2) and (shaped or min(pitches) < 1.2 * tallest):
                continue
        listed.update(chain)
    return listed


def _name_pairs(blocks: list[Block], width: float = 1, height: float = 1) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    used: set[int] = set()

    def px(block: Block) -> tuple[float, float, float, float, float]:
        return (block.box[0] * width, block.box[1] * height, block.box[2] * width, block.box[3] * height,
                block.line_height * height)

    for i, name in enumerate(blocks):
        header = name.text.isupper() and len(name.text.split()) == 1 and len(name.text) > 3
        if i in used or not _name_like(name.text) or name.vertical or header or name.listed:
            continue
        nx, ny, nw, nh, nlh = px(name)
        # A label in a bar of items on its own row (a site menu: 홈 웹툰 베스트도전) is not a speaker.
        if any(k != i and abs((other.box[1] + other.box[3] / 2) - (name.box[1] + name.box[3] / 2)) * height < 0.5 * nlh
               and 0 < max(other.box[0] * width - nx - nw, nx - (other.box[0] + other.box[2]) * width) < 6 * nlh
               for k, other in enumerate(blocks)):
            continue
        for j, block in enumerate(blocks):
            if j == i or j in used or block.vertical or block.listed:
                continue
            words = sum(ch.isalpha() for ch in block.text)
            enough = max(4, 2 * len(name.text)) if _cjk(block.text) else max(12, 2 * len(name.text))
            # A stat readout ("1,430 / 1,600") under a party member's name is not their line.
            spoken = words >= 4 and (len(block.text) >= enough or block.text.rstrip().endswith(_ENDS))
            same_size = 0.6 * block.line_height <= name.line_height <= 1.3 * block.line_height
            # A list entry, a chat row with its [tag] or a backlog row with its own Name「 already has its speaker.
            labelled = (_CHAT_ROW.match(block.text) or re.match(r"\s*[-–•・]\s", block.text)
                        or "「" in block.text[:12])
            if not (spoken and same_size) or labelled:
                continue
            bx, by, bw, bh, blh = px(block)
            drop = by - (ny + nh)
            # A heading set larger, flush over the paragraph it heads and close to it, rather than in a tab of its own.
            heading = (abs(bx - nx) < 0.25 * blh and nlh >= 1.1 * blh
                       and (drop < 0.15 * blh or drop < 0.6 * blh and bh >= 1.8 * blh))
            centred = abs((bx + bw / 2) - (nx + nw / 2)) <= blh
            lined_up = abs(bx - nx) <= 1.5 * blh or centred
            left, right = min(bx, nx), max(bx + bw, nx + nw)
            between = any(k not in (i, j) and ny < other.box[1] * height < by
                          and other.box[0] * width < right and (other.box[0] + other.box[2]) * width > left
                          for k, other in enumerate(blocks))
            if -0.5 * blh <= drop <= 1.5 * blh and lined_up and not between and not heading:
                pairs.append((i, j))
                used.update((i, j))
                break
    return pairs


def _join_pairs(blocks: list[Block], pairs: list[tuple[int, int]]) -> list[Block]:
    bodies = {j: i for i, j in pairs}
    names = {i for i, _j in pairs}
    merged: list[Block] = []
    for index, block in enumerate(blocks):
        if index in bodies:
            continue
        if index not in names:
            merged.append(block)
            continue
        body = blocks[next(j for i, j in pairs if i == index)]
        left, top = min(block.box[0], body.box[0]), min(block.box[1], body.box[1])
        right = max(block.box[0] + block.box[2], body.box[0] + body.box[2])
        bottom = max(block.box[1] + block.box[3], body.box[1] + body.box[3])
        merged.append(Block(f"{block.text}\n{body.text}", (left, top, right - left, bottom - top), body.line_height,
                            body.bg, body.fg))
    return merged


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


_MENU_MARK = re.compile(r"^\s*[>▶●►]")
_CHAT_ROW = re.compile(r"\s*(?:\[[^\]]{1,20}\]|【[^】]{1,10}】)\s*[^\s:：]{1,12}[:：]")
_CHAT_META = re.compile(r"(?:(?:오전|오후|午前|午後|上午|下午|AM|PM)\s*)?\d{1,2}:\d{2}(?:\s*(?:AM|PM))?|已读|未读|既読|읽음|Read")
_READOUT = re.compile(r"[^\W\d_]{1,4}\s*[\d,./:+\-% ]*\d")
_KANA_ONLY = re.compile("[\u3041-\u30ff]+(?: [\u3041-\u30ff]+)*")
_MID_PHRASE = re.compile(r"[、，,;:\-]$|[でにをはがのとへもや]$|\b(the|a|an|to|of|and|with|in|on|for|or|your|my)$", re.I)
_ENDS = (".", "!", "?", "。", "！", "？", "…", ":", "：")
_STAT_END = re.compile(r"\d[\d,./:%+\-]*\)?$")
_ENTRY_START = re.compile(r"^\s*(?:[-–－](?:\s|(?=[^\d\s]))|[+\-−]\d|\d+%|\d{1,2}[.)]\s|[①-⑩▶>]|\[[^\]]{1,20}\]|"
                          r"[^\s:：\[\]]{1,24}:\s|[^\s:：\[\]]{1,12}：)")
_KOREAN_END = re.compile("(?:요|다|까|죠|네)[.!?]?$")
_TAG_START = re.compile(r"^\s*(?:\[[^\]]{1,20}\]|【[^】]{1,10}】)")
_NUMBER_START = re.compile(r"^\s*(?:[+\-−]\d|\d+%)")


def _short(text: str) -> bool:
    return len(text.replace(" ", "")) <= 8 if _cjk(text) else len(text.split()) <= 3


def _separate_entries(upper: str, lower: str) -> bool:
    """HUD stats, chat lines or two speakers in one subtitle, rather than one sentence wrapped onto the next row.

    A row ending in a number counts only over another number ("HP 76" / "MP 40", not "Only 3" / "tickets left?!"),
    and a tag, name or dash only after a row that ends or is itself an entry, not mid-sentence.
    """
    upper, lower = upper.rstrip(), lower.strip()
    if _STAT_END.search(upper) and _short(upper) and (_STAT_END.search(lower) or lower[:1].isdigit()):
        return True
    # A [tag] starts a chat or system row, unless the row above stops mid-sentence ("…も" / "[E] で開く").
    if _TAG_START.match(lower):
        return not (_MID_PHRASE.search(upper) and not _TAG_START.match(upper))
    if _TAG_START.match(upper) and not _TAG_START.match(lower) and (upper.endswith(_ENDS) or _KOREAN_END.search(upper)):
        return True
    if not _ENTRY_START.match(lower):
        return False
    ended = upper.endswith(_ENDS) or bool(_ENTRY_START.match(upper)) or bool(_TAG_START.match(upper))
    if _NUMBER_START.match(lower):
        return ended or bool(_STAT_END.search(upper))
    return ended or bool(_STAT_END.search(upper)) or _short(upper)


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
        # A chat bubble's time or read mark beside the message is not part of it.
        if bool(_CHAT_META.fullmatch(a[5].strip())) != bool(_CHAT_META.fullmatch(b[5].strip())):
            return False
        small, big = (a, b) if a[3] < b[3] else (b, a)
        # OCR cuts the last kana of a choice into a smaller box of its own ("そっとしてお" + "く").
        tail = (len(small[5].strip()) <= 2 and gap < 0.3 * height and small[0] > big[0]
                and big[1] <= small[1] and small[1] + small[3] <= big[1] + big[3])
        # A wider hole on one row is usually a button icon or a gap between separate labels.
        return (ratio >= 0.7 or tail) and gap < 0.6 * height
    upper, lower = (a, b) if a[1] < b[1] else (b, a)
    gap = lower[1] - (upper[1] + upper[3])
    overlap = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    same_size = ratio >= 0.75 or (ratio >= 0.6 and gap <= 0.3 * height) or _lead_in(upper, lower)
    # Lines of one paragraph or speech bubble sit close together; more widely spaced rows join only when the upper
    # one stops mid-phrase, so menu items, list entries and chat lines stay apart.
    # A stat ending the upper row, or a tag, name, or dash starting the lower one, marks separate entries: HUD stacks,
    # chat lines, two speakers in one subtitle.
    text = upper[5].rstrip()
    if _separate_entries(text, lower[5]) or _separate_items(upper, lower):
        return False
    # A party member's name over their HP ("Bram" / "1,430 / 1,600") is a label over a readout.
    if re.fullmatch(r"[\d,./ :%+\-]+", lower[5].strip()) and re.search(r"[^\W\d_]", text):
        return False
    if _name_over_line(upper, lower):
        return False
    mid_phrase = _MID_PHRASE.search(text) and not re.fullmatch(r"[A-Za-z]+", text)
    # A shorter CJK row of another size, set apart, is a header or title; a bubble's rows touch.
    cjk_title = (_cjk(text) and _cjk(lower[5]) and ratio < 0.88 and upper[2] < 0.95 * lower[2]
                 and gap > 0.12 * height and not text.endswith(("，", "、", ",")) and not mid_phrase)
    # Comic lettering is all capitals with loose leading.
    centred_rows = abs((upper[0] + upper[2] / 2) - (lower[0] + lower[2] / 2)) < 0.5 * height
    flush_rows = abs(upper[0] - lower[0]) < 0.5 * height
    stats = bool(re.search(r"\d\s*[A-Z%]{0,3}$|/", text) or re.search(r"\d\s*[A-Z%]{0,3}$|/", lower[5]))
    lettering = (text.isupper() and lower[5].isupper() and gap <= 0.6 * height and (centred_rows or flush_rows)
                 and not text.endswith(_ENDS) and not stats)
    # A bubble's rows are centred on each other, the lower one reaching past the upper on both sides.
    bubble = (centred_rows and ratio >= 0.8 and gap <= 0.6 * height and not text.endswith(_ENDS)
              and lower[0] < upper[0] - 0.1 * height and lower[0] + lower[2] > upper[0] + upper[2] + 0.1 * height)
    wraps = not cjk_title and (gap <= 0.25 * height or mid_phrase or lower[5][:1].islower() or lettering or bubble
                               or (_cjk(text) and len(text.replace(" ", "")) >= 8 and not text.endswith(_ENDS)
                                   and gap <= 0.35 * height))
    # Padded OCR boxes of tightly set or slightly rotated lines overlap by up to half a line.
    return wraps and same_size and -0.5 * height <= gap <= 0.6 * height and overlap >= 0.3 * min(a[2], b[2])


def _lead_in(upper: tuple, lower: tuple) -> bool:
    """Manga sets a shout's lead-in small over it ("いいかげんに" / "しろ！！")."""
    height = max(upper[3], lower[3])
    gap = lower[1] - (upper[1] + upper[3])
    # A katakana or kanji name over its line is a tab ("ユイ" / "お兄ちゃん、待って！"); lead-ins are hiragana.
    return bool(re.fullmatch("[ぁ-ゖー]+", upper[5].strip()) and not upper[5].rstrip().endswith(_ENDS) and re.search(r"[!！?？]$", lower[5].strip())
                and min(upper[3], lower[3]) >= 0.4 * height and gap <= 0.3 * height
                and abs((upper[0] + upper[2] / 2) - (lower[0] + lower[2] / 2)) < 0.5 * height)


def _separate_items(upper: tuple, lower: tuple) -> bool:
    """Stacked HUD rows, toasts, menu items and labels that only sit close together: a wrapped row fills its line."""
    above, below = upper[5].strip(), lower[5].strip()
    height = max(upper[3], lower[3])
    if above in _TITLE_WORDS and _subject_of(below):
        return False
    if above in CHOICE_WORDS or below in CHOICE_WORDS or "://" in above + below:
        return True
    # A score or stat change ending the row ("Drift x3 +1,250"), or two readouts one over the other.
    if re.search(r"[+\-−]\d[\d,.]*%?$", above):
        return True
    if _STAT_END.search(above) and _STAT_END.search(below) and len(above.split()) <= 5 and len(below.split()) <= 5:
        return True
    # A counter row under a line of text ("REACH THE EXTRACTION POINT" / "SURVIVORS 3/4").
    if (re.search(r"\d/\d+$", below) and len(below.split()) <= 3 and not re.search(r"\d$", above)
            and not _MID_PHRASE.search(above)):
        return True
    # Toasts repeat their label ("Picked up: Iron Ore x5" / "Picked up: Wolf Pelt").
    label = re.match(r"[^:：]{2,24}[:：]", above)
    if label and below.startswith(label.group(0)):
        return True
    ended = above.endswith(_ENDS) or bool(_MID_PHRASE.search(above))
    # A lone capitalised word under a label ("Daily Login Reward" / "Claim") is a button, not the end of a sentence.
    if re.fullmatch(r"[A-Z][a-z]+", below) and not ended and not above.isupper() and len(above.split()) > 1:
        return True
    if ended or not above or above.isupper():
        return False
    # A flush row with room left for the next row's first word was not wrapped ("Hold R2 to sprint" / "L1 Aim…").
    flush_left = abs(upper[0] - lower[0]) < 0.5 * height
    flush_right = abs((upper[0] + upper[2]) - (lower[0] + lower[2])) < 0.5 * height
    if flush_left == flush_right:
        return False
    char = upper[2] / max(1, len(above))
    first = re.match(r"\S+", below).group(0)
    needed = 1.5 * char if _cjk(first[0]) else (len(first) + 1) * char
    return lower[2] - upper[2] >= needed


def _same_bubble(a: tuple, b: tuple) -> bool:
    size = max(a[2], b[2])
    gap = max(a[0], b[0]) - min(a[0] + a[2], b[0] + b[2])
    overlap = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    # Vertical text reads right to left: a finished sentence followed by a wider gutter is the next bubble.
    first = a if a[0] > b[0] else b
    if gap > 0.3 * size and re.search(r"[。！？!?°]$", first[5].strip()):
        return False
    return min(a[2], b[2]) >= 0.6 * size and -0.5 * size <= gap <= 0.8 * size and overlap >= 0.3 * min(a[3], b[3])
