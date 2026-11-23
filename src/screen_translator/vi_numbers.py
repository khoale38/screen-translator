"""Numbers written out in Vietnamese words for the voice, which otherwise nearly skips digits ("15%" in 0.1 s)."""

from __future__ import annotations

import re

_UNITS = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
_SCALES = ["", "nghìn", "triệu", "tỷ"]


def _three(n: int, full: bool) -> list[str]:
    hundreds, tens, ones = n // 100, n // 10 % 10, n % 10
    words = [_UNITS[hundreds], "trăm"] if full or hundreds else []
    if tens == 0:
        if ones:
            words += (["linh"] if words else []) + [_UNITS[ones]]
    elif tens == 1:
        words += ["mười"] + (["lăm" if ones == 5 else _UNITS[ones]] if ones else [])
    else:
        last = {1: "mốt", 4: "tư", 5: "lăm"}.get(ones, _UNITS[ones]) if ones else None
        words += [_UNITS[tens], "mươi"] + ([last] if last else [])
    return words


def spoken_integer(n: int) -> str:
    if n == 0:
        return "không"
    groups = []
    while n:
        groups.append(n % 1000)
        n //= 1000
    if len(groups) > len(_SCALES):
        return " ".join(_UNITS[int(d)] for d in str(sum(g * 1000 ** i for i, g in enumerate(groups))))
    words: list[str] = []
    for index in range(len(groups) - 1, -1, -1):
        group = groups[index]
        if not group:
            continue
        words += _three(group, full=bool(words)) + ([_SCALES[index]] if _SCALES[index] else [])
    return " ".join(words)


def _number(text: str) -> str:
    whole, _, fraction = text.partition(",") if "," in text and not re.fullmatch(r"\d{1,3}(,\d{3})+", text) \
        else text.partition(".") if not re.fullmatch(r"\d{1,3}(\.\d{3})+", text) else (text.replace(".", ""), "", "")
    whole = whole.replace(",", "").replace(".", "")
    spoken = spoken_integer(int(whole)) if whole else "không"
    return f"{spoken} phẩy {' '.join(_UNITS[int(d)] for d in fraction)}" if fraction else spoken


_NUMBER = r"\d+(?:[.,]\d+)*"
_TIME = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_PERCENT = re.compile(rf"({_NUMBER})\s*%")
_DOLLARS = re.compile(rf"\$\s*({_NUMBER})")
_PLAIN = re.compile(_NUMBER)


def spoken_numbers(text: str) -> str:
    """'Pin còn 15% lúc 7:30, giá $999' -> 'Pin còn mười lăm phần trăm lúc bảy giờ ba mươi, giá chín trăm chín mươi chín đô'."""
    text = re.sub(r"(\d)(?=[^\W\d_])", r"\1 ", text)
    text = _TIME.sub(lambda m: f"{spoken_integer(int(m.group(1)))} giờ"
                     + (f" {spoken_integer(int(m.group(2)))}" if int(m.group(2)) else ""), text)
    text = _PERCENT.sub(lambda m: f"{_number(m.group(1))} phần trăm", text)
    text = _DOLLARS.sub(lambda m: f"{_number(m.group(1))} đô", text)
    return _PLAIN.sub(lambda m: _number(m.group(0)), text)
