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


def _digits(text: str) -> str:
    return " ".join(_UNITS[int(d)] for d in text if d.isdigit())


def _number(text: str) -> str:
    if re.fullmatch(r"\d{1,3}(\.\d{3})+|\d{1,3}(,\d{3})+", text):
        return spoken_integer(int(re.sub(r"\D", "", text)))
    separators = re.findall(r"[.,]", text)
    if len(separators) > 1:
        # Versions, addresses and countdowns: "18.2.1", "192.168.1.1", "3,2,1".
        joiner = " chấm " if separators[0] == "." else ", "
        return joiner.join(_number(part) for part in re.split(r"[.,]", text))
    whole, _, fraction = text.partition(separators[0]) if separators else (text, "", "")
    if len(whole) > 1 and whole.startswith("0"):
        spoken = _digits(whole)
    else:
        spoken = spoken_integer(int(whole)) if whole else "không"
    return f"{spoken} phẩy {_digits(fraction)}" if fraction else spoken


def _date(match: re.Match) -> str:
    day, month, year = match.group(1), match.group(2), match.group(3)
    spoken = f"{spoken_integer(int(day))} tháng {spoken_integer(int(month))}"
    return f"{spoken} năm {spoken_integer(int(year))}" if year else spoken


def _slash(match: re.Match) -> str:
    top, bottom = int(match.group(1)), int(match.group(2))
    dated = re.search(r"(ngày|hôm|từ|đến|tới|sáng|chiều|tối)\s*$", match.string[:match.start()], re.IGNORECASE)
    if top < bottom <= 4 and not dated:
        return "nửa" if (top, bottom) == (1, 2) else f"{spoken_integer(top)} phần {spoken_integer(bottom)}"
    if 1 <= top <= 31 and 1 <= bottom <= 12 and (dated or match.group(3) or top >= bottom):
        return _date(match)
    return f"{spoken_integer(top)} trên {spoken_integer(bottom)}"


_NUMBER = r"\d+(?:[.,]\d+)*"
_TIME = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_HOURS = re.compile(r"\b(\d{1,2})h(\d{2})?\b")
_FULL_DATE = re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b")
_SLASH = re.compile(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{4}))?\b")
_PHONE = re.compile(r"(?<![\d.,])0\d{2,}(?: \d{2,})*")
_PERCENT = re.compile(rf"({_NUMBER})\s*%")
_DOLLARS = re.compile(rf"\$\s*({_NUMBER})")
_DEGREES = re.compile(r"(\d)\s*°\s*([CF]\b)?")
_NEGATIVE = re.compile(r"(?<![\w.,])-(?=\d)")
_TIMES = re.compile(r"(?<![\w])[x×](?=\d)|(?<=\d)[x×](?!\w)")
_UNIT_WORDS = [("kg", "ki lô gam"), ("km", "ki lô mét"), ("cm", "xen ti mét"), ("mm", "mi li mét"),
               ("ml", "mi li lít"), ("g", "gam"), ("k", "nghìn"), ("K", "nghìn"), ("đ", "đồng"), ("vnđ", "đồng"),
               ("VNĐ", "đồng"), ("VND", "đồng")]
_UNITS_AFTER = re.compile(r"(\d)\s*(" + "|".join(unit for unit, _ in sorted(_UNIT_WORDS, key=lambda u: -len(u[0])))
                          + r")(?!\w)")
_PLAIN = re.compile(_NUMBER)


def spoken_numbers(text: str) -> str:
    """'Pin còn 15% lúc 7:30, giá $999' -> 'Pin còn mười lăm phần trăm lúc bảy giờ ba mươi, giá chín trăm chín mươi chín đô'."""
    text = _TIME.sub(lambda m: f"{spoken_integer(int(m.group(1)))} giờ"
                     + (f" {spoken_integer(int(m.group(2)))}" if int(m.group(2)) else ""), text)
    text = _HOURS.sub(lambda m: f"{spoken_integer(int(m.group(1)))} giờ"
                      + (f" {spoken_integer(int(m.group(2)))}" if m.group(2) and int(m.group(2)) else ""), text)
    text = _FULL_DATE.sub(_date, text)
    text = _SLASH.sub(_slash, text)
    text = _PHONE.sub(lambda m: _digits(m.group(0)), text)
    text = _DEGREES.sub(lambda m: f"{m.group(1)} độ" + (f" {m.group(2)}" if m.group(2) else ""), text)
    text = _NEGATIVE.sub("âm ", text)
    text = _TIMES.sub(lambda m: " nhân " if m.start() == 0 or not text[m.start() - 1].isdigit() else " lần", text)
    units = dict(_UNIT_WORDS)
    text = _UNITS_AFTER.sub(lambda m: f"{m.group(1)} {units[m.group(2)]}", text)
    text = re.sub(r"(\d)(?=[^\W\d_])", r"\1 ", text)
    text = _PERCENT.sub(lambda m: f"{_number(m.group(1))} phần trăm", text)
    text = _DOLLARS.sub(lambda m: f"{_number(m.group(1))} đô", text)
    return re.sub(r"\s{2,}", " ", _PLAIN.sub(lambda m: _number(m.group(0)), text)).strip()
