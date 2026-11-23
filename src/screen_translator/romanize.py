"""Character names in Latin letters, by fixed rules: kana in Hepburn, hangul in Revised Romanization.

A small model asked for a name in Latin letters often returns it unchanged, or translates its meaning.
"""

from __future__ import annotations

import re

_KANA = {
    "あ": "a", "い": "i", "う": "u", "え": "e", "お": "o", "か": "ka", "き": "ki", "く": "ku", "け": "ke", "こ": "ko",
    "さ": "sa", "し": "shi", "す": "su", "せ": "se", "そ": "so", "た": "ta", "ち": "chi", "つ": "tsu", "て": "te",
    "と": "to", "な": "na", "に": "ni", "ぬ": "nu", "ね": "ne", "の": "no", "は": "ha", "ひ": "hi", "ふ": "fu",
    "へ": "he", "ほ": "ho", "ま": "ma", "み": "mi", "む": "mu", "め": "me", "も": "mo", "や": "ya", "ゆ": "yu",
    "よ": "yo", "ら": "ra", "り": "ri", "る": "ru", "れ": "re", "ろ": "ro", "わ": "wa", "を": "o", "ん": "n",
    "が": "ga", "ぎ": "gi", "ぐ": "gu", "げ": "ge", "ご": "go", "ざ": "za", "じ": "ji", "ず": "zu", "ぜ": "ze",
    "ぞ": "zo", "だ": "da", "ぢ": "ji", "づ": "zu", "で": "de", "ど": "do", "ば": "ba", "び": "bi", "ぶ": "bu",
    "べ": "be", "ぼ": "bo", "ぱ": "pa", "ぴ": "pi", "ぷ": "pu", "ぺ": "pe", "ぽ": "po", "ゔ": "vu",
    "ぁ": "a", "ぃ": "i", "ぅ": "u", "ぇ": "e", "ぉ": "o",
}
_DIGRAPHS = {"ゃ": "a", "ゅ": "u", "ょ": "o"}

_INITIALS = ["g", "kk", "n", "d", "tt", "r", "m", "b", "pp", "s", "ss", "", "j", "jj", "ch", "k", "t", "p", "h"]
_VOWELS = ["a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae", "oe", "yo", "u", "wo", "we", "wi", "yu",
           "eu", "ui", "i"]
_FINALS = ["", "k", "k", "k", "n", "n", "n", "t", "l", "k", "m", "l", "l", "l", "p", "l", "m", "p", "p", "t", "t",
           "ng", "t", "t", "k", "t", "p", "t"]
_SURNAMES = {"김": "Kim", "이": "Lee", "박": "Park", "최": "Choi", "정": "Jung", "강": "Kang", "조": "Cho", "윤": "Yoon",
             "장": "Jang", "임": "Lim", "한": "Han", "오": "Oh", "서": "Seo", "신": "Shin", "권": "Kwon", "황": "Hwang",
             "안": "Ahn", "송": "Song", "유": "Yoo", "홍": "Hong"}

KANA_NAME = re.compile("[ぁ-ヿー・]+")
HANGUL_NAME = re.compile("[가-힣 ]+")
NOT_LATIN = re.compile("[぀-ヿ㐀-鿿가-힯]")


def _kana(name: str) -> str:
    text = "".join(chr(ord(ch) - 0x60) if "ァ" <= ch <= "ヶ" else ch for ch in name.replace("・", " "))
    sounds: list[str] = []
    double = False
    for ch in text:
        if ch == "っ":
            double = True
            continue
        if ch == "ー":
            continue
        if ch in _DIGRAPHS and sounds and sounds[-1].endswith("i"):
            last = sounds.pop()
            sounds.append(last[:-1] + _DIGRAPHS[ch] if last in ("shi", "chi", "ji") else last[:-1] + "y" + _DIGRAPHS[ch])
            continue
        sound = _KANA.get(ch, ch)
        if double and sound[:1].isalpha():
            sound = ("t" if sound.startswith("ch") else sound[0]) + sound
        double = False
        sounds.append(sound)
    # Names drop the long vowel in everyday spelling: Ryu, Shota, Kyoko.
    spelled = "".join(sounds).replace("ou", "o").replace("uu", "u")
    return " ".join(word.capitalize() for word in spelled.split())


def _syllables(word: str) -> str:
    out = ""
    for ch in word:
        code = ord(ch) - 0xAC00
        if not 0 <= code < 11172:
            out += ch
            continue
        out += _INITIALS[code // 588] + _VOWELS[code % 588 // 28] + _FINALS[code % 28]
    return out


def _hangul(name: str) -> str:
    words = name.split()
    if len(words) == 1 and len(name) == 3 and name[0] in _SURNAMES:
        return f"{_SURNAMES[name[0]]} {_syllables(name[1:]).capitalize()}"
    return " ".join(_syllables(word).capitalize() for word in words)


# Kinship and job titles used as labels: they are said in Vietnamese, not spelled out.
ROLES_VI = {
    "할머니": "Bà", "할아버지": "Ông", "엄마": "Mẹ", "어머니": "Mẹ", "아빠": "Bố", "아버지": "Bố", "선생님": "Thầy",
    "오빠": "Anh", "형": "Anh", "언니": "Chị", "누나": "Chị", "팀장": "Trưởng nhóm", "팀장님": "Trưởng nhóm",
    "부장": "Trưởng phòng", "부장님": "Trưởng phòng", "과장": "Trưởng ban", "과장님": "Trưởng ban", "대리": "Phó phòng",
    "대리님": "Phó phòng", "사장": "Giám đốc", "사장님": "Giám đốc",
    "おばあちゃん": "Bà", "おじいちゃん": "Ông", "お母さん": "Mẹ", "おかあさん": "Mẹ", "母": "Mẹ", "お父さん": "Bố",
    "おとうさん": "Bố", "父": "Bố", "先生": "Thầy", "せんせい": "Thầy", "お兄ちゃん": "Anh", "おにいちゃん": "Anh",
    "お姉ちゃん": "Chị", "おねえちゃん": "Chị", "师父": "Sư phụ", "师傅": "Sư phụ", "妈妈": "Mẹ", "爸爸": "Bố",
    "奶奶": "Bà", "爷爷": "Ông", "老师": "Thầy",
}


def role_label(name: str) -> str | None:
    """A title used as a speaker's label, in Vietnamese: 할머니 is Bà, "강 부장" is "Trưởng phòng Kang"."""
    name = name.strip()
    if name in ROLES_VI:
        return ROLES_VI[name]
    surname, _, title = name.partition(" ")
    if title in ROLES_VI and len(surname) == 1:
        return f"{ROLES_VI[title]} {_SURNAMES.get(surname) or romanized(surname) or surname}"
    return None


def romanized(name: str) -> str | None:
    """The name in Latin letters when it is written only in kana or hangul; None otherwise."""
    name = name.strip()
    if name in ROLES_VI or role_label(name):
        return None
    if KANA_NAME.fullmatch(name):
        return _kana(name)
    if HANGUL_NAME.fullmatch(name):
        return _hangul(name)
    return None
