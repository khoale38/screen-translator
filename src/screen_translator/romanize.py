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
    if len(words) == 1 and len(name) == 3:
        surname = _SURNAMES.get(name[0], _syllables(name[0]).capitalize())
        return f"{surname} {_syllables(name[1:]).capitalize()}"
    return " ".join(_syllables(word).capitalize() for word in words)


def romanized(name: str) -> str | None:
    """The name in Latin letters when it is written only in kana or hangul; None otherwise."""
    name = name.strip()
    if KANA_NAME.fullmatch(name):
        return _kana(name)
    if HANGUL_NAME.fullmatch(name):
        return _hangul(name)
    return None
