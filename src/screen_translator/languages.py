LANGUAGES: dict[str, str] = {
    "auto": "Auto-detect",
    "en": "English",
    "ja": "Japanese",
    "zh": "Chinese (Simplified)",
    "zh-Hant": "Chinese (Traditional)",
    "ko": "Korean",
    "vi": "Vietnamese",
    "th": "Thai",
    "id": "Indonesian",
    "ms": "Malay",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "pl": "Polish",
    "tr": "Turkish",
    "ru": "Russian",
    "uk": "Ukrainian",
    "ar": "Arabic",
    "hi": "Hindi",
}


def name(code: str) -> str:
    return LANGUAGES.get(code, code)


def is_spaceless(code: str) -> bool:
    return code in {"ja", "zh", "zh-Hant", "th"}


_SCRIPTS = {
    "ko": ((0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F)),
    "ru": ((0x0400, 0x04FF),),
    "uk": ((0x0400, 0x04FF),),
    "th": ((0x0E00, 0x0E7F),),
    "ar": ((0x0600, 0x06FF), (0x0750, 0x077F)),
    "hi": ((0x0900, 0x097F),),
}
_KANA = ((0x3040, 0x30FF), (0x31F0, 0x31FF))
_HAN = ((0x4E00, 0x9FFF), (0x3400, 0x4DBF))
_VIETNAMESE = set("đăơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ")


def _share(letters: list[str], ranges) -> float:
    return sum(any(lo <= ord(ch) <= hi for lo, hi in ranges) for ch in letters) / len(letters)


def already_in(code: str, text: str) -> bool:
    """Cheap check for text that is plainly in the target language already, so it is left alone, not rewritten."""
    letters = [ch for ch in text.lower() if ch.isalpha()]
    if not letters:
        return False
    if code in _SCRIPTS:
        return _share(letters, _SCRIPTS[code]) >= 0.6
    if code == "ja":
        return _share(letters, _KANA) > 0 and _share(letters, _KANA + _HAN) >= 0.6
    if code in ("zh", "zh-Hant"):
        return _share(letters, _KANA) == 0 and _share(letters, _HAN) >= 0.6
    if code == "vi":
        marked = sum(ch in _VIETNAMESE for ch in letters)
        return marked >= 1 and marked / len(letters) >= 0.04
    return False
