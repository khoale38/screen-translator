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
