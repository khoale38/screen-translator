"""Asks GitHub once whether a newer published release exists. Sends nothing but the request itself."""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version

LATEST = "https://api.github.com/repos/khoale38/screen-translator/releases/latest"


@dataclass
class Release:
    version: str
    url: str


def installed_version() -> str:
    try:
        return version("screen-translator")
    except PackageNotFoundError:
        return "0"


def _parts(text: str) -> tuple[int, ...] | None:
    match = re.fullmatch(r"v?(\d+(?:\.\d+)*)", text.strip())
    return tuple(int(p) for p in match.group(1).split(".")) if match else None


def is_newer(candidate: str, current: str) -> bool:
    new, old = _parts(candidate), _parts(current)
    if new is None or old is None:
        return False
    width = max(len(new), len(old))
    return new + (0,) * (width - len(new)) > old + (0,) * (width - len(old))


def check(current: str | None = None, timeout: float = 5.0) -> Release | None:
    """The latest release if it is newer than `current`; None when up to date, offline or rate-limited."""
    request = urllib.request.Request(LATEST, headers={"Accept": "application/vnd.github+json",
                                                      "User-Agent": "ScreenTranslator"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            release = json.load(response)
    except (OSError, ValueError):
        return None
    tag = str(release.get("tag_name", ""))
    if release.get("draft") or release.get("prerelease") or not is_newer(tag, current or installed_version()):
        return None
    return Release(tag.lstrip("v"), str(release.get("html_url") or "https://github.com/khoale38/screen-translator/releases/latest"))
