"""The frozen acceptance set: every case here was found by the scenario rounds and decided once.

A change that makes one of these fail is either a regression or a new decision, which updates the data file on purpose.
"""

import json
import pathlib

import pytest

from screen_translator.ocr import OcrLine
from screen_translator.text import group_blocks
from screen_translator.translator import is_name_tab

HERE = pathlib.Path(__file__).parent / "acceptance"
SCREENS = json.loads((HERE / "screens.json").read_text())


@pytest.mark.parametrize("screen", SCREENS, ids=[s["name"] for s in SCREENS])
def test_saved_screens_group_as_approved(screen):
    lines = [OcrLine(text, confidence, tuple(box)) for text, confidence, box in screen["lines"]]
    blocks = group_blocks(lines, screen["width"], screen["height"])
    got = [b.text + (" [TAB]" if is_name_tab(b.text) else "") for b in blocks]
    assert got == screen["expect"]


def _cue(case):
    import types

    from screen_translator.address import Relationships, guess_facts, vietnamese

    class Judge:
        def answer(self, question):
            return "{}"

    settings = types.SimpleNamespace(target_lang="vi", notes="")
    if case.get("speech"):
        from screen_translator.pipeline import _punctuated

        case = {**case, "line": _punctuated(case["line"])}
    if case.get("context") or case.get("gender") or "\n" in case["line"]:
        rel = Relationships(lambda: Judge(), lambda: settings, lambda e: None)
        try:
            for earlier in case.get("context", []):
                rel.heard(earlier)
            address = rel.address(case["line"], case.get("gender", ""))
            return [address.me, address.you] if address else None
        finally:
            rel.close()
    facts = guess_facts(case["line"], case.get("speaker", "A"), case.get("listener", "B"))
    return list(vietnamese(facts, case.get("speaker", "A"), case.get("listener", "B"))) if facts else None


def _group(case):
    from screen_translator.text import assemble

    width, height = case["frame"]
    lines = [OcrLine(text, 0.99, (x / width, y / height, w / width, h / height)) for text, x, y, w, h in case["rows"]]
    if case.get("mode") == "area":
        return assemble(lines, width=width, height=height, menus=True)
    return [b.text for b in group_blocks(lines, width, height)]


def _run(case):
    from screen_translator.address import speaker_of
    from screen_translator.pipeline import _punctuated
    from screen_translator.translator import label_speaker, sentence_case
    from screen_translator.vi_numbers import spoken_numbers

    kind = case["kind"]
    if kind == "cue":
        return _cue(case)
    if kind == "group":
        return _group(case)
    if kind == "number":
        return spoken_numbers(case["text"])
    if kind == "speaker":
        return speaker_of(case["text"])
    if kind == "label":
        return label_speaker(case["source"], case["translation"], case.get("names", {}))
    if kind == "case":
        return sentence_case(case["text"])
    if kind == "punctuated":
        return _punctuated(case["text"])
    raise ValueError(f"unknown kind {kind}")


CASES = [case for path in sorted(HERE.glob("cases_*.json")) for case in json.loads(path.read_text())]


@pytest.mark.parametrize("case", CASES, ids=[f"{c.get('area', '')}-{i}" for i, c in enumerate(CASES)])
def test_decided_cases(case):
    got = _run(case)
    accepted = [case["expect"], *case.get("accept", [])]
    assert got in accepted, f"{case.get('why', '')}: got {got!r}"
