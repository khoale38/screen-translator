"""Real-model checks for LlamaTranslator's saved-state speed-up. Opt in with ST_E2E=1 (slow, downloads a model)."""

import os
import threading
import time
from statistics import median

import pytest

from screen_translator.context import ContextWindow
from screen_translator.models import MODELS, download
from screen_translator.translator import LlamaTranslator, build_batch_messages, build_messages, label_speaker

pytestmark = pytest.mark.skipif(os.environ.get("ST_E2E") != "1", reason="set ST_E2E=1 to run")

LINES = ["Lose the cops.", "Collect the cash", "Deliver the Phoenix to Brian's garage",
         "Hold L1 to open your loadout", "Get in the car.", "Take out the guards."]
NOTES = "GTA gameplay: mission objectives, police chases, street slang."


@pytest.fixture(scope="module")
def translator():
    spec = MODELS[os.environ.get("ST_E2E_MODEL", "qwen3.5-2b")]
    if not spec.downloaded:
        download(spec, lambda done, total: None, threading.Event())
    model = LlamaTranslator(str(spec.path))
    model.temperature = 0
    yield model
    model.close()


def run_dialogue(translator, notes=NOTES):
    context = ContextWindow(8, 300)
    outputs, waits = [], []
    for text in LINES:
        started = time.monotonic()
        first = []
        reply = translator.translate(build_messages(text, context.turns(), "en", "vi", notes=notes),
                                     lambda _partial: first or first.append(time.monotonic() - started))
        waits.append(first[0] if first else time.monotonic() - started)
        translation = label_speaker(text, reply)
        context.add(text, translation)
        outputs.append(translation)
    return outputs, waits


def test_restoring_the_saved_state_keeps_translations_identical_and_is_faster(translator, monkeypatch):
    translator._snapshots.clear()
    monkeypatch.setattr(translator, "min_shared_tokens", 10**9)
    plain, plain_waits = run_dialogue(translator)
    assert translator._snapshots == {}

    monkeypatch.setattr(translator, "min_shared_tokens", LlamaTranslator.min_shared_tokens)
    restores = []
    load_state = translator._llm.load_state
    monkeypatch.setattr(translator._llm, "load_state", lambda state: restores.append(1) or load_state(state))
    fast, fast_waits = run_dialogue(translator)

    print(f"\nfirst word, lines 2+: re-read {median(plain_waits[1:]) * 1000:.0f} ms, "
          f"restored {median(fast_waits[1:]) * 1000:.0f} ms")
    for source, a, b in zip(LINES, plain, fast):
        print(f"  {source!r}: {a!r} | {b!r}")
    assert fast == plain
    assert len(translator._snapshots) == 1
    assert len(restores) == len(LINES) - 1
    assert median(fast_waits[1:]) < 0.8 * median(plain_waits[1:])


def test_each_instruction_set_gets_its_own_saved_state(translator):
    translator._snapshots.clear()
    translator.translate(build_messages("Collect the cash", [], "en", "vi", notes=NOTES))
    translator.translate(build_messages("Collect the cash", [], "en", "vi", notes="A Korean drama"))
    translator.translate(build_batch_messages(["Settings", "Quit"], [], "en", "vi", notes=NOTES))
    assert len(translator._snapshots) == 3

    translator.translate(build_messages("Collect the cash", [], "en", "ja", notes=NOTES))
    assert len(translator._snapshots) == LlamaTranslator.max_snapshots

    restored = translator.translate(build_messages("Collect the cash", [], "en", "ja", notes=NOTES))
    translator._snapshots.clear()
    assert restored == translator.translate(build_messages("Collect the cash", [], "en", "ja", notes=NOTES))
