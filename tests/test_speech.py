import time

import numpy as np

from screen_translator.config import SettingsStore
from screen_translator.pipeline import Pipeline
from screen_translator.speech import CHUNK, RATE, Segmenter, fbank, stack_frames

from test_app import EchoTranslator, store  # noqa: F401


def block(prob):
    return np.full(CHUNK, prob, np.float32)


class FakeVad:
    """Reads the speech probability straight out of the block."""

    def __call__(self, chunk):
        return float(chunk[0])

    def reset(self):
        pass


def test_features_have_the_shape_sensevoice_expects():
    second = np.sin(np.arange(RATE) * 2 * np.pi * 440 / RATE).astype(np.float32) * 0.1
    features = fbank(second)
    assert features.shape == (98, 80)
    assert stack_frames(features).shape == (17, 560)


def test_an_utterance_keeps_a_little_lead_in_and_ends_after_a_short_pause():
    segmenter = Segmenter(FakeVad(), silence_s=0.25, partial_s=10, pad_s=0.1)
    quiet, voiced = [block(0.0)] * 5, [block(0.9)] * 10
    results = [segmenter.feed(b) for b in quiet + voiced + [block(0.0)] * 10]
    heard = [r for r in results if r]
    assert [r.final for r in heard] == [False, True]
    assert len(heard[0].samples) == (3 + 10 + 1) * CHUNK
    assert len(heard[1].samples) == (3 + 10 + 3) * CHUNK
    assert not segmenter.speaking


def test_long_speech_gives_live_partials_and_is_cut_at_the_limit():
    segmenter = Segmenter(FakeVad(), partial_s=0.5, max_s=2.0, pad_s=0)
    heard = [r for r in (segmenter.feed(block(0.9)) for _ in range(130)) if r]
    assert [r.final for r in heard] == [False, False, False, True, False, False, False, True]


class FakeAudio:
    def __init__(self, blocks, delay=0.0):
        self.blocks, self.delay = list(blocks), delay

    def read(self, timeout=0.5):
        if not self.blocks:
            time.sleep(0.01)
            return None
        time.sleep(self.delay)
        return self.blocks.pop(0)

    def close(self):
        pass


class LengthRecognizer:
    """Short speech comes out as `short`, longer speech as `long`; the language is fixed per script."""

    def __init__(self, short, long, lang="ja", cutoff=12):
        self.short, self.long, self.lang, self.cutoff = short, long, lang, cutoff

    def recognize(self, samples, lang):
        return (self.short if len(samples) // CHUNK <= self.cutoff else self.long), self.lang


def run_speech(store, audio, recognizer, translator, seconds=3.0):
    store.update({"scope": "speech", "source_lang": "ja", "target_lang": "en"})
    events = []
    pipeline = Pipeline(store, events.append, translator_factory=lambda settings: translator,
                        audio_factory=lambda: audio, recognizer_factory=lambda: (recognizer, FakeVad()))
    from screen_translator.models import SPEECH

    original = type(SPEECH).downloaded
    type(SPEECH).downloaded = property(lambda self: True)
    try:
        pipeline.start()
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and not any(e["type"] == "line" and e["final"] for e in events):
            time.sleep(0.02)
        time.sleep(0.2)
        pipeline.stop()
    finally:
        type(SPEECH).downloaded = original
    return events, pipeline


def speech(voiced_blocks):
    return [block(0.0)] * 3 + [block(0.9)] * voiced_blocks + [block(0.0)] * 12


def test_a_sentence_is_translated_once_when_the_live_caption_already_matched_it(store):
    translator = EchoTranslator()
    events, pipeline = run_speech(store, FakeAudio(speech(10), delay=0.01), LengthRecognizer("こんにちは。", "こんにちは", cutoff=15),
                                  translator)
    finals = [(e["source"], e["translation"]) for e in events if e["type"] == "line" and e["final"]]
    assert finals == [("こんにちは", "EN(こんにちは。)")]
    assert len(translator.calls) == 1
    assert [t.source for t in pipeline.context.turns()] == ["こんにちは"]


def test_speech_already_in_the_target_language_is_shown_as_said(store):
    translator = EchoTranslator()
    events, _ = run_speech(store, FakeAudio(speech(10)), LengthRecognizer("Hold on.", "Hold on.", "en"), translator)
    finals = [(e["source"], e["translation"]) for e in events if e["type"] == "line" and e["final"]]
    assert finals == [("Hold on.", "Hold on.")]
    assert translator.calls == []


class SlowOnShortText(EchoTranslator):
    def translate(self, messages, on_partial=None):
        self.calls.append(messages)
        text = messages[-1]["content"]
        if text == "待って":
            for _ in range(100):
                time.sleep(0.02)
                on_partial("Wait…")
        return f"EN({text})"


def test_a_finished_sentence_cuts_short_a_stale_live_translation(store):
    translator = SlowOnShortText()
    started = time.monotonic()
    events, _ = run_speech(store, FakeAudio(speech(10), delay=0.005),
                           LengthRecognizer("待って", "待って、どこに行くの", cutoff=15), translator)
    finals = [e["translation"] for e in events if e["type"] == "line" and e["final"]]
    assert finals == ["EN(待って、どこに行くの)"]
    assert time.monotonic() - started < 1.5
