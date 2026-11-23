import time
from types import SimpleNamespace

import numpy as np

from screen_translator.dub import (
    Delivery, Dubber, Line, Voice, delivery, gender_of, pick_voice, voice_choices,
)

VOICES = [
    Voice("an", "Microsoft An", "vi-VN", "male"),
    Voice("david", "Microsoft David", "en-US", "male"),
    Voice("zira", "Microsoft Zira", "en-US", "female"),
    Voice("hazel", "Microsoft Hazel", "en-GB", "female"),
    Voice("george", "Microsoft George", "en-GB", "male"),
]


def test_the_voice_follows_the_speaker_and_the_chosen_accent():
    assert pick_voice(VOICES, "en", "female").id == "zira"
    assert pick_voice(VOICES, "en", "male", "accent:en-GB").id == "george"
    assert pick_voice(VOICES, "en", "female", "voice:david").id == "david"
    assert pick_voice(VOICES, "vi", "female").id == "an"
    assert pick_voice(VOICES, "ja", "female") is None


def test_the_voice_menu_offers_automatic_accents_and_voices():
    labels = [label for _value, label in voice_choices(VOICES, "en")]
    assert labels[:3] == ["Match the speaker", "Match the speaker · UK accent", "Match the speaker · US accent"]
    assert "Microsoft Hazel · en-GB · female" in labels
    assert voice_choices(VOICES, "vi") == [("", "Match the speaker"), ("voice:an", "Microsoft An · vi-VN · male")]


def test_delivery_carries_emotion_voice_height_and_catches_up_when_behind():
    assert gender_of(110) == "male" and gender_of(230) == "female" and gender_of(0) == ""
    assert delivery("", 0, "") == Delivery(1.0, 0, 1.0)
    sad = delivery("SAD", 0, "female")
    assert sad.pace < 1 and sad.pitch < 0 and sad.loudness < 1
    angry = delivery("ANGRY", 0, "male")
    assert angry.loudness > 1 and angry.pace > 1
    assert delivery("", 260, "female").pitch > 0 > delivery("", 95, "male").pitch
    assert delivery("", 0, "", backlog=2).pace == 1.3


class FakeVoices:
    voices = VOICES

    def __init__(self):
        self.spoken = []

    def synthesize(self, text, voice, how):
        self.spoken.append((text, voice.id, how))
        return np.zeros(2205, np.float32), 22050


class Ducking:
    log = []

    def __init__(self, mode):
        self.mode = mode

    def __enter__(self):
        Ducking.log.append(("down", self.mode))

    def __exit__(self, *_):
        Ducking.log.append(("up", self.mode))


def make_dubber(voice="", original="duck"):
    voices, played, events = FakeVoices(), [], []
    settings = SimpleNamespace(dub_voice=voice, dub_original=original)
    dubber = Dubber(lambda: settings, voices_factory=lambda: voices,
                    player=lambda audio, rate: played.append(len(audio) / rate), original_sound=Ducking,
                    emit=events.append)
    return dubber, voices, played, events


def wait(predicate, seconds=2.0):
    deadline = time.monotonic() + seconds
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)


def test_each_line_is_spoken_with_the_game_turned_down_and_the_listener_told_to_ignore_it():
    Ducking.log.clear()
    dubber, voices, played, events = make_dubber()
    dubber.say(Line("Wait, where are you going?", "en", "SURPRISED", 230.0))
    dubber.say(Line("Get in the car.", "en", "ANGRY", 110.0))
    wait(lambda: len(Ducking.log) == 2)
    assert dubber.speaking()
    assert [(text, voice) for text, voice, _how in voices.spoken] == [
        ("Wait, where are you going?", "zira"), ("Get in the car.", "david")]
    assert voices.spoken[0][2].pitch > 0 and voices.spoken[1][2].loudness > 1
    assert Ducking.log == [("down", "duck"), ("up", "duck")]
    assert [e["voice"] for e in events] == ["Microsoft Zira", "Microsoft David"]
    dubber.close()


def test_a_missing_voice_is_reported_instead_of_crashing():
    dubber, _voices, played, events = make_dubber()
    dubber.say(Line("待って", "ja", "", 0.0))
    wait(lambda: events)
    assert "No voice for ja" in events[0]["error"] and not played
    dubber.close()


def test_natural_voices_come_first_once_downloaded(monkeypatch):
    from screen_translator import models
    from screen_translator.dub import CombinedVoices
    from screen_translator.natural_voices import NaturalVoices

    natural = NaturalVoices(folder=models.models_dir())
    combined = CombinedVoices(FakeVoices(), natural)
    monkeypatch.setattr(type(models.NATURAL_VOICES), "downloaded", property(lambda self: False))
    assert pick_voice(combined.voices, "vi", "female").id == "an"
    monkeypatch.setattr(type(models.NATURAL_VOICES), "downloaded", property(lambda self: True))
    assert pick_voice(combined.voices, "vi", "female").id == "natural:vi:F1"
    assert pick_voice(combined.voices, "ja", "male").id == "natural:ja:M1"
    assert pick_voice(combined.voices, "ms", "male").id == "natural:ms:M1"
    assert pick_voice(combined.voices, "zh", "female") is None
    assert [label for _v, label in voice_choices(combined.voices, "vi")][1:3] == [
        "Natural voice 1 · vi · female", "Natural voice 1 · vi · male"]


def test_text_is_prepared_the_way_supertonic_reads_it():
    from screen_translator.natural_voices import pieces, prepare

    assert prepare("Wait — “now”", "en") == '<en>Wait - "now"</en>'
    assert prepare("Get in", "en") == "<en>Get in.</en>"
    assert prepare("Đi thôi!", "vi").startswith("<vi>") and prepare("Đi thôi!", "vi").endswith("!</vi>")
    long = "One sentence here. " * 30
    assert all(len(piece) <= 300 for piece in pieces(long, "en")) and len(pieces(long, "en")) > 1
    assert pieces("短い。", "ja") == ["短い。"]


def test_a_dub_may_say_what_fits_the_time_the_speaker_took():
    from screen_translator.dub import speakable

    assert speakable("vi", 4.0) == "13 words"
    assert speakable("ja", 2.0).endswith("characters")


def test_dubbed_speech_asks_for_an_interpreters_condensed_line():
    from screen_translator.translator import build_messages, for_hy_mt

    messages = build_messages("I have a bucket, I have a sponge.", [], "en", "vi", budget="9 words")
    assert "at most 9 words" in messages[-1]["content"] and messages[-1]["item"] == "I have a bucket, I have a sponge."
    assert "不超过9个词" in for_hy_mt(messages)[0]["content"]
    assert "不超过" not in for_hy_mt(build_messages("Hi", [], "en", "vi"))[0]["content"]


class SlowVoices:
    """Speaks everything at 4 s regardless of pace, like a voice that cannot be hurried any further."""

    def __init__(self, voice_id):
        self.voices = [Voice(voice_id, "Voice", "vi", "female")]
        self.calls = []

    def synthesize(self, text, voice, how):
        self.calls.append(how)
        return np.zeros(4 * 16000, np.float32), 16000


def test_a_natural_voice_already_fitted_to_its_line_is_not_voiced_again():
    for voice_id, calls in (("natural:vi:F1", 1), ("an", 2)):
        voices, played = SlowVoices(voice_id), []
        dubber = Dubber(lambda: SimpleNamespace(dub_voice="", dub_original="keep"), voices_factory=lambda: voices,
                        player=lambda audio, rate: played.append(1), original_sound=Ducking)
        dubber.say(Line("Trưởng bộ môn đã rất giận.", "vi", "", 200.0, seconds=2.0))
        wait(lambda: played)
        dubber.close()
        assert len(voices.calls) == calls


def test_speech_tells_the_translator_who_is_speaking_by_their_voice():
    from screen_translator.pipeline import Pipeline

    pipeline = Pipeline.__new__(Pipeline)
    pipeline._last_voice = None
    pipeline._previous_voice, pipeline._previous_pitch = "", 0.0
    assert pipeline._voice_note({"id": 1, "pitch": 210.0}).startswith("Voice: a woman.")
    assert pipeline._voice_note({"id": 2, "pitch": 110.0}).startswith("Voice: a man (the line before was a woman's).")
    assert pipeline._voice_note({"id": 2, "pitch": 115.0}).startswith("Voice: a man (the line before was a woman's).")
    assert pipeline._voice_note({"id": 3, "pitch": 0.0}) == ""


def test_two_men_with_clearly_different_voices_count_as_different_people():
    from screen_translator.pipeline import Pipeline

    pipeline = Pipeline.__new__(Pipeline)
    pipeline._last_voice, pipeline._previous_voice, pipeline._previous_pitch = None, "", 0.0
    pipeline._voice_note({"id": 1, "pitch": 105.0})
    assert "different person" in pipeline._voice_note({"id": 2, "pitch": 80.0})
    assert "different" not in pipeline._voice_note({"id": 3, "pitch": 84.0})


def test_each_voice_gets_a_stand_in_name_so_speech_keeps_who_talks_to_whom():
    from screen_translator.pipeline import Pipeline

    pipeline = Pipeline.__new__(Pipeline)
    pipeline._voices, pipeline._voice_labels = [], {}
    assert pipeline._voiced({"id": 1, "text": "오빠, 어디야?", "pitch": 220.0}) == "Woman: 오빠, 어디야?"
    assert pipeline._voiced({"id": 2, "text": "집이야.", "pitch": 110.0}) == "Man: 집이야."
    assert pipeline._voiced({"id": 3, "text": "Hi.", "pitch": 80.0}) == "Man 2: Hi."
    assert pipeline._voiced({"id": 4, "text": "Okay.", "pitch": 115.0}) == "Man: Okay."
    assert pipeline._voiced({"id": 5, "text": "...", "pitch": 0.0}) == "..."
