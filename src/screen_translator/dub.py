"""Dubbing: speaks each translated line with a natural voice (once downloaded) or one already installed in the OS.

The voice follows the speaker: male or female from their pitch, higher or lower within that, and the emotion
SenseVoice heard (excited, sad, angry...) shifts pace, pitch and loudness.
"""

from __future__ import annotations

import asyncio
import io
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from html import escape
from typing import Callable

import numpy as np

MALE_BELOW_HZ = 150.0
TYPICAL_HZ = {"male": 120.0, "female": 210.0}

# (pace, pitch %, loudness) per emotion SenseVoice can hear.
EXPRESSION = {
    "HAPPY": (1.06, 8, 1.0),
    "SAD": (0.88, -6, 0.8),
    "ANGRY": (1.08, 3, 1.25),
    "FEARFUL": (1.1, 10, 0.9),
    "DISGUSTED": (0.95, -4, 1.0),
    "SURPRISED": (1.04, 12, 1.1),
}

ORIGINAL_SOUND = {"duck": 0.25, "mute": 0.0, "keep": 1.0}

MAX_PACE = 1.7
# A line heard longer ago than this is skipped once a newer one waits, so the voice stays with the speaker.
STALE_S = 5.0
# How much the natural voices say per second at normal pace, measured; zh, ja and th in characters, others in words.
SPOKEN_PER_S = {"vi": 2.2, "en": 3.2, "es": 2.7, "ko": 1.9, "ja": 5.5, "zh": 5.0, "th": 7.0}
CHARACTER_LANGS = {"ja", "zh", "th"}


def speakable(target: str, seconds: float) -> str:
    """How much a dub can say in `seconds` at the fastest pace that still sounds natural, e.g. "12 words"."""
    lang = _base(target)
    count = max(3, round(seconds * SPOKEN_PER_S.get(lang, 2.7) * MAX_PACE * 0.9))
    return f"{count} characters" if lang in CHARACTER_LANGS else f"{count} words"

REGIONS = {"US": "US", "GB": "UK", "AU": "Australian", "IE": "Irish", "IN": "Indian", "ZA": "South African",
           "CA": "Canadian", "NZ": "New Zealand", "SC": "Scottish", "ES": "Spain", "MX": "Mexican",
           "BR": "Brazilian", "PT": "Portugal", "FR": "France", "BE": "Belgian", "CH": "Swiss", "CN": "Mainland",
           "TW": "Taiwan", "HK": "Hong Kong"}


def accent_name(locale: str) -> str:
    region = locale.split("-")[-1].upper()
    return f"{REGIONS.get(region, region)} accent"


@dataclass(frozen=True)
class Voice:
    id: str
    name: str
    lang: str
    gender: str = ""

    @property
    def label(self) -> str:
        return f"{self.name} · {self.lang}" + (f" · {self.gender}" if self.gender else "")


@dataclass(frozen=True)
class Delivery:
    pace: float = 1.0
    pitch: int = 0
    loudness: float = 1.0
    fit_s: float = 0.0


def gender_of(pitch: float) -> str:
    return "" if pitch <= 0 else "male" if pitch < MALE_BELOW_HZ else "female"


def delivery(emotion: str, pitch: float, gender: str, backlog: int = 0, fit_s: float = 0.0) -> Delivery:
    """How to say a line: the speaker's emotion and voice height, sped up a little when lines are piling up."""
    pace, shift, loudness = EXPRESSION.get(emotion, (1.0, 0, 1.0))
    if pitch > 0 and gender in TYPICAL_HZ:
        shift += round(max(-12.0, min(12.0, (pitch / TYPICAL_HZ[gender] - 1) * 40)))
    pace = min(MAX_PACE, pace * (1 + 0.15 * backlog))
    return Delivery(round(pace, 2), int(shift), loudness, fit_s)


def _base(lang: str) -> str:
    return {"zh-Hant": "zh"}.get(lang, lang).split("-")[0].lower()


def pick_voice(voices: list[Voice], target: str, gender: str, choice: str = "") -> Voice | None:
    """`choice` is "" (match the speaker), "accent:<locale>" (match the speaker in that accent) or "voice:<id>"."""
    if choice.startswith("voice:"):
        chosen = next((v for v in voices if v.id == choice[6:]), None)
        if chosen:
            return chosen
    matches = [v for v in voices if _base(v.lang) == _base(target)]
    if target == "zh-Hant":
        matches = [v for v in matches if v.lang.lower() in ("zh-tw", "zh-hk")] or matches
    if choice.startswith("accent:"):
        matches = [v for v in matches if v.lang.lower() == choice[7:].lower()] or matches
    return next((v for v in matches if gender and v.gender == gender), matches[0] if matches else None)


def voice_choices(voices: list[Voice], target: str) -> list[tuple[str, str]]:
    """(value, label) pairs for a voice menu: automatic, each accent, then each voice."""
    matches = [v for v in voices if _base(v.lang) == _base(target)]
    choices = [("", "Match the speaker")]
    # Natural voices are listed by bare language ("vi"); only regional system voices have an accent.
    accents = sorted({v.lang for v in matches if "-" in v.lang})
    if len(accents) > 1:
        choices += [(f"accent:{lang}", f"Match the speaker · {accent_name(lang)}") for lang in accents]
    return choices + [(f"voice:{v.id}", v.label) for v in matches]


def _wav_samples(data: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(data)) as clip:
        frames = clip.readframes(clip.getnframes())
        width, channels, rate = clip.getsampwidth(), clip.getnchannels(), clip.getframerate()
    kind = {1: np.uint8, 2: np.int16, 4: np.int32}[width]
    audio = np.frombuffer(frames, kind).astype(np.float32)
    audio = (audio - 128) / 128 if width == 1 else audio / float(2 ** (8 * width - 1))
    return audio.reshape(-1, channels).mean(axis=1), rate


class WindowsVoices:
    """Windows' own voices (Settings → Time & language → Speech adds more languages)."""

    def __init__(self) -> None:
        from winrt.windows.media.speechsynthesis import SpeechSynthesizer, VoiceGender

        self._synth = SpeechSynthesizer()
        self._infos = {info.id: info for info in SpeechSynthesizer.all_voices}
        self.voices = [
            Voice(info.id, info.display_name, info.language,
                  "female" if info.gender == VoiceGender.FEMALE else "male")
            for info in self._infos.values()
        ]

    def synthesize(self, text: str, voice: Voice, how: Delivery) -> tuple[np.ndarray, int]:
        return asyncio.run(self._synthesize(text, voice, how))

    async def _synthesize(self, text: str, voice: Voice, how: Delivery) -> tuple[np.ndarray, int]:
        from winrt.windows.storage.streams import DataReader

        self._synth.voice = self._infos[voice.id]
        ssml = (f"<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='{voice.lang}'>"
                f"<prosody rate='{how.pace:.2f}' pitch='{how.pitch:+d}%' volume='{min(100, 80 * how.loudness):.0f}'>"
                f"{escape(text)}</prosody></speak>")
        stream = await self._synth.synthesize_ssml_to_stream_async(ssml)
        reader = DataReader(stream.get_input_stream_at(0))
        size = await reader.load_async(stream.size)
        data = bytearray(size)
        reader.read_bytes(data)
        return _wav_samples(bytes(data))


class MacVoices:
    """macOS voices through `say`, for trying dubbing on a Mac."""

    def __init__(self) -> None:
        listing = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, check=True).stdout
        self.voices = []
        for line in listing.splitlines():
            match = re.match(r"^(.+?)\s+([a-z]{2,3}_[A-Z0-9]{2,3})\s+#", line)
            if match:
                self.voices.append(Voice(match.group(1).strip(), match.group(1).strip(), match.group(2).replace("_", "-")))

    def synthesize(self, text: str, voice: Voice, how: Delivery) -> tuple[np.ndarray, int]:
        with tempfile.TemporaryDirectory() as folder:
            out = os.path.join(folder, "line.wav")
            subprocess.run(["say", "-v", voice.id, "-r", str(round(185 * how.pace)), "-o", out,
                            "--file-format=WAVE", "--data-format=LEI16@22050", text], check=True)
            with open(out, "rb") as clip:
                audio, rate = _wav_samples(clip.read())
        return np.clip(audio * how.loudness, -1, 1), rate


class CombinedVoices:
    """Natural voices first, then the system's; natural voices appear as soon as their download finishes."""

    def __init__(self, system, natural=None) -> None:
        self.system, self.natural = system, natural

    def _sources(self) -> list:
        from .models import NATURAL_VOICES

        ready = self.natural is not None and NATURAL_VOICES.downloaded
        return [self.natural, self.system] if ready else [self.system]

    @property
    def voices(self) -> list[Voice]:
        return [voice for source in self._sources() for voice in source.voices]

    def synthesize(self, text: str, voice: Voice, how: Delivery) -> tuple[np.ndarray, int]:
        source = self.natural if voice.id.startswith("natural:") else self.system
        return source.synthesize(text, voice, how)


def default_voices():
    from .natural_voices import NaturalVoices

    return CombinedVoices(WindowsVoices() if sys.platform == "win32" else MacVoices(), NaturalVoices())


class OriginalSound:
    """Turns other apps down (or off) while a line is dubbed, then back to where they were."""

    def __init__(self, mode: str) -> None:
        self.level = ORIGINAL_SOUND.get(mode, ORIGINAL_SOUND["duck"])
        self._saved: list = []

    def __enter__(self) -> "OriginalSound":
        if self.level >= 1.0 or sys.platform != "win32":
            return self
        try:
            from pycaw.pycaw import AudioUtilities

            for session in AudioUtilities.GetAllSessions():
                if session.ProcessId and session.ProcessId != os.getpid():
                    volume = session.SimpleAudioVolume
                    before = volume.GetMasterVolume()
                    volume.SetMasterVolume(before * self.level, None)
                    self._saved.append((volume, before))
        except Exception as exc:
            print(f"Could not turn other apps down: {exc}", flush=True)
        return self

    def __exit__(self, *_exc) -> None:
        for volume, before in self._saved:
            try:
                volume.SetMasterVolume(before, None)
            except Exception:
                pass
        self._saved = []


def default_player(audio: np.ndarray, rate: int) -> None:
    import soundcard

    soundcard.default_speaker().play(audio, samplerate=rate)


@dataclass
class Line:
    text: str
    target: str
    emotion: str
    pitch: float
    seconds: float = 0.0
    heard_at: float = 0.0


class Dubber:
    """Speaks lines one after another on threads of their own, so translation never waits for the voice.

    The next line is voiced while the current one plays, so lines follow each other without a gap.
    """

    def __init__(self, settings: Callable, voices_factory=default_voices, player=default_player,
                 original_sound=OriginalSound, emit: Callable[[dict], None] = lambda event: None) -> None:
        self.settings = settings
        self.voices_factory = voices_factory
        self.player = player
        self.original_sound = original_sound
        self.emit = emit
        self._lines: queue.Queue[Line | None] = queue.Queue()
        self._ready: queue.Queue[tuple | None] = queue.Queue(maxsize=1)
        self._quiet_after = 0.0
        self._threads = [threading.Thread(target=self._prepare, name="dub-voice", daemon=True),
                         threading.Thread(target=self._play, name="dub-play", daemon=True)]
        for thread in self._threads:
            thread.start()

    def say(self, line: Line) -> None:
        self._lines.put(line)

    def speaking(self) -> bool:
        """True while a dubbed line is playing, and a moment after, so it is not heard as new speech."""
        return time.monotonic() < self._quiet_after

    def close(self) -> None:
        self._lines.put(None)
        for thread in self._threads:
            thread.join(timeout=2)

    def _waiting(self) -> int:
        return self._lines.qsize() + self._ready.qsize()

    @staticmethod
    def _stale(line: Line) -> bool:
        return bool(line.heard_at) and time.monotonic() - line.heard_at > STALE_S

    def _prepare(self) -> None:
        if sys.platform == "win32":
            sys.coinit_flags = 0
        voices = None
        while True:
            line = self._lines.get()
            if line is None:
                self._ready.put(None)
                return
            if self._stale(line) and self._waiting():
                continue
            try:
                voices = voices or self.voices_factory()
                voiced = self._voice(voices, line)
            except Exception as exc:
                self.emit({"type": "dub", "error": f"Could not speak the line: {exc}"})
                continue
            if voiced:
                self._ready.put(voiced)

    def _play(self) -> None:
        if sys.platform == "win32":
            sys.coinit_flags = 0
        while True:
            item = self._ready.get()
            if item is None:
                return
            with self.original_sound(self.settings().dub_original):
                while item is not None:
                    line, audio, rate = item
                    if not (self._stale(line) and self._waiting()):
                        self._quiet_after = time.monotonic() + len(audio) / rate + 0.3
                        try:
                            self.player(audio, rate)
                        except Exception as exc:
                            self.emit({"type": "dub", "error": f"Could not play the line: {exc}"})
                    try:
                        item = self._ready.get(timeout=0.4)
                    except queue.Empty:
                        item = None
                        break
                    if item is None:
                        self._quiet_after = time.monotonic() + 0.3
                        return
            self._quiet_after = time.monotonic() + 0.3

    def _voice(self, voices, line: Line) -> tuple | None:
        settings = self.settings()
        gender = gender_of(line.pitch)
        voice = pick_voice(voices.voices, line.target, gender, settings.dub_voice)
        if voice is None:
            self.emit({"type": "dub", "error": f"No voice for {line.target} is installed. On Windows, add one "
                                               "under Settings → Time & language → Speech."})
            return None
        how = delivery(line.emotion, line.pitch, voice.gender or gender, self._waiting(), line.seconds)
        started = time.monotonic()
        audio, rate = voices.synthesize(line.text, voice, how)
        if how.fit_s and len(audio) / rate > 1.05 * how.fit_s and how.pace < MAX_PACE:
            faster = min(MAX_PACE, how.pace * len(audio) / rate / how.fit_s)
            audio, rate = voices.synthesize(line.text, voice, Delivery(faster, how.pitch, how.loudness))
        self.emit({"type": "dub", "voice": voice.name, "ms": round((time.monotonic() - started) * 1000),
                   "seconds": round(len(audio) / rate, 1), "text": line.text})
        return line, audio, rate
