"""Speech to text, fully local: Silero VAD finds where speech starts and stops, SenseVoice-Small writes it down.

Both run on the onnxruntime the app already ships for OCR; a second copy of onnxruntime.dll in the same process
breaks on Windows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np

from .dub import MALE_BELOW_HZ

RATE = 16000
CHUNK = 512  # 32 ms, the block size Silero VAD expects at 16 kHz

EMOTIONS = ("HAPPY", "SAD", "ANGRY", "FEARFUL", "DISGUSTED", "SURPRISED")


class Recognition(NamedTuple):
    text: str
    lang: str
    emotion: str = ""


SENSEVOICE_LANGS = {"zh": "zh", "zh-Hant": "zh", "en": "en", "ja": "ja", "ko": "ko", "yue": "yue"}
_SPECIAL = re.compile(r"<\|[^|]*\|>")
# Hangul is included: SenseVoice spaces Korean between syllables ("받 았 어"), which reads worse than no spaces.
_CJK = "\u3000-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uff00-\uffef"
_CJK_GAP = re.compile(f"(?<=[{_CJK}]) (?=[{_CJK}0-9])|(?<=[0-9]) (?=[{_CJK}])")


@lru_cache(maxsize=1)
def _mel_banks(n_fft: int = 512, n_mels: int = 80, low: float = 20.0, high: float = RATE / 2) -> np.ndarray:
    """Kaldi's triangular mel filters (what SenseVoice was trained with), shape (n_mels, n_fft // 2)."""
    mel = lambda f: 1127.0 * np.log(1.0 + f / 700.0)
    lo, hi = mel(low), mel(high)
    step = (hi - lo) / (n_mels + 1)
    bin_mel = mel(np.arange(n_fft // 2) * RATE / n_fft)
    banks = np.zeros((n_mels, n_fft // 2), np.float32)
    for m in range(n_mels):
        left, center, right = lo + m * step, lo + (m + 1) * step, lo + (m + 2) * step
        up = (bin_mel - left) / (center - left)
        down = (right - bin_mel) / (right - center)
        banks[m] = np.maximum(0.0, np.minimum(up, down))
    return banks


def fbank(samples: np.ndarray) -> np.ndarray:
    """80-dim log mel filterbank matching kaldi.fbank(dither=0, energy_floor=0, window=hamming, snip_edges=True)."""
    wave = samples.astype(np.float64) * 32768.0
    length, shift = 400, 160
    if len(wave) < length:
        return np.zeros((0, 80), np.float32)
    count = 1 + (len(wave) - length) // shift
    frames = np.lib.stride_tricks.sliding_window_view(wave, length)[::shift][:count].copy()
    frames -= frames.mean(axis=1, keepdims=True)
    frames[:, 1:] -= 0.97 * frames[:, :-1]
    frames[:, 0] *= 1.0 - 0.97
    frames *= np.hamming(length)
    power = np.abs(np.fft.rfft(frames, 512)) ** 2
    energies = power[:, :256] @ _mel_banks().T.astype(np.float64)
    return np.log(np.maximum(energies, np.finfo(np.float32).eps)).astype(np.float32)


def stack_frames(features: np.ndarray, window: int = 7, shift: int = 6) -> np.ndarray:
    """SenseVoice's low frame rate input: 7 frames side by side every 6 frames, padded at the start like FunASR."""
    if len(features) == 0:
        return np.zeros((0, features.shape[1] * window), np.float32)
    padded = np.concatenate([np.repeat(features[:1], (window - 1) // 2, axis=0), features])
    count = (len(features) + shift - 1) // shift
    need = (count - 1) * shift + window
    if len(padded) < need:
        padded = np.concatenate([padded, np.repeat(padded[-1:], need - len(padded), axis=0)])
    return np.stack([padded[i * shift:i * shift + window].reshape(-1) for i in range(count)])


def _session(path: Path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 4
    options.intra_op_num_threads = 2
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


class SenseVoice:
    """SenseVoice-Small: Chinese, English, Japanese, Korean and Cantonese, about 70 ms for 10 s of speech."""

    def __init__(self, model: Path, tokens: Path) -> None:
        self._session = _session(model)
        meta = self._session.get_modelmeta().custom_metadata_map
        self._neg_mean = np.array([float(v) for v in meta["neg_mean"].split(",")], np.float32)
        self._inv_std = np.array([float(v) for v in meta["inv_stddev"].split(",")], np.float32)
        self._window, self._shift = int(meta["lfr_window_size"]), int(meta["lfr_window_shift"])
        self._langs = {k[5:]: int(v) for k, v in meta.items() if k.startswith("lang_")}
        self._with_itn = int(meta["with_itn"])
        self._tokens = {}
        for line in tokens.read_text(encoding="utf-8").splitlines():
            piece, _, index = line.rpartition(" ")
            if index.isdigit():
                self._tokens[int(index)] = piece

    def transcribe(self, samples: np.ndarray, lang: str = "auto") -> str:
        return self.recognize(samples, lang)[0]

    def recognize(self, samples: np.ndarray, lang: str = "auto") -> Recognition:
        """The words, the language they were spoken in and the emotion heard, e.g. ("待って!", "ja", "ANGRY")."""
        features = stack_frames(fbank(samples), self._window, self._shift)
        if len(features) == 0:
            return Recognition("", "")
        features = (features + self._neg_mean) * self._inv_std
        language = self._langs.get(SENSEVOICE_LANGS.get(lang, "auto"), self._langs["auto"])
        (logits,) = self._session.run(None, {
            "x": features[None].astype(np.float32),
            "x_length": np.array([len(features)], np.int32),
            "language": np.array([language], np.int32),
            "text_norm": np.array([self._with_itn], np.int32),
        })
        best = logits[0].argmax(axis=-1)
        pieces, previous = [], -1
        for index in best:
            if index != previous and index != 0:
                pieces.append(self._tokens.get(int(index), ""))
            previous = index
        raw = "".join(pieces)
        tags = [tag[2:-2] for tag in _SPECIAL.findall(raw)]
        spoken = next((tag for tag in tags if tag in self._langs), "")
        emotion = next((tag for tag in tags if tag in EMOTIONS), "")
        text = " ".join(_SPECIAL.sub("", raw).replace("▁", " ").split())
        return Recognition(_CJK_GAP.sub("", text), spoken, emotion)


def pitch_hz(samples: np.ndarray) -> float:
    """Median voice pitch, from the autocorrelation of 40 ms frames; 0 when nothing sounds voiced."""
    length, hop = 640, 320
    low, high = RATE // 400, RATE // 70
    pitches = []
    for start in range(0, len(samples) - length, hop):
        frame = samples[start:start + length].astype(np.float64)
        frame -= frame.mean()
        energy = float(frame @ frame)
        if energy < 1e-3:
            continue
        corr = np.correlate(frame, frame, "full")[length - 1:]
        lag = low + int(np.argmax(corr[low:high]))
        if corr[lag] / energy > 0.45:
            pitches.append(RATE / lag)
    return float(np.median(pitches)) if pitches else 0.0


class Vad:
    """Silero VAD v4: the chance that a 32 ms block is speech."""

    def __init__(self, model: Path) -> None:
        self._session = _session(model)
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((2, 1, 64), np.float32)
        self._c = np.zeros((2, 1, 64), np.float32)

    def __call__(self, chunk: np.ndarray) -> float:
        prob, self._h, self._c = self._session.run(None, {
            "x": chunk.reshape(1, CHUNK).astype(np.float32), "h": self._h, "c": self._c,
        })
        return float(prob.reshape(-1)[0])


@dataclass
class Utterance:
    samples: np.ndarray
    final: bool
    ending: bool = False
    # Why a final utterance ended: "pause", "breath", "turn" (another voice took over) or "max".
    cut: str = ""


class Segmenter:
    """Cuts a stream of 32 ms blocks into utterances.

    An utterance ends after `silence_s` without speech, or at `max_s`. Fast talkers barely pause, so once an
    utterance is `long_s` old a breath of `breath_s` ends it too, and after `very_long_s` an even shorter
    `short_silence_s`; the breath is longer than the catch of a Japanese っ, which would otherwise split a word.
    A pause of `turn_s` also ends it when the voice after it is a man's and the one before a woman's, or the other way
    round. While someone is still talking it hands out the speech so far every `partial_s`, and once more the moment
    the voice drops, so its translation is usually done by the time the utterance is confirmed finished.
    """

    def __init__(self, vad, threshold: float = 0.5, silence_s: float = 0.45, max_s: float = 8.0,
                 partial_s: float = 0.5, pad_s: float = 0.2, long_s: float = 3.0, breath_s: float = 0.15,
                 very_long_s: float = 5.0, short_silence_s: float = 0.08, turn_s: float = 0.15,
                 pitch: Callable[[np.ndarray], float] | None = None) -> None:
        blocks = lambda seconds: max(1, round(seconds * RATE / CHUNK))
        self.vad = vad
        self.threshold = threshold
        self.silence_blocks = blocks(silence_s)
        self.max_blocks = round(max_s * RATE / CHUNK)
        self.long_blocks = round(long_s * RATE / CHUNK)
        self.breath_blocks = blocks(breath_s)
        self.very_long_blocks = round(very_long_s * RATE / CHUNK)
        self.short_silence_blocks = blocks(short_silence_s)
        self.turn_blocks = blocks(turn_s)
        self.turn_check_blocks = blocks(0.4)
        self.partial_blocks = blocks(partial_s)
        self.pad_blocks = round(pad_s * RATE / CHUNK)
        self.pitch = pitch or pitch_hz
        self._before: list[np.ndarray] = []
        self._speech: list[np.ndarray] = []
        self._quiet = 0
        self._since_partial = 0
        self._turn: tuple[int, int] | None = None

    @property
    def speaking(self) -> bool:
        return bool(self._speech)

    def feed(self, chunk: np.ndarray) -> Utterance | None:
        voiced = self.vad(chunk) >= self.threshold
        if not self._speech:
            if not voiced:
                self._before = (self._before + [chunk])[-self.pad_blocks:] if self.pad_blocks else []
                return None
            self._speech, self._before = self._before + [chunk], []
            self._quiet = self._since_partial = 0
            self._turn = None
            return None
        self._speech.append(chunk)
        if voiced:
            if self._quiet >= self.turn_blocks:
                self._turn = (len(self._speech) - 1 - self._quiet, len(self._speech) - 1)
            self._quiet = 0
        else:
            self._quiet += 1
        length = len(self._speech)
        needed = (self.short_silence_blocks if length >= self.very_long_blocks
                  else self.breath_blocks if length >= self.long_blocks else self.silence_blocks)
        if self._quiet >= needed:
            return self._finish("pause" if needed == self.silence_blocks else "breath")
        if length >= self.max_blocks:
            return self._finish("max")
        if self._turn and length - self._turn[1] >= self.turn_check_blocks:
            handed_over = self._new_voice()
            if handed_over:
                return handed_over
        self._since_partial += 1
        if self._since_partial >= self.partial_blocks or self._quiet == 1:
            self._since_partial = 0
            return Utterance(np.concatenate(self._speech), final=False, ending=self._quiet == 1)
        return None

    def _new_voice(self) -> Utterance | None:
        """The speech before the last pause, if a voice of the other sex started after it."""
        pause, resumed = self._turn
        self._turn = None
        before = self.pitch(np.concatenate(self._speech[max(0, pause - 16):pause]))
        after = self.pitch(np.concatenate(self._speech[resumed:]))
        if not before or not after or (before < MALE_BELOW_HZ) == (after < MALE_BELOW_HZ):
            return None
        samples = np.concatenate(self._speech[:pause + min(self.pad_blocks, resumed - pause)])
        self._speech = self._speech[max(pause, resumed - self.pad_blocks):]
        self._since_partial = 0
        return Utterance(samples, final=True, cut="turn")

    def flush(self) -> Utterance | None:
        return self._finish("pause") if self._speech else None

    def _finish(self, cut: str) -> Utterance:
        keep = len(self._speech) - max(0, self._quiet - self.pad_blocks)
        samples = np.concatenate(self._speech[:keep])
        self._speech, self._quiet, self._since_partial, self._turn = [], 0, 0, None
        self.vad.reset()
        return Utterance(samples, final=True, cut=cut)


def default_recognizer() -> tuple[SenseVoice, Vad]:
    from .models import SPEECH

    if not SPEECH.downloaded:
        raise FileNotFoundError("Speech recognition is not downloaded yet.")
    vad, tokens, model = (part.path for part in SPEECH.parts)
    recognizer = SenseVoice(model, tokens)
    recognizer.transcribe(np.zeros(RATE, np.float32))
    return recognizer, Vad(vad)
