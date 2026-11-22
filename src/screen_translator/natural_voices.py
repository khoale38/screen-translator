"""Natural-sounding offline voices for dubbing: Supertonic 3, run on ONNX Runtime.

https://github.com/supertone-inc/supertonic (MIT). One model speaks 31 languages in ten voices; the 8-bit copy of the
model comes from sherpa-onnx's packaging, the voices and character table from Supertone's own release.
"""

from __future__ import annotations

import json
import os
import re
import threading
import unicodedata
from pathlib import Path

import numpy as np

from .dub import MAX_PACE, Delivery, Voice

LANGS = {"en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et", "fi", "fr", "hi", "hr", "hu", "id", "it",
         "lt", "lv", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "vi"}
# Malay has no voice of its own; written Malay reads well enough with Indonesian sounds.
BORROWED = {"ms": "id"}
# Measured: F1-F5 speak at 158-198 Hz, M1-M5 at 88-129 Hz. The first of each is the one used to match a speaker.
STYLES = ("F1", "M1", "F2", "M2", "F3", "M3", "F4", "M4", "F5", "M5")
STEPS = 5
GAP_S = 0.2
_SYMBOLS = str.maketrans({"–": "-", "‑": "-", "—": "-", "_": " ", "“": '"', "”": '"', "‘": "'",
                          "’": "'", "´": "'", "`": "'", "[": " ", "]": " ", "|": " ", "/": " ", "#": " ",
                          "→": " ", "←": " ", "♥": "", "☆": "", "♡": "", "©": "", "\\": ""})
_EMOJI = re.compile("[\U0001f300-\U0001faff☀-➿\U0001f1e6-\U0001f1ff]+")
_SENTENCE_END = re.compile(r"(?<=[.!?。！？])\s+")


def model_folder() -> Path:
    from .models import NATURAL_VOICES

    return NATURAL_VOICES.parts[0].path.parent


def language_of(target: str) -> str | None:
    base = target.split("-")[0].lower()
    base = BORROWED.get(base, base)
    return base if base in LANGS else None


def prepare(text: str, lang: str) -> str:
    """The text as the model was trained to read it, wrapped in its language tag."""
    text = _EMOJI.sub("", unicodedata.normalize("NFKD", text)).translate(_SYMBOLS).replace("@", " at ")
    text = re.sub(r" ([,.!?;:'])", r"\1", text)
    text = re.sub(r"\s+", " ", re.sub(r"([\"'])\1+", r"\1", text)).strip()
    if not re.search(r"[.!?;:,'\")\]}…。」』】〉》›»]$", text):
        text += "."
    return f"<{lang}>{text}</{lang}>"


def pieces(text: str, lang: str) -> list[str]:
    """Sentences grouped up to the length the model reads well in one go."""
    limit = 120 if lang in ("ko", "ja") else 300
    chunks: list[str] = []
    for sentence in _SENTENCE_END.split(text.strip()):
        if chunks and len(chunks[-1]) + len(sentence) + 1 <= limit:
            chunks[-1] += " " + sentence
        elif sentence:
            chunks.append(sentence)
    return chunks


def _mask(lengths: np.ndarray) -> np.ndarray:
    return (np.arange(lengths.max()) < lengths[:, None]).astype(np.float32)[:, None, :]


class NaturalVoices:
    """Lists its voices without loading anything; the model loads on the first line it speaks."""

    def __init__(self, folder: Path | None = None) -> None:
        self.folder = folder or model_folder()
        self._lock = threading.Lock()
        self._sessions = None

    @property
    def voices(self) -> list[Voice]:
        return [Voice(f"natural:{lang}:{style}", f"Natural voice {style[1]}", lang,
                      "female" if style.startswith("F") else "male")
                for lang in sorted(LANGS | set(BORROWED)) for style in STYLES]

    def _load(self):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = min(4, os.cpu_count() or 1)
        model = next(self.folder.glob("sherpa-onnx-supertonic-*"))
        session = lambda name: ort.InferenceSession(str(model / f"{name}.int8.onnx"), options,
                                                    providers=["CPUExecutionProvider"])
        config = json.loads((model / "tts.json").read_text(encoding="utf-8"))
        indexer = np.array(json.loads((self.folder / "unicode_indexer.json").read_text(encoding="utf-8")), np.int64)
        styles = {}
        for style in STYLES:
            data = json.loads((self.folder / f"{style}.json").read_text(encoding="utf-8"))
            styles[style] = tuple(np.array(data[key]["data"], np.float32).reshape(data[key]["dims"])
                                  for key in ("style_ttl", "style_dp"))
        return (session("duration_predictor"), session("text_encoder"), session("vector_estimator"),
                session("vocoder"), config, indexer, styles)

    def synthesize(self, text: str, voice: Voice, how: Delivery) -> tuple[np.ndarray, int]:
        with self._lock:
            if self._sessions is None:
                self._sessions = self._load()
            durations, encoder, estimator, vocoder, config, indexer, styles = self._sessions
        _, lang, style = voice.id.split(":")
        lang = BORROWED.get(lang, lang)
        ttl, dp = styles[style]
        rate = config["ae"]["sample_rate"]
        chunk = config["ae"]["base_chunk_size"] * config["ttl"]["chunk_compress_factor"]
        latent_dim = config["ttl"]["latent_dim"] * config["ttl"]["chunk_compress_factor"]
        rng = np.random.default_rng()
        planned = []
        for piece in pieces(text, lang):
            codes = np.array([ord(c) for c in prepare(piece, lang)])
            ids = indexer[np.minimum(codes, len(indexer) - 1)][None, :]
            text_mask = _mask(np.array([ids.shape[1]]))
            duration = durations.run(None, {"text_ids": ids, "style_dp": dp, "text_mask": text_mask})[0]
            planned.append((ids, text_mask, duration))
        pace = how.pace
        if how.fit_s:
            natural = sum(float(d[0]) for _, _, d in planned) / 1.05 + GAP_S * (len(planned) - 1)
            pace = min(MAX_PACE, max(pace, natural / how.fit_s))
        audio = []
        for ids, text_mask, duration in planned:
            duration = duration / (1.05 * pace)
            embedded = encoder.run(None, {"text_ids": ids, "style_ttl": ttl, "text_mask": text_mask})[0]
            samples = (duration * rate).astype(np.int64)
            latent_mask = _mask((samples + chunk - 1) // chunk)
            latent = rng.standard_normal((1, latent_dim, latent_mask.shape[2])).astype(np.float32) * latent_mask
            total = np.array([STEPS], np.float32)
            for step in range(STEPS):
                latent = estimator.run(None, {
                    "noisy_latent": latent, "text_emb": embedded, "style_ttl": ttl, "text_mask": text_mask,
                    "latent_mask": latent_mask, "current_step": np.array([step], np.float32), "total_step": total,
                })[0]
            if audio:
                audio.append(np.zeros(int(GAP_S / pace * rate), np.float32))
            audio.append(vocoder.run(None, {"latent": latent})[0][0, : int(samples[0])])
        return np.clip(np.concatenate(audio) * how.loudness, -1, 1).astype(np.float32), rate
