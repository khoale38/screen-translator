from __future__ import annotations

import errno
import hashlib
import shutil
import tarfile
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import models_dir


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    repo: str
    filename: str
    size_bytes: int
    sha256: str
    source: str = ""
    folder: str = ""
    unpack: bool = False

    @property
    def url(self) -> str:
        return self.source or f"https://huggingface.co/{self.repo}/resolve/main/{self.filename}"

    @property
    def path(self) -> Path:
        return models_dir() / self.folder / self.filename

    @property
    def parts(self) -> tuple["ModelSpec", ...]:
        return (self,)

    @property
    def unpacked_marker(self) -> Path:
        return self.path.with_name(self.filename + ".unpacked")

    @property
    def downloaded(self) -> bool:
        if self.unpack:
            return self.unpacked_marker.exists()
        return self.path.exists() and self.path.stat().st_size == self.size_bytes


MODELS: dict[str, ModelSpec] = {
    spec.key: spec
    for spec in (
        ModelSpec(
            "qwen3.5-4b", "Qwen3.5 4B · recommended (2.7 GB)", "unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf",
            2740937888, "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4",
        ),
        ModelSpec(
            "hy-mt2-1.8b", "Hy-MT2 1.8B · fastest, made for translation (1.5 GB)", "tencent/Hy-MT2-1.8B-GGUF",
            "Hy-MT2-1.8B-Q6_K.gguf", 1474785120,
            "d98fe604dec1f28f58f80d7d560f7177e584d3b8e5835862687660e5ff97cb40",
        ),
        ModelSpec(
            "qwen3.5-2b", "Qwen3.5 2B · for older machines (2.0 GB)", "unsloth/Qwen3.5-2B-GGUF",
            "Qwen3.5-2B-Q8_0.gguf", 2012012800,
            "1b04acba824817554f4ce23639bc8495ff70453b8fcb047900c731521021f2c1",
        ),
        ModelSpec(
            "qwen3.5-9b", "Qwen3.5 9B · best quality, needs 16 GB RAM (5.7 GB)", "unsloth/Qwen3.5-9B-GGUF",
            "Qwen3.5-9B-Q4_K_M.gguf", 5680522464,
            "03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8",
        ),
        ModelSpec(
            "gemma3-4b", "Gemma 3 4B · widest language coverage (2.5 GB)", "unsloth/gemma-3-4b-it-GGUF",
            "gemma-3-4b-it-Q4_K_M.gguf", 2489894016,
            "04a43a22e8d2003deda5acc262f68ec1005fa76c735a9962a8c77042a74a7d19",
        ),
    )
}

DEFAULT_MODEL = "qwen3.5-4b"


@dataclass(frozen=True)
class ModelBundle:
    """Several files that are downloaded and used together."""

    key: str
    label: str
    parts: tuple[ModelSpec, ...]

    @property
    def size_bytes(self) -> int:
        return sum(part.size_bytes for part in self.parts)

    @property
    def downloaded(self) -> bool:
        return all(part.downloaded for part in self.parts)


_SENSEVOICE = "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
SPEECH = ModelBundle("speech", "Speech recognition (SenseVoice-Small)", (
    ModelSpec("speech-vad", "Silero VAD", "", "silero_vad.onnx", 643854,
              "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6",
              source="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx",
              folder="speech"),
    ModelSpec("speech-tokens", "SenseVoice tokens", _SENSEVOICE, "tokens.txt", 315894,
              "f449eb28dc567533d7fa59be34e2abca8784f771850c78a47fb731a31429a1dc", folder="speech"),
    ModelSpec("speech-model", "SenseVoice-Small", _SENSEVOICE, "model.int8.onnx", 239233841,
              "c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51", folder="speech"),
))
_SUPERTONIC = "https://huggingface.co/Supertone/supertonic-3/resolve/3cadd1ee6394adea1bd021217a0e650ede09a323"
NATURAL_VOICES = ModelBundle("voices", "Natural voices (Supertonic 3)", (
    ModelSpec("voices-model", "Supertonic 3, 8-bit", "", "sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2",
              128774318, "82fa96f91c4ef8abaae3a14a3f4153facf88bed821d1f7331cec2700f432c427",
              source="https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/"
                     "sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2",
              folder="voices", unpack=True),
    ModelSpec("voices-characters", "Supertonic character table", "", "unicode_indexer.json", 277676,
              "9bf7346e43883a81f8645c81224f786d43c5b57f3641f6e7671a7d6c493cb24f",
              source=f"{_SUPERTONIC}/onnx/unicode_indexer.json", folder="voices"),
    ModelSpec("voice-F1", "Voice F1", "", "F1.json", 292046, "bbdec6ee00231c2c742ad05483df5334cab3b52fda3ba38e6a07059c4563dbc2",
              source=f"{_SUPERTONIC}/voice_styles/F1.json", folder="voices"),
    ModelSpec("voice-F2", "Voice F2", "", "F2.json", 292423, "7c722c6a72707b1a77f035d67f0d1351ba187738e06f7683e8c72b1df3477fc6",
              source=f"{_SUPERTONIC}/voice_styles/F2.json", folder="voices"),
    ModelSpec("voice-F3", "Voice F3", "", "F3.json", 290794, "12f6ef2573baa2defa1128069cb59f203e3ab67c92af77b42df8a0e3a2f7c6ab",
              source=f"{_SUPERTONIC}/voice_styles/F3.json", folder="voices"),
    ModelSpec("voice-F4", "Voice F4", "", "F4.json", 291808, "c2fa764c1225a76dfc3e2c73e8aa4f70d9ee48793860eb34c295fff01c2e032b",
              source=f"{_SUPERTONIC}/voice_styles/F4.json", folder="voices"),
    ModelSpec("voice-F5", "Voice F5", "", "F5.json", 291479, "45966e73316415626cf41a7d1c6f3b4c70dbc1ba2bee5c1978ef0ce33244fc8d",
              source=f"{_SUPERTONIC}/voice_styles/F5.json", folder="voices"),
    ModelSpec("voice-M1", "Voice M1", "", "M1.json", 291748, "e35604687f5d23694b8e91593a93eec0e4eca6c0b02bb8ed69139ab2ea6b0a5b",
              source=f"{_SUPERTONIC}/voice_styles/M1.json", folder="voices"),
    ModelSpec("voice-M2", "Voice M2", "", "M2.json", 292055, "b76cbf62bac707c710cf0ae5aba5e31eea1a6339a9734bfae33ab98499534a50",
              source=f"{_SUPERTONIC}/voice_styles/M2.json", folder="voices"),
    ModelSpec("voice-M3", "Voice M3", "", "M3.json", 290198, "ea1ac35ccb91b0d7ecad533a2fbd0eec10c91513d8951e3b25fbba99954e159b",
              source=f"{_SUPERTONIC}/voice_styles/M3.json", folder="voices"),
    ModelSpec("voice-M4", "Voice M4", "", "M4.json", 291522, "ca8eefad4fcd989c9379032ff3e50738adc547eeb5e221b82593a6d7b3bac303",
              source=f"{_SUPERTONIC}/voice_styles/M4.json", folder="voices"),
    ModelSpec("voice-M5", "Voice M5", "", "M5.json", 291469, "dd22b92740314321f8ae11c5e87f8dd60d060f15dd3a632b5adf77f471f77af2",
              source=f"{_SUPERTONIC}/voice_styles/M5.json", folder="voices"),
))
DOWNLOADS: dict[str, "ModelSpec | ModelBundle"] = {**MODELS, SPEECH.key: SPEECH, NATURAL_VOICES.key: NATURAL_VOICES}

ProgressFn = Callable[[int, int], None]


class DownloadCancelled(Exception):
    pass


class NotEnoughSpace(OSError):
    pass


SPACE_MARGIN = 512 * 1024 * 1024


def _remaining(spec: ModelSpec) -> int:
    if spec.downloaded:
        return 0
    part = spec.path.with_suffix(spec.path.suffix + ".part")
    return spec.size_bytes - (part.stat().st_size if part.exists() else 0)


def check_space(spec: "ModelSpec | ModelBundle") -> None:
    """Refuse a download that would fill the disk; a full disk breaks other apps, not just this one."""
    remaining = sum(_remaining(part) for part in spec.parts)
    free = shutil.disk_usage(models_dir()).free
    if free < remaining + SPACE_MARGIN:
        short = remaining + SPACE_MARGIN - free
        raise NotEnoughSpace(
            f"Not enough disk space for {spec.label.split(' · ')[0]}: it needs {(remaining + SPACE_MARGIN) / 1e9:.1f} GB free and "
            f"only {free / 1e9:.1f} GB is free. Free up {short / 1e9:.1f} GB and try again."
        )


def download_all(item: "ModelSpec | ModelBundle", progress: ProgressFn, cancel: threading.Event) -> None:
    check_space(item)
    before = 0
    for spec in item.parts:
        if not spec.downloaded:
            download(spec, lambda done, _total, before=before: progress(before + done, item.size_bytes), cancel)
            if spec.unpack:
                unpack(spec)
        before += spec.size_bytes


def download(spec: ModelSpec, progress: ProgressFn, cancel: threading.Event, chunk: int = 1 << 20) -> Path:
    check_space(spec)
    spec.path.parent.mkdir(parents=True, exist_ok=True)
    part = spec.path.with_suffix(spec.path.suffix + ".part")
    done = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(spec.url, headers={"User-Agent": "screen-translator"})
    if done:
        request.add_header("Range", f"bytes={done}-")

    with urllib.request.urlopen(request, timeout=30) as response:
        if done and response.status != 206:
            done = 0
        last_report = 0.0
        with open(part, "ab" if done else "wb") as out:
            while block := response.read(chunk):
                if cancel.is_set():
                    raise DownloadCancelled()
                try:
                    out.write(block)
                except OSError as exc:
                    if exc.errno != errno.ENOSPC:
                        raise
                    out.close()
                    part.unlink(missing_ok=True)
                    raise NotEnoughSpace(f"The disk filled up while downloading {spec.label.split(' · ')[0]}; the partial file "
                                         f"was removed. Free up space and try again.") from exc
                done += len(block)
                now = time.monotonic()
                if now - last_report > 0.5:
                    progress(done, spec.size_bytes)
                    last_report = now

    progress(done, spec.size_bytes)
    if sha256_of(part) != spec.sha256:
        part.unlink()
        raise ValueError(f"Checksum mismatch for {spec.filename}; the partial file was removed, please retry.")
    part.rename(spec.path)
    return spec.path


def unpack(spec: ModelSpec) -> None:
    """Unpack a downloaded archive beside it and remove the archive, keeping a marker that it is done."""
    with tarfile.open(spec.path) as archive:
        archive.extractall(spec.path.parent, filter="data")
    spec.path.unlink()
    spec.unpacked_marker.write_text("", encoding="utf-8")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()
