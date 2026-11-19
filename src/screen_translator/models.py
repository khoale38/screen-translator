from __future__ import annotations

import errno
import hashlib
import shutil
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

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/main/{self.filename}"

    @property
    def path(self) -> Path:
        return models_dir() / self.filename

    @property
    def downloaded(self) -> bool:
        return self.path.exists() and self.path.stat().st_size == self.size_bytes


MODELS: dict[str, ModelSpec] = {
    spec.key: spec
    for spec in (
        ModelSpec(
            "qwen3.5-4b", "Qwen3.5 4B · recommended (2.7 GB)", "unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf",
            2740937888, "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4",
        ),
        ModelSpec(
            "qwen3.5-2b", "Qwen3.5 2B · fastest, for older machines (2.0 GB)", "unsloth/Qwen3.5-2B-GGUF",
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

ProgressFn = Callable[[int, int], None]


class DownloadCancelled(Exception):
    pass


class NotEnoughSpace(OSError):
    pass


SPACE_MARGIN = 512 * 1024 * 1024


def check_space(spec: ModelSpec) -> None:
    """Refuse a download that would fill the disk; a full disk breaks other apps, not just this one."""
    part = spec.path.with_suffix(spec.path.suffix + ".part")
    remaining = spec.size_bytes - (part.stat().st_size if part.exists() else 0)
    free = shutil.disk_usage(spec.path.parent).free
    if free < remaining + SPACE_MARGIN:
        short = remaining + SPACE_MARGIN - free
        raise NotEnoughSpace(
            f"Not enough disk space for {spec.label.split(' · ')[0]}: it needs {(remaining + SPACE_MARGIN) / 1e9:.1f} GB free and "
            f"only {free / 1e9:.1f} GB is free. Free up {short / 1e9:.1f} GB and try again."
        )


def download(spec: ModelSpec, progress: ProgressFn, cancel: threading.Event, chunk: int = 1 << 20) -> Path:
    check_space(spec)
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


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()
