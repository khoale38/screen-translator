import errno
import io
import threading
from collections import namedtuple

import pytest

from screen_translator import models
from screen_translator.models import ModelSpec, NotEnoughSpace, check_space, download

Usage = namedtuple("Usage", "total used free")
GB = 1_000_000_000


@pytest.fixture
def spec(tmp_path, monkeypatch):
    monkeypatch.setenv("SCREEN_TRANSLATOR_HOME", str(tmp_path))
    return ModelSpec("tiny", "Tiny model", "owner/repo", "tiny.gguf", 5 * GB, "0" * 64)


def test_a_download_that_would_fill_the_disk_is_refused(spec, monkeypatch):
    monkeypatch.setattr(models.shutil, "disk_usage", lambda path: Usage(500 * GB, 495 * GB, int(4.7 * GB)))
    with pytest.raises(NotEnoughSpace, match=r"needs 5\.5 GB free and only 4\.7 GB is free"):
        check_space(spec)


def test_a_partial_download_only_needs_room_for_the_rest(spec, monkeypatch):
    with open(spec.path.with_suffix(".gguf.part"), "wb") as part:
        part.truncate(4 * GB)
    monkeypatch.setattr(models.shutil, "disk_usage", lambda path: Usage(500 * GB, 495 * GB, int(1.6 * GB)))
    check_space(spec)
    spec.path.with_suffix(".gguf.part").unlink()
    with pytest.raises(NotEnoughSpace):
        check_space(spec)


def test_running_out_of_space_mid_download_removes_the_partial_file(spec, monkeypatch):
    monkeypatch.setattr(models.shutil, "disk_usage", lambda path: Usage(500 * GB, 400 * GB, 100 * GB))

    class Response(io.BytesIO):
        status = 200

    monkeypatch.setattr(models.urllib.request, "urlopen", lambda request, timeout: Response(b"a" * 4096))
    real_open = open

    class FullDisk:
        def __init__(self, path, mode):
            self.file = real_open(path, mode)

        def write(self, block):
            self.file.write(block[:10])
            raise OSError(errno.ENOSPC, "No space left on device")

        def close(self):
            self.file.close()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.file.close()

    monkeypatch.setattr(models, "open", FullDisk, raising=False)
    with pytest.raises(NotEnoughSpace, match="filled up"):
        download(spec, lambda done, total: None, threading.Event(), chunk=1024)
    assert not spec.path.with_suffix(".gguf.part").exists()
