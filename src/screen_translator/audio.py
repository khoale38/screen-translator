"""What the PC is playing, as 16 kHz mono blocks for speech recognition."""

from __future__ import annotations

import queue
import sys
import threading

import numpy as np

from .speech import CHUNK, RATE


class _SoundcardRecorder:
    def __init__(self, recorder) -> None:
        self._recorder = recorder
        recorder.__enter__()

    def read(self, frames: int) -> np.ndarray:
        return np.asarray(self._recorder.record(numframes=frames), np.float32).reshape(-1)

    def close(self) -> None:
        self._recorder.__exit__(None, None, None)


class AudioSource:
    """On Windows, everything the PC plays except this app (so dubbing is not heard back), or, before Windows 10
    2004, a loopback of the default speakers; elsewhere the default microphone.

    A thread of its own drains the device, so a slow recognizer delays the text but never drops sound.
    """

    def __init__(self) -> None:
        self._blocks: queue.Queue[np.ndarray] = queue.Queue(maxsize=int(30 * RATE / CHUNK))
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self.name = ""
        self.excludes_own_sound = False
        self._thread = threading.Thread(target=self._run, name="audio", daemon=True)
        self._thread.start()
        self._ready.wait(10)
        if self._error:
            raise self._error

    def _open(self):
        if sys.platform == "win32":
            try:
                from .winloopback import ProcessLoopback

                recorder = ProcessLoopback(RATE)
                self.name, self.excludes_own_sound = "all apps except this one", True
                return recorder
            except Exception as exc:
                print(f"Recording all apps except this one is unavailable ({exc}); using the speaker loopback",
                      flush=True)
        # Imported here: soundcard sets up COM for the thread that imports it.
        import soundcard

        if sys.platform == "win32":
            device = soundcard.get_microphone(str(soundcard.default_speaker().name), include_loopback=True)
        else:
            device = soundcard.default_microphone()
        self.name = device.name
        # A larger WASAPI buffer rides out pauses in this process; Core Audio caps the block at 512 frames.
        return _SoundcardRecorder(
            device.recorder(samplerate=RATE, channels=1, blocksize=RATE // 4 if sys.platform == "win32" else CHUNK))

    def _run(self) -> None:
        try:
            recorder = self._open()
        except Exception as exc:
            self._error = RuntimeError(f"Could not listen to the PC's sound: {exc}")
            self._ready.set()
            return
        self._ready.set()
        try:
            while not self._stop.is_set():
                block = recorder.read(CHUNK)
                try:
                    self._blocks.put_nowait(block)
                except queue.Full:
                    self._blocks.get_nowait()
                    self._blocks.put_nowait(block)
        finally:
            recorder.close()

    def read(self, timeout: float = 0.5) -> np.ndarray | None:
        try:
            return self._blocks.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
