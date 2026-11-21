"""Records everything the PC plays except this app (Windows 10 2004+ process loopback), so dubbing is not heard back.

https://learn.microsoft.com/windows/win32/api/audioclientactivationparams/ns-audioclientactivationparams-audioclient_activation_params
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
import time
from ctypes import POINTER, Structure, byref, c_int64, c_uint16, c_uint32, c_uint64, c_ubyte, c_ulong, c_void_p
from ctypes.wintypes import DWORD, LPCWSTR, WORD

# comtypes joins COM in the thread that imports it; audio activation needs the multithreaded apartment.
sys.coinit_flags = 0

import comtypes
from comtypes import COMMETHOD, COMObject, GUID, HRESULT, IUnknown

import numpy as np

VIRTUAL_DEVICE = "VAD\\Process_Loopback"
ACTIVATION_TYPE_PROCESS_LOOPBACK = 1
EXCLUDE_TARGET_PROCESS_TREE = 1
VT_BLOB = 65
SHARED = 0
LOOPBACK = 0x00020000
AUTOCONVERT = 0x80000000 | 0x08000000
SILENT = 0x2


class WAVEFORMATEX(Structure):
    _fields_ = [("wFormatTag", WORD), ("nChannels", WORD), ("nSamplesPerSec", DWORD), ("nAvgBytesPerSec", DWORD),
                ("nBlockAlign", WORD), ("wBitsPerSample", WORD), ("cbSize", WORD)]


class LOOPBACK_PARAMS(Structure):
    _fields_ = [("ActivationType", c_uint32), ("TargetProcessId", DWORD), ("ProcessLoopbackMode", c_uint32)]


class BLOB(Structure):
    _fields_ = [("cbSize", c_ulong), ("pBlobData", c_void_p)]


class PROPVARIANT(Structure):
    _fields_ = [("vt", c_uint16), ("r1", c_uint16), ("r2", c_uint16), ("r3", c_uint16), ("blob", BLOB),
                ("pad", c_uint64)]


class IAudioClient(IUnknown):
    _iid_ = GUID("{1CB9AD4C-DBFA-4c32-B178-C2F568A703B2}")
    _methods_ = [
        COMMETHOD([], HRESULT, "Initialize", (["in"], DWORD, "ShareMode"), (["in"], DWORD, "StreamFlags"),
                  (["in"], c_int64, "hnsBufferDuration"), (["in"], c_int64, "hnsPeriodicity"),
                  (["in"], POINTER(WAVEFORMATEX), "pFormat"), (["in"], POINTER(GUID), "AudioSessionGuid")),
        COMMETHOD([], HRESULT, "GetBufferSize", (["out"], POINTER(c_uint32), "pNumBufferFrames")),
        COMMETHOD([], HRESULT, "GetStreamLatency", (["out"], POINTER(c_int64), "phnsLatency")),
        COMMETHOD([], HRESULT, "GetCurrentPadding", (["out"], POINTER(c_uint32), "pNumPaddingFrames")),
        COMMETHOD([], HRESULT, "IsFormatSupported", (["in"], DWORD, "ShareMode"),
                  (["in"], POINTER(WAVEFORMATEX), "pFormat"), (["out"], POINTER(POINTER(WAVEFORMATEX)), "ppClosest")),
        COMMETHOD([], HRESULT, "GetMixFormat", (["out"], POINTER(POINTER(WAVEFORMATEX)), "ppDeviceFormat")),
        COMMETHOD([], HRESULT, "GetDevicePeriod", (["out"], POINTER(c_int64), "phnsDefault"),
                  (["out"], POINTER(c_int64), "phnsMinimum")),
        COMMETHOD([], HRESULT, "Start"),
        COMMETHOD([], HRESULT, "Stop"),
        COMMETHOD([], HRESULT, "Reset"),
        COMMETHOD([], HRESULT, "SetEventHandle", (["in"], c_void_p, "eventHandle")),
        COMMETHOD([], HRESULT, "GetService", (["in"], POINTER(GUID), "riid"), (["out"], POINTER(c_void_p), "ppv")),
    ]


class IAudioCaptureClient(IUnknown):
    _iid_ = GUID("{C8ADBD64-E71E-48a0-A4DE-185C395CD317}")
    _methods_ = [
        COMMETHOD([], HRESULT, "GetBuffer", (["out"], POINTER(POINTER(c_ubyte)), "ppData"),
                  (["out"], POINTER(c_uint32), "pNumFramesToRead"), (["out"], POINTER(DWORD), "pdwFlags"),
                  (["out"], POINTER(c_uint64), "pu64DevicePosition"), (["out"], POINTER(c_uint64), "pu64QPCPosition")),
        COMMETHOD([], HRESULT, "ReleaseBuffer", (["in"], c_uint32, "NumFramesRead")),
        COMMETHOD([], HRESULT, "GetNextPacketSize", (["out"], POINTER(c_uint32), "pNumFramesInNextPacket")),
    ]


class IActivateAudioInterfaceAsyncOperation(IUnknown):
    _iid_ = GUID("{72A22D78-CDE4-431D-B8CC-843A71199B6D}")
    _methods_ = [
        COMMETHOD([], HRESULT, "GetActivateResult", (["out"], POINTER(HRESULT), "activateResult"),
                  (["out"], POINTER(POINTER(IUnknown)), "activatedInterface")),
    ]


class IActivateAudioInterfaceCompletionHandler(IUnknown):
    _iid_ = GUID("{41D949AB-9862-444A-80F6-C261334DA5EB}")
    _methods_ = [
        COMMETHOD([], HRESULT, "ActivateCompleted",
                  (["in"], POINTER(IActivateAudioInterfaceAsyncOperation), "activateOperation")),
    ]


class IAgileObject(IUnknown):
    _iid_ = GUID("{94ea2b94-e9cc-49e0-c0ff-ee64ca8f5b90}")
    _methods_ = []


class _Activated(COMObject):
    _com_interfaces_ = [IActivateAudioInterfaceCompletionHandler, IAgileObject]

    def __init__(self) -> None:
        super().__init__()
        self.done = threading.Event()
        self.result: int = -1
        self.interface = None

    def ActivateCompleted(self, operation) -> int:
        try:
            self.result, self.interface = operation.GetActivateResult()
        finally:
            self.done.set()
        return 0


def _activate() -> IAudioClient:
    params = LOOPBACK_PARAMS(ACTIVATION_TYPE_PROCESS_LOOPBACK, os.getpid(), EXCLUDE_TARGET_PROCESS_TREE)
    variant = PROPVARIANT(vt=VT_BLOB, blob=BLOB(ctypes.sizeof(params), ctypes.cast(byref(params), c_void_p)))
    activate = ctypes.WinDLL("Mmdevapi.dll").ActivateAudioInterfaceAsync
    activate.argtypes = [LPCWSTR, POINTER(GUID), POINTER(PROPVARIANT),
                         POINTER(IActivateAudioInterfaceCompletionHandler),
                         POINTER(POINTER(IActivateAudioInterfaceAsyncOperation))]
    activate.restype = HRESULT
    handler = _Activated()
    operation = POINTER(IActivateAudioInterfaceAsyncOperation)()
    activate(VIRTUAL_DEVICE, byref(IAudioClient._iid_), byref(variant),
             handler.QueryInterface(IActivateAudioInterfaceCompletionHandler), byref(operation))
    if not handler.done.wait(5):
        raise TimeoutError("Windows did not answer the request to record app sound")
    if handler.result != 0 or handler.interface is None:
        raise OSError(f"Recording app sound was refused (0x{handler.result & 0xFFFFFFFF:08X})")
    return handler.interface.QueryInterface(IAudioClient)


def _format(rate: int, channels: int) -> WAVEFORMATEX:
    align = channels * 2
    return WAVEFORMATEX(1, channels, rate, rate * align, align, 16, 0)


class ProcessLoopback:
    """16 kHz mono blocks of all PC sound except this app. Silence is filled in when nothing is playing."""

    excludes_own_sound = True

    def __init__(self, rate: int = 16000) -> None:
        try:
            comtypes.CoInitializeEx(comtypes.COINIT_MULTITHREADED)
        except OSError:
            pass
        self.rate = rate
        self._client = _activate()
        for source_rate, channels, flags in ((rate, 1, LOOPBACK | AUTOCONVERT), (44100, 2, LOOPBACK)):
            fmt = _format(source_rate, channels)
            try:
                self._client.Initialize(SHARED, flags, 2_000_000, 0, byref(fmt), None)
            except comtypes.COMError:
                self._client = _activate()
                continue
            self.source_rate, self.channels = source_rate, channels
            break
        else:
            raise OSError("No recording format was accepted for app sound")
        pointer = self._client.GetService(byref(IAudioCaptureClient._iid_))
        self._capture = ctypes.cast(pointer, POINTER(IAudioCaptureClient))
        self._client.Start()
        self._pending = np.zeros(0, np.float32)
        self._last_data = time.monotonic()

    def read(self, frames: int) -> np.ndarray:
        while len(self._pending) < frames:
            got = self._drain()
            if not got:
                idle = time.monotonic() - self._last_data
                if idle > 0.1:
                    self._pending = np.concatenate([self._pending, np.zeros(int(idle * self.rate), np.float32)])
                    self._last_data = time.monotonic()
                else:
                    time.sleep(0.01)
        block, self._pending = self._pending[:frames], self._pending[frames:]
        return block

    def _drain(self) -> bool:
        got = False
        while self._capture.GetNextPacketSize():
            data, count, flags, _device, _qpc = self._capture.GetBuffer()
            raw = ctypes.string_at(data, count * self.channels * 2)
            self._capture.ReleaseBuffer(count)
            audio = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
            if flags & SILENT:
                audio[:] = 0
            audio = audio.reshape(-1, self.channels).mean(axis=1)
            if self.source_rate != self.rate:
                audio = np.convolve(audio, np.ones(3, np.float32) / 3, "same")
                positions = np.arange(0, len(audio), self.source_rate / self.rate)
                audio = np.interp(positions, np.arange(len(audio)), audio).astype(np.float32)
            self._pending = np.concatenate([self._pending, audio])
            self._last_data = time.monotonic()
            got = True
        return got

    def close(self) -> None:
        try:
            self._client.Stop()
        except comtypes.COMError:
            pass
