"""System-wide shortcuts (Control+Shift+key) that work while a game or full-screen video has focus.

macOS uses Carbon's RegisterEventHotKey and Windows uses RegisterHotKey; neither needs accessibility permission.
"""

from __future__ import annotations

import ctypes
import sys

from PySide6.QtCore import QAbstractNativeEventFilter, QCoreApplication, QObject, Signal

ACTIONS = {"toggle": "T", "hide": "H", **{f"screen{n}": str(n) for n in range(1, 10)}}

_MAC_KEYCODES = {"T": 0x11, "H": 0x04, "1": 0x12, "2": 0x13, "3": 0x14, "4": 0x15, "5": 0x17, "6": 0x16,
                 "7": 0x1A, "8": 0x1C, "9": 0x19}


def label(key: str) -> str:
    return f"⌃⇧{key}" if sys.platform == "darwin" else f"Ctrl+Shift+{key}"


def _fourcc(code: str) -> int:
    return int.from_bytes(code.encode(), "big")


class _EventHotKeyID(ctypes.Structure):
    _fields_ = [("signature", ctypes.c_uint32), ("id", ctypes.c_uint32)]


class _EventTypeSpec(ctypes.Structure):
    _fields_ = [("eventClass", ctypes.c_uint32), ("eventKind", ctypes.c_uint32)]


class _MacBackend:
    SIGNATURE = _fourcc("StTr")
    KEYBOARD_EVENTS = _fourcc("keyb")
    HOTKEY_PRESSED = 5
    MODIFIERS = 0x1000 | 0x0200  # controlKey | shiftKey

    def __init__(self, on_press) -> None:
        carbon = ctypes.CDLL("/System/Library/Frameworks/Carbon.framework/Carbon")
        carbon.GetApplicationEventTarget.restype = ctypes.c_void_p
        carbon.InstallEventHandler.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong,
                                               ctypes.POINTER(_EventTypeSpec), ctypes.c_void_p,
                                               ctypes.POINTER(ctypes.c_void_p)]
        carbon.RegisterEventHotKey.argtypes = [ctypes.c_uint32, ctypes.c_uint32, _EventHotKeyID, ctypes.c_void_p,
                                               ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)]
        carbon.UnregisterEventHotKey.argtypes = [ctypes.c_void_p]
        carbon.GetEventParameter.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
                                             ctypes.c_ulong, ctypes.c_void_p, ctypes.c_void_p]
        self.carbon = carbon
        self.on_press = on_press
        self.target = carbon.GetApplicationEventTarget()
        self.refs: list[ctypes.c_void_p] = []
        handler_type = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
        self._handler = handler_type(self._dispatch)
        spec = _EventTypeSpec(self.KEYBOARD_EVENTS, self.HOTKEY_PRESSED)
        self._handler_ref = ctypes.c_void_p()
        carbon.InstallEventHandler(self.target, ctypes.cast(self._handler, ctypes.c_void_p), 1, ctypes.byref(spec),
                                   None, ctypes.byref(self._handler_ref))

    def _dispatch(self, _call, event, _data) -> int:
        hotkey = _EventHotKeyID()
        status = self.carbon.GetEventParameter(event, _fourcc("----"), _fourcc("hkid"), None,
                                               ctypes.sizeof(hotkey), None, ctypes.byref(hotkey))
        if status == 0 and hotkey.signature == self.SIGNATURE:
            self.on_press(hotkey.id)
            return 0
        return -9874  # eventNotHandledErr, so other handlers still see it

    def add(self, ident: int, key: str) -> bool:
        ref = ctypes.c_void_p()
        status = self.carbon.RegisterEventHotKey(_MAC_KEYCODES[key], self.MODIFIERS,
                                                 _EventHotKeyID(self.SIGNATURE, ident), self.target, 0,
                                                 ctypes.byref(ref))
        if status == 0:
            self.refs.append(ref)
        return status == 0

    def clear(self) -> None:
        for ref in self.refs:
            self.carbon.UnregisterEventHotKey(ref)
        self.refs = []


class _WindowsBackend(QAbstractNativeEventFilter):
    WM_HOTKEY = 0x0312
    MODIFIERS = 0x0002 | 0x0004 | 0x4000  # MOD_CONTROL | MOD_SHIFT | MOD_NOREPEAT

    def __init__(self, on_press) -> None:
        super().__init__()
        from ctypes import wintypes

        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.user32.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT]
        self.user32.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
        self.msg_type = wintypes.MSG
        self.on_press = on_press
        self.ids: list[int] = []
        QCoreApplication.instance().installNativeEventFilter(self)

    def add(self, ident: int, key: str) -> bool:
        # No window handle: WM_HOTKEY then goes to this thread's queue, which Qt's dispatcher shows to the filter.
        if self.user32.RegisterHotKey(None, ident, self.MODIFIERS, ord(key)):
            self.ids.append(ident)
            return True
        return False

    def nativeEventFilter(self, event_type, message):
        if bytes(event_type) in (b"windows_dispatcher_MSG", b"windows_generic_MSG"):
            msg = self.msg_type.from_address(int(message))
            if msg.message == self.WM_HOTKEY and msg.wParam in self.ids:
                self.on_press(int(msg.wParam))
                return True, 0
        return False, 0

    def clear(self) -> None:
        for ident in self.ids:
            self.user32.UnregisterHotKey(None, ident)
        self.ids = []


class GlobalHotkeys(QObject):
    """Emits `triggered` with an action name from ACTIONS when its shortcut is pressed anywhere."""

    triggered = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._names = list(ACTIONS)
        on_press = lambda ident: self.triggered.emit(self._names[ident - 1])
        if sys.platform == "darwin":
            self._backend = _MacBackend(on_press)
        elif sys.platform == "win32":
            self._backend = _WindowsBackend(on_press)
        else:
            self._backend = None

    def register(self) -> list[str]:
        """Claim every shortcut; returns the labels another app already holds (or all of them if unsupported)."""
        if self._backend is None:
            return [label(key) for key in ACTIONS.values()]
        self._backend.clear()
        return [label(key) for ident, key in enumerate(ACTIONS.values(), 1) if not self._backend.add(ident, key)]

    def clear(self) -> None:
        if self._backend is not None:
            self._backend.clear()
