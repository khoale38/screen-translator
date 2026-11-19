import ctypes
import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from screen_translator import hotkeys  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def shortcuts(app):
    keys = hotkeys.GlobalHotkeys()
    pressed = []
    keys.triggered.connect(pressed.append)
    yield keys, pressed
    keys.clear()


def press_mac(ident):
    """Deliver the event macOS sends for a pressed hotkey, without needing a real keyboard."""
    carbon = ctypes.CDLL("/System/Library/Frameworks/Carbon.framework/Carbon")
    carbon.GetApplicationEventTarget.restype = ctypes.c_void_p
    carbon.CreateEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_double,
                                   ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)]
    carbon.SetEventParameter.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_ulong,
                                         ctypes.c_void_p]
    carbon.SendEventToEventTarget.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    carbon.ReleaseEvent.argtypes = [ctypes.c_void_p]
    event = ctypes.c_void_p()
    assert carbon.CreateEvent(None, hotkeys._fourcc("keyb"), 5, 0, 0, ctypes.byref(event)) == 0
    key = hotkeys._EventHotKeyID(hotkeys._MacBackend.SIGNATURE, ident)
    carbon.SetEventParameter(event, hotkeys._fourcc("----"), hotkeys._fourcc("hkid"), ctypes.sizeof(key),
                             ctypes.byref(key))
    carbon.SendEventToEventTarget(event, carbon.GetApplicationEventTarget())
    carbon.ReleaseEvent(event)


def press_windows(app, ident):
    """Post the WM_HOTKEY message Windows sends for a pressed hotkey, then let Qt dispatch it."""
    user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
    assert user32.PostThreadMessageW(kernel32.GetCurrentThreadId(), 0x0312, ident, 0)
    app.processEvents()


@pytest.mark.skipif(sys.platform not in ("darwin", "win32"), reason="global hotkeys exist on macOS and Windows")
def test_every_shortcut_is_claimed_and_reports_its_action(app, shortcuts):
    keys, pressed = shortcuts
    assert keys.register() == []
    names = list(hotkeys.ACTIONS)
    for ident in (1, 2, 4):
        press_mac(ident) if sys.platform == "darwin" else press_windows(app, ident)
    assert pressed == [names[0], names[1], names[3]] == ["toggle", "hide", "screen2"]


def test_labels_match_the_platform():
    assert hotkeys.label("T") == ("⌃⇧T" if sys.platform == "darwin" else "Ctrl+Shift+T")
