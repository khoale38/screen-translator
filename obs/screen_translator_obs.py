"""Runs inside OBS's own Python (Tools → Scripts), so it must stay standard-library only."""

import json
import os
import shutil
import subprocess
import sys
import urllib.request
import webbrowser

import obspython as obs

SOURCE_NAME = "Live Translation"

_settings = {"command": "", "port": 8765, "autostart": True, "lines": 2, "size": 40, "show_source": False}
_process = None
_hotkeys = {}


def _base_url():
    return f"http://127.0.0.1:{int(_settings['port'])}"


def _overlay_url():
    query = f"lines={int(_settings['lines'])}&size={int(_settings['size'])}"
    if _settings["show_source"]:
        query += "&source=1"
    return f"{_base_url()}/overlay?{query}"


def _api(path, body=None, timeout=2):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(_base_url() + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read() or b"{}")


def _server_up():
    try:
        _api("/api/state", timeout=0.5)
        return True
    except OSError:
        return False


def _find_command():
    if _settings["command"]:
        return _settings["command"]
    found = shutil.which("screen-translator")
    if found:
        return found
    for candidate in (
        "/Applications/Screen Translator.app/Contents/MacOS/screen-translator",
        os.path.expanduser("~/.local/bin/screen-translator"),
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Screen Translator\screen-translator.exe"),
        os.path.expandvars(r"%ProgramFiles%\Screen Translator\screen-translator.exe"),
    ):
        if os.path.exists(candidate):
            return candidate
    return ""


def start_app(*_):
    global _process
    if _server_up() or (_process and _process.poll() is None):
        return True
    command = _find_command()
    if not command:
        obs.script_log(obs.LOG_WARNING, "Set 'Screen Translator command' to the screen-translator executable.")
        return True
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    _process = subprocess.Popen(
        [command, "--headless", "--port", str(int(_settings["port"]))],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags,
    )
    obs.script_log(obs.LOG_INFO, f"Started Screen Translator (pid {_process.pid}).")
    return True


def stop_app(*_):
    global _process
    if _process and _process.poll() is None:
        _process.terminate()
        try:
            _process.wait(5)
        except subprocess.TimeoutExpired:
            _process.kill()
    _process = None
    return True


def open_panel(*_):
    webbrowser.open(f"{_base_url()}/panel")
    return True


def toggle_translation(*_):
    try:
        state = _api("/api/state")
        _api("/api/stop" if state.get("running") else "/api/start", {})
    except OSError as exc:
        obs.script_log(obs.LOG_WARNING, f"Screen Translator is not reachable: {exc}")
    return True


def clear_context(*_):
    try:
        _api("/api/context/clear", {})
    except OSError:
        pass
    return True


def _browser_settings():
    data = obs.obs_data_create()
    obs.obs_data_set_string(data, "url", _overlay_url())
    obs.obs_data_set_int(data, "width", 1920)
    obs.obs_data_set_int(data, "height", 1080)
    obs.obs_data_set_bool(data, "shutdown", False)
    obs.obs_data_set_bool(data, "reroute_audio", False)
    return data


def add_overlay(*_):
    data = _browser_settings()
    source = obs.obs_get_source_by_name(SOURCE_NAME)
    if source is None:
        source = obs.obs_source_create("browser_source", SOURCE_NAME, data, None)
    else:
        obs.obs_source_update(source, data)
    obs.obs_data_release(data)

    scene_source = obs.obs_frontend_get_current_scene()
    scene = obs.obs_scene_from_source(scene_source)
    if obs.obs_scene_find_source(scene, SOURCE_NAME) is None:
        obs.obs_scene_add(scene, source)
    obs.obs_source_release(scene_source)
    obs.obs_source_release(source)
    return True


def _refresh_existing_overlay():
    source = obs.obs_get_source_by_name(SOURCE_NAME)
    if source is not None:
        data = _browser_settings()
        obs.obs_source_update(source, data)
        obs.obs_data_release(data)
        obs.obs_source_release(source)


def script_description():
    return (
        "<b>Screen Translator</b><br>Real-time OCR translation of anything on screen, with context, "
        "running fully on this computer.<br>Start the app, open the control panel to pick the capture area, "
        "then add the overlay to your scene."
    )


def script_properties():
    props = obs.obs_properties_create()
    obs.obs_properties_add_path(props, "command", "Screen Translator command", obs.OBS_PATH_FILE, "", None)
    obs.obs_properties_add_int(props, "port", "Port", 1024, 65535, 1)
    obs.obs_properties_add_bool(props, "autostart", "Start with OBS")
    obs.obs_properties_add_int_slider(props, "lines", "Lines on screen", 1, 5, 1)
    obs.obs_properties_add_int_slider(props, "size", "Font size", 16, 96, 2)
    obs.obs_properties_add_bool(props, "show_source", "Show original text above translation")
    obs.obs_properties_add_button(props, "start", "Start app", start_app)
    obs.obs_properties_add_button(props, "stop", "Stop app", stop_app)
    obs.obs_properties_add_button(props, "panel", "Open control panel", open_panel)
    obs.obs_properties_add_button(props, "toggle", "Start / stop translating", toggle_translation)
    obs.obs_properties_add_button(props, "clear", "Forget context", clear_context)
    obs.obs_properties_add_button(props, "overlay", "Add overlay to current scene", add_overlay)
    return props


def script_defaults(settings):
    obs.obs_data_set_default_int(settings, "port", 8765)
    obs.obs_data_set_default_bool(settings, "autostart", True)
    obs.obs_data_set_default_int(settings, "lines", 2)
    obs.obs_data_set_default_int(settings, "size", 40)


def script_update(settings):
    _settings.update(
        command=obs.obs_data_get_string(settings, "command"),
        port=obs.obs_data_get_int(settings, "port"),
        autostart=obs.obs_data_get_bool(settings, "autostart"),
        lines=obs.obs_data_get_int(settings, "lines"),
        size=obs.obs_data_get_int(settings, "size"),
        show_source=obs.obs_data_get_bool(settings, "show_source"),
    )
    _refresh_existing_overlay()


def script_load(settings):
    script_update(settings)
    for name, label, action in (
        ("screen_translator_toggle", "Screen Translator: start/stop translating", toggle_translation),
        ("screen_translator_clear", "Screen Translator: forget context", clear_context),
    ):
        hotkey_id = obs.obs_hotkey_register_frontend(name, label, _hotkey(action))
        saved = obs.obs_data_get_array(settings, name)
        obs.obs_hotkey_load(hotkey_id, saved)
        obs.obs_data_array_release(saved)
        _hotkeys[name] = hotkey_id
    if _settings["autostart"]:
        start_app()


def script_save(settings):
    for name, hotkey_id in _hotkeys.items():
        saved = obs.obs_hotkey_save(hotkey_id)
        obs.obs_data_set_array(settings, name, saved)
        obs.obs_data_array_release(saved)


def _hotkey(action):
    return lambda pressed: pressed and action()


def script_unload():
    stop_app()
