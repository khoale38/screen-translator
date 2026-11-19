import os
import sys


def _ensure_std_streams() -> None:
    # Windowed builds start without stdout/stderr, but llama.cpp redirects fds 1 and 2 directly.
    if sys.stdout is not None and sys.stderr is not None:
        return
    from screen_translator.config import data_dir

    log = open(data_dir() / "screen-translator.log", "w", encoding="utf-8", buffering=1)
    for fd in (1, 2):
        os.dup2(log.fileno(), fd)
    sys.stdout = sys.stderr = log


def _no_error_dialogs() -> None:
    """Without a GPU driver, loading the Vulkan backend can raise a modal "vulkan-1.dll was not found" box that
    blocks until clicked; fail quietly instead so llama.cpp falls back to the CPU."""
    if sys.platform == "win32":
        import ctypes

        # SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)


def _ggml():
    from llama_cpp import _ggml, llama_cpp

    if sys.platform == "win32":
        return _ggml.libggml
    # Wheels hold libggml.dylib and libggml.1.dylib as two copies with separate backend registries;
    # looking symbols up through libllama finds the copy llama.cpp actually uses.
    return llama_cpp._lib


def _ggml_base():
    """Device names live in ggml-base; Windows' GetProcAddress, unlike dlsym, does not search dependent DLLs."""
    if sys.platform != "win32":
        return _ggml()
    import ctypes

    from llama_cpp import _ggml as bindings

    return ctypes.CDLL(str(bindings.libggml_base_path / "ggml-base.dll"))


def _load_llama_backends() -> None:
    # GGML_BACKEND_DL builds only search next to the executable, but the backend libraries live in llama_cpp/lib.
    import ctypes

    from llama_cpp import _ggml as bindings

    ggml = _ggml()
    ggml.ggml_backend_reg_count.restype = ctypes.c_size_t
    if ggml.ggml_backend_reg_count() == 0:
        ggml.ggml_backend_load_all_from_path.argtypes = [ctypes.c_char_p]
        ggml.ggml_backend_load_all_from_path(str(bindings.libggml_base_path).encode("utf-8"))


def _llama_devices() -> list[tuple[str, str, int]]:
    import ctypes

    ggml, base = _ggml(), _ggml_base()
    ggml.ggml_backend_dev_count.restype = ctypes.c_size_t
    ggml.ggml_backend_dev_get.argtypes = [ctypes.c_size_t]
    ggml.ggml_backend_dev_get.restype = ctypes.c_void_p
    for fn in (base.ggml_backend_dev_name, base.ggml_backend_dev_description):
        fn.argtypes, fn.restype = [ctypes.c_void_p], ctypes.c_char_p
    base.ggml_backend_dev_type.argtypes, base.ggml_backend_dev_type.restype = [ctypes.c_void_p], ctypes.c_int
    devices = []
    for i in range(ggml.ggml_backend_dev_count()):
        dev = ggml.ggml_backend_dev_get(i)
        name, desc = base.ggml_backend_dev_name(dev), base.ggml_backend_dev_description(dev)
        devices.append((name.decode(), desc.decode(errors="replace"), base.ggml_backend_dev_type(dev)))
    return devices


def _self_test() -> int:
    """Checks that the bundled native pieces load; CI runs this against every build."""
    import time

    import llama_cpp
    from PIL import Image, ImageDraw, ImageFont

    from screen_translator.capture import Frame
    from screen_translator.models import MODELS
    from screen_translator.ocr import default_engine
    from screen_translator.text import assemble
    from screen_translator.translator import LlamaTranslator, build_messages

    from PySide6.QtCore import qVersion

    import screen_translator.desktop  # noqa: F401

    print(f"Desktop UI: Qt {qVersion()}", flush=True)
    print("llama.cpp:", llama_cpp.llama_print_system_info().decode(), flush=True)
    devices = _llama_devices()
    for name, desc, kind in devices:
        print(f"Device: {name} ({desc}), type {kind}", flush=True)
    gpus = [f"{name} ({desc})" for name, desc, kind in devices if kind in (1, 2)]
    print("Backend:", gpus[0] if gpus else "CPU", flush=True)

    image = Image.new("RGB", (900, 160), "white")
    ImageDraw.Draw(image).text((30, 40), "HELLO WORLD", fill="black", font=ImageFont.load_default(72))
    engine = default_engine()
    text = assemble(engine.recognize(Frame(image.convert("RGBA").tobytes("raw", "BGRA"), *image.size), "en"))
    print(f"OCR ({engine.name}): {text!r}", flush=True)
    if "HELLO" not in text.upper():
        print("Self-test failed: OCR did not read the test image", flush=True)
        return 1

    spec = next((m for m in MODELS.values() if m.downloaded), None)
    if spec is None:
        print("No model downloaded; skipping translation", flush=True)
    else:
        started = time.perf_counter()
        translator = LlamaTranslator(str(spec.path))
        print(f"Model {spec.key} loaded in {time.perf_counter() - started:.1f} s", flush=True)
        for line in ("こんにちは、世界。", "ユキはまだ帰ってこないの？", "彼女、昨日から連絡がないんだ。"):
            started = time.perf_counter()
            result = translator.translate(build_messages(line, [], "ja", "en"))
            print(f"Translation: {line} -> {result!r} in {time.perf_counter() - started:.2f} s", flush=True)
        translator.close()
    print("Self-test passed", flush=True)
    return 0


def _exit_now(code: int) -> None:
    """Skip interpreter teardown: on Windows a GPU driver's DLL shutdown can hang it, leaving an invisible process."""
    for stream in (sys.stdout, sys.stderr):
        if stream is not None:
            stream.flush()
    os._exit(code)


if __name__ == "__main__":
    _ensure_std_streams()
    _no_error_dialogs()
    _load_llama_backends()
    if sys.argv[1:] == ["--self-test"]:
        try:
            code = _self_test()
        except BaseException:
            import traceback

            traceback.print_exc()
            code = 1
        _exit_now(code)
    from screen_translator.__main__ import main

    try:
        main()
    except SystemExit as stop:
        _exit_now(stop.code if isinstance(stop.code, int) else 1)
    _exit_now(0)
