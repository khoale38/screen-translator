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


def _load_llama_backends() -> None:
    # GGML_BACKEND_DL builds only search next to the executable, but the backend libraries live in llama_cpp/lib.
    import ctypes

    from llama_cpp import _ggml as bindings

    from screen_translator import devices

    ggml = devices.ggml_lib()
    ggml.ggml_backend_reg_count.restype = ctypes.c_size_t
    if ggml.ggml_backend_reg_count() == 0:
        ggml.ggml_backend_load_all_from_path.argtypes = [ctypes.c_char_p]
        ggml.ggml_backend_load_all_from_path(str(bindings.libggml_base_path).encode("utf-8"))


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
    from screen_translator.devices import llama_devices, model_device

    for name, desc, kind in llama_devices():
        print(f"Device: {name} ({desc}), type {kind}", flush=True)
    print("Backend:", model_device(), flush=True)

    image = Image.new("RGB", (900, 160), "white")
    ImageDraw.Draw(image).text((30, 40), "HELLO WORLD", fill="black", font=ImageFont.load_default(72))
    engine = default_engine()
    text = assemble(engine.recognize(Frame(image.convert("RGBA").tobytes("raw", "BGRA"), *image.size), "en"))
    print(f"OCR ({engine.name} on {engine.device}): {text!r}", flush=True)
    if "HELLO" not in text.upper():
        print("Self-test failed: OCR did not read the test image", flush=True)
        return 1

    specs = [m for m in MODELS.values() if m.downloaded]
    if not specs:
        print("No model downloaded; skipping translation", flush=True)
    for spec in specs:
        started = time.perf_counter()
        translator = LlamaTranslator(str(spec.path))
        print(f"Model {spec.key} loaded in {time.perf_counter() - started:.1f} s", flush=True)
        line = "ユキはまだ帰ってこないの？"
        started = time.perf_counter()
        result = translator.translate(build_messages(line, [], "ja", "en"))
        print(f"Translation: {line} -> {result!r} in {time.perf_counter() - started:.2f} s", flush=True)
        from screen_translator.translator import build_screen_messages

        screen = ["設定", "ユキはまだ帰ってこないの？", "ミッション失敗"]
        started = time.perf_counter()
        results = translator.translate_each(
            [build_screen_messages(i, screen, [], "ja", "en") for i in range(len(screen))], [len(t) for t in screen])
        print(f"Side by side: {results!r} in {time.perf_counter() - started:.2f} s", flush=True)
        translator.close()
        if not result or not all(results):
            print(f"Self-test failed: {spec.key} returned no translation", flush=True)
            return 1
    if not _online_self_test() or not _relationship_self_test() or not _speech_self_test():
        return 1
    print("Self-test passed", flush=True)
    return 0


def _online_self_test() -> bool:
    """A made-up key must reach Claude and be turned away, which proves the SDK and HTTPS work in the bundle."""
    from screen_translator.online import ClaudeTranslator, OnlineError, OnlineHiccup

    try:
        ClaudeTranslator("sk-ant-self-test").translate([{"role": "user", "content": "OK"}])
    except OnlineHiccup as exc:
        cause = exc.__cause__.__cause__ if exc.__cause__ else None
        print(f"Online: skipped, {exc} ({cause!r})", flush=True)
        return True
    except OnlineError as exc:
        print(f"Online: Claude answered ({exc})", flush=True)
        return True
    print("Self-test failed: Claude accepted a made-up key", flush=True)
    return False


def _relationship_self_test() -> bool:
    import time

    from screen_translator.address import QUESTION, RELATIONS, parse_facts, vietnamese
    from screen_translator.pipeline import default_judge

    started = time.perf_counter()
    judge = default_judge()
    answer = judge.answer(QUESTION.format(
        notes="GTA V", lines="Franklin: Yo, what's good, homie?\nLamar: Damn, you trippin', fool.",
        a="Franklin", b="Lamar", relations=", ".join(f'"{r}"' for r in RELATIONS)))
    judge.close()
    facts = parse_facts(answer)
    print(f"Relationship: {facts or answer!r} in {time.perf_counter() - started:.2f} s", flush=True)
    if not facts:
        print("Self-test failed: the relationship model gave no facts", flush=True)
        return False
    print(f"Pronouns: Franklin to Lamar {' / '.join(vietnamese(facts, 'Franklin', 'Lamar'))}", flush=True)
    return True


def _speech_self_test() -> bool:
    import time
    import wave

    import numpy as np

    from screen_translator.models import SPEECH

    if not SPEECH.downloaded:
        print("No speech model downloaded; skipping speech", flush=True)
        return True
    from screen_translator.speech import RATE, default_recognizer

    recognizer, _vad = default_recognizer()
    sample = os.environ.get("ST_SPEECH_SAMPLE")
    if sample:
        with wave.open(sample) as clip:
            if clip.getframerate() != RATE or clip.getnchannels() != 1:
                print(f"Self-test failed: {sample} is not 16 kHz mono", flush=True)
                return False
            audio = np.frombuffer(clip.readframes(clip.getnframes()), np.int16).astype(np.float32) / 32768
        started = time.perf_counter()
        text, spoken, _emotion = recognizer.recognize(audio, "auto")
        print(f"Speech ({spoken}): {text!r} in {time.perf_counter() - started:.2f} s", flush=True)
        if "police" not in text.lower():
            print("Self-test failed: the speech sample was not recognized", flush=True)
            return False
    try:
        from screen_translator.audio import AudioSource

        source = AudioSource()
        print(f"Listening to: {source.name}", flush=True)
        source.close()
    except Exception as exc:
        print(f"No sound device to listen to here: {exc}", flush=True)
    return _dub_self_test()


def _dub_self_test() -> bool:
    import time

    from screen_translator.dub import Delivery, default_voices

    voices = default_voices()
    english = [v for v in voices.voices if v.lang.lower().startswith("en")]
    print(f"Dub voices: {len(voices.voices)}; English: {', '.join(v.label for v in english[:12]) or 'none'}",
          flush=True)
    natural = next((v for v in english if v.id.startswith("natural:")), None)
    system = next((v for v in english if not v.id.startswith("natural:")), None)
    if not (natural or system):
        print("No English voice installed here; skipping dubbing", flush=True)
        return True
    for voice in (natural, system):
        if voice is None:
            continue
        started = time.perf_counter()
        audio, rate = voices.synthesize("Get in the car.", voice, Delivery(1.1, 6, 1.2))
        print(f"Dub: {voice.name} spoke {len(audio) / rate:.1f} s of audio in {time.perf_counter() - started:.2f} s",
              flush=True)
        if len(audio) < rate * 0.3:
            print(f"Self-test failed: {voice.name} produced no audio", flush=True)
            return False
    return True


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
