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


def _self_test() -> int:
    """Checks that the bundled native pieces load; CI runs this against every build."""
    import llama_cpp
    from PIL import Image, ImageDraw, ImageFont

    from screen_translator.capture import Frame
    from screen_translator.models import MODELS
    from screen_translator.ocr import default_engine
    from screen_translator.text import assemble
    from screen_translator.translator import LlamaTranslator, build_messages

    print("llama.cpp:", llama_cpp.llama_print_system_info().decode(), flush=True)

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
        translator = LlamaTranslator(str(spec.path))
        result = translator.translate(build_messages("こんにちは、世界。", [], "ja", "en"))
        translator.close()
        print(f"Translation ({spec.key}): {result!r}", flush=True)
    print("Self-test passed", flush=True)
    return 0


if __name__ == "__main__":
    _ensure_std_streams()
    if sys.argv[1:] == ["--self-test"]:
        sys.exit(_self_test())
    from screen_translator.__main__ import main

    main()
