from screen_translator.capture import Frame
from screen_translator.ocr import RapidOcr


def test_ocr_falls_back_to_the_cpu_when_directml_fails(tmp_path):
    class Result:
        boxes = txts = scores = None

    class FlakyGpu(RapidOcr):
        def _engine(self, lang):
            if self._gpu:
                raise RuntimeError("no DirectX 12 device")
            self._engines["multi"] = lambda image: Result()
            return self._engines["multi"]

    ocr = FlakyGpu(tmp_path, gpu=True)
    assert ocr.recognize(Frame(bytes(4 * 4 * 4), 4, 4), "auto") == []
    assert ocr.device == "CPU" and not ocr._gpu


def test_a_subtitle_dash_read_as_its_own_box_goes_back_on_its_line():
    from screen_translator.ocr import OcrLine, _with_dashes

    lines = [OcrLine("-", 0.9, (0.28, 0.12, 0.01, 0.3)), OcrLine("Where were you?", 0.99, (0.30, 0.10, 0.3, 0.3)),
             OcrLine("At home.", 0.99, (0.30, 0.55, 0.2, 0.3))]
    merged = _with_dashes(lines, 1280, 170)
    assert [line.text for line in merged] == ["- Where were you?", "At home."]
