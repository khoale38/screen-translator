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
