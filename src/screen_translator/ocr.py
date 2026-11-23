from __future__ import annotations

import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .capture import Frame
from .config import data_dir


@dataclass
class OcrLine:
    text: str
    confidence: float
    box: tuple[float, float, float, float]


# The L1/R1 button icon comes out of OCR as "LI", "Lt", "L|" or "L!"; "Lt." stays, as that is lieutenant.
BUTTON_MISREAD = re.compile(r"(?<![A-Za-z])([LR])[Il|t!](?![A-Za-z0-9.])")
_BUTTON = re.compile(r"(?<![A-Za-z])[LR][12Il|t!](?![A-Za-z0-9.])")
_LONE_BUTTON = re.compile(r"(?<![A-Za-z])[LR][12Il|t!]?(?![A-Za-z0-9.])")


def prefer_button_reading(candidates: list[str]) -> str:
    """Vision's first guess often drops the 1 from an L1 icon; take a close runner-up that still has it."""
    top = candidates[0]
    if _BUTTON.search(top):
        return top
    bare = lambda text: " ".join(_LONE_BUTTON.sub(" ", text).split())
    return next((c for c in candidates[1:] if _BUTTON.search(c) and bare(c) == bare(top)), top)


class OcrEngine(Protocol):
    name: str
    hint: str
    device: str

    def recognize(self, frame: Frame, lang: str) -> list[OcrLine]: ...


class AppleVisionOcr:
    name = "Apple Vision"
    hint = ""
    device = "macOS"

    def __init__(self) -> None:
        import Vision

        self._vision = Vision
        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        supported, _ = request.supportedRecognitionLanguagesAndReturnError_(None)
        self.supported = [str(code) for code in (supported or [])]

    def vision_languages(self, lang: str) -> list[str] | None:
        if lang == "auto":
            return None
        wanted = {"zh": "zh-Hans", "zh-Hant": "zh-Hant"}.get(lang, lang)
        matches = [c for c in self.supported if c == wanted or c.split("-")[0] == wanted]
        if not matches:
            return None
        if not wanted.startswith("en"):
            matches.append("en-US")
        return matches

    def recognize(self, frame: Frame, lang: str) -> list[OcrLine]:
        import objc

        Vision = self._vision
        with objc.autorelease_pool():
            request = Vision.VNRecognizeTextRequest.alloc().init()
            request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
            request.setUsesLanguageCorrection_(True)
            languages = self.vision_languages(lang)
            if languages:
                request.setRecognitionLanguages_(languages)
            elif hasattr(request, "setAutomaticallyDetectsLanguage_"):
                request.setAutomaticallyDetectsLanguage_(True)

            handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(_cgimage(frame), None)
            ok, _ = handler.performRequests_error_([request], None)
            if not ok:
                return []

            lines = []
            for observation in request.results() or []:
                candidates = observation.topCandidates_(5)
                if not candidates:
                    continue
                text = prefer_button_reading([str(c.string()) for c in candidates])
                bb = observation.boundingBox()
                x, y, w, h = bb.origin.x, bb.origin.y, bb.size.width, bb.size.height
                lines.append(OcrLine(text, float(candidates[0].confidence()), (x, 1.0 - y - h, w, h)))
            return lines


def _cgimage(frame: Frame):
    import Quartz

    provider = Quartz.CGDataProviderCreateWithData(None, frame.bgra, len(frame.bgra), None)
    return Quartz.CGImageCreate(
        frame.width, frame.height, 8, 32, frame.width * 4,
        Quartz.CGColorSpaceCreateDeviceRGB(),
        Quartz.kCGBitmapByteOrder32Little | Quartz.kCGImageAlphaNoneSkipFirst,
        provider, None, False, Quartz.kCGRenderingIntentDefault,
    )


class RapidOcr:
    name = "RapidOCR"
    hint = "Auto-detect reads Chinese, Japanese and Latin-script text. Pick Korean, Russian, Ukrainian, Thai, Arabic or Hindi explicitly."

    def __init__(self, model_dir: Path | None = None, gpu: bool = sys.platform == "win32") -> None:
        self._model_dir = model_dir or ocr_models_dir()
        self._engines: dict[str, object] = {}
        self._gpu = gpu
        self.device = "CPU"
        self.timing: dict[str, int] = {}

    def _engine(self, lang: str):
        group = RAPID_GROUPS.get(lang, "multi")
        if group not in self._engines:
            from rapidocr import LangDet, LangRec, ModelType, OCRVersion, RapidOCR

            if group == "multi":
                rec = {"Rec.ocr_version": OCRVersion.PPOCRV6, "Rec.model_type": ModelType.SMALL, "Rec.lang_type": LangRec.JAPAN}
            else:
                rec = {"Rec.ocr_version": OCRVersion.PPOCRV5, "Rec.model_type": ModelType.MOBILE, "Rec.lang_type": LangRec(group)}
            self._engines[group] = RapidOCR(params={
                "Global.model_root_dir": str(self._model_dir),
                "Global.log_level": "error",
                "Global.use_cls": False,
                "EngineConfig.onnxruntime.intra_op_num_threads": max(1, (os.cpu_count() or 2) // 2),
                "EngineConfig.onnxruntime.use_dml": self._gpu,
                "Det.ocr_version": OCRVersion.PPOCRV6,
                "Det.model_type": ModelType.SMALL,
                "Det.lang_type": LangDet.CH,
                **rec,
            })
        return self._engines[group]

    def recognize(self, frame: Frame, lang: str) -> list[OcrLine]:
        image = frame.array()[:, :, :3]
        # The detector splits rows and repeats letters on very wide, short crops such as one subtitle line; a margin of
        # the edge colour above and below gives it room.
        pad = min(250, max(0, frame.width // 4 - frame.height) // 2)
        if pad:
            image = np.pad(image, ((pad, pad), (0, 0), (0, 0)), mode="edge")
        try:
            result = self._engine(lang)(image)
        except Exception:
            if not self._gpu:
                raise
            # DirectML needs a DirectX 12 GPU; without one, read on the CPU from now on.
            self._gpu, self._engines = False, {}
            result = self._engine(lang)(image)
        self.device = _ort_device(self._engines[RAPID_GROUPS.get(lang, "multi")])
        steps = list(getattr(result, "elapse_list", None) or [])
        if len(steps) >= 3 and all(isinstance(v, float) for v in (steps[0], steps[2])):
            self.timing = {"find": round(steps[0] * 1000), "read": round(steps[2] * 1000)}
        if result.boxes is None or result.txts is None:
            return []
        lines = []
        for box, text, score in zip(result.boxes, result.txts, result.scores):
            xs, ys = box[:, 0], np.clip(box[:, 1] - pad, 0, frame.height)
            x0, y0 = float(xs.min()) / frame.width, float(ys.min()) / frame.height
            w, h = float(xs.max()) / frame.width - x0, float(ys.max()) / frame.height - y0
            lines.append(OcrLine(text, float(score), (x0, y0, w, h)))
        return _with_dashes(lines, frame.width, frame.height)


def _with_dashes(lines: list[OcrLine], width: int, height: int) -> list[OcrLine]:
    """A two-speaker subtitle's leading dash, read as a box of its own, put back on the line it starts."""
    def same_row(line: OcrLine, mark: OcrLine) -> bool:
        return abs((line.box[1] + line.box[3] / 2) - (mark.box[1] + mark.box[3] / 2)) < line.box[3] / 2

    def starts_row(mark: OcrLine) -> bool:
        return not any(other is not mark and same_row(other, mark) and other.box[0] < mark.box[0] for other in lines)

    def small_mark(line: OcrLine) -> bool:
        # OCR also reads a lone subtitle dash as "0", "口" or "一"; only such a squat glyph opening its row counts.
        text = line.text.strip()
        if text in ("-", "–", "—", "－", "·", "•"):
            return starts_row(line)
        rows = [other.box[3] for other in lines if other is not line and same_row(other, line)
                and other.box[2] * width > other.box[3] * height]
        return (text in ("0", "o", "O", "口", "一", "ー") and bool(rows) and line.box[3] < 0.6 * max(rows)
                and line.box[2] * width < 1.5 * line.box[3] * height and starts_row(line))

    dashes = [line for line in lines if small_mark(line)]
    kept = [line for line in lines if line not in dashes]
    for dash in dashes:
        row = [line for line in kept if same_row(line, dash)
               and -0.5 * line.box[3] * height <= (line.box[0] - dash.box[0] - dash.box[2]) * width
               <= 1.5 * line.box[3] * height]
        if row:
            line = min(row, key=lambda item: item.box[0])
            right = line.box[0] + line.box[2]
            kept[kept.index(line)] = OcrLine(f"- {line.text.lstrip()}", line.confidence,
                                             (dash.box[0], line.box[1], right - dash.box[0], line.box[3]))
    return kept


def _ort_device(engine) -> str:
    session = getattr(getattr(getattr(engine, "text_det", None), "session", None), "session", None)
    providers = session.get_providers() if session is not None else []
    return "GPU (DirectML)" if providers and providers[0] == "DmlExecutionProvider" else "CPU"


RAPID_GROUPS = {"ko": "korean", "ru": "eslav", "uk": "eslav", "th": "th", "ar": "arabic", "hi": "devanagari"}


def ocr_models_dir() -> Path:
    """RapidOCR downloads missing models next to the ones it has, so they must live somewhere writable."""
    target = data_dir() / "ocr-models"
    target.mkdir(parents=True, exist_ok=True)
    bundled = Path(getattr(sys, "_MEIPASS", "")) / "ocr_models"
    if getattr(sys, "frozen", False) and bundled.is_dir():
        for model in bundled.iterdir():
            if not (target / model.name).exists():
                shutil.copy2(model, target / model.name)
    return target


def prefetch_rapid_models(target: Path) -> list[str]:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (480, 80), (0, 0, 0))
    ImageDraw.Draw(image).text((20, 25), "Screen Translator 123", fill=(255, 255, 255), font_size=28)
    raw = image.convert("RGBA").tobytes("raw", "BGRA")
    RapidOcr(target).recognize(Frame(raw, image.width, image.height), "auto")
    return sorted(p.name for p in target.glob("*.onnx"))


def default_engine() -> OcrEngine:
    if sys.platform == "darwin" and os.environ.get("SCREEN_TRANSLATOR_OCR") != "rapidocr":
        return AppleVisionOcr()
    return RapidOcr()
