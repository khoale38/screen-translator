from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Protocol

from .capture import Frame


@dataclass
class OcrLine:
    text: str
    confidence: float
    box: tuple[float, float, float, float]


class OcrEngine(Protocol):
    name: str

    def recognize(self, frame: Frame, lang: str) -> list[OcrLine]: ...


class AppleVisionOcr:
    name = "Apple Vision"

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
                candidates = observation.topCandidates_(1)
                if not candidates:
                    continue
                best = candidates[0]
                bb = observation.boundingBox()
                x, y, w, h = bb.origin.x, bb.origin.y, bb.size.width, bb.size.height
                lines.append(OcrLine(str(best.string()), float(best.confidence()), (x, 1.0 - y - h, w, h)))
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

    def __init__(self) -> None:
        from rapidocr_onnxruntime import RapidOCR

        self._engine = RapidOCR()

    def recognize(self, frame: Frame, lang: str) -> list[OcrLine]:
        bgr = frame.array()[:, :, :3]
        result, _ = self._engine(bgr)
        lines = []
        for box, text, score in result or []:
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            x0, y0 = min(xs) / frame.width, min(ys) / frame.height
            lines.append(OcrLine(text, float(score), (x0, y0, max(xs) / frame.width - x0, max(ys) / frame.height - y0)))
        return lines


def default_engine() -> OcrEngine:
    if sys.platform == "darwin":
        return AppleVisionOcr()
    return RapidOcr()
