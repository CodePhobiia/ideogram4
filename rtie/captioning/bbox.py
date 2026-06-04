"""Geometry stage: extract bounding boxes for the Ideogram caption.

TEXT boxes come from OCR (RapidOCR, ONNX/CPU) — exact strings + pixel-accurate
polygons, far better than a model guessing coordinates. SUBJECT (obj) boxes need
an open-vocab detector (YOLO-World / Grounding DINO); that's GPU work best run on
Vast at scale (see note in detect_subjects).

All boxes are returned in the Ideogram schema's normalized space:
`[ymin, xmin, ymax, xmax]`, ints in 0..1000, origin top-left.
"""

from __future__ import annotations

from PIL import Image

_engine = None


def _ocr():
    global _engine
    if _engine is None:
        from rapidocr_onnxruntime import RapidOCR

        _engine = RapidOCR()
    return _engine


def _norm(xs, ys, W, H) -> list[int]:
    bbox = [min(ys) / H * 1000, min(xs) / W * 1000, max(ys) / H * 1000, max(xs) / W * 1000]
    return [max(0, min(1000, int(round(v)))) for v in bbox]


def text_boxes(path) -> list[dict]:
    """OCR text regions as [{text, bbox:[ymin,xmin,ymax,xmax], score}], reading order."""
    with Image.open(path) as im:
        W, H = im.size
    result, _ = _ocr()(str(path))
    out: list[dict] = []
    for poly, text, score in result or []:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        try:
            score_f = float(score)
        except (TypeError, ValueError):
            score_f = None
        out.append({"text": str(text), "bbox": _norm(xs, ys, W, H), "score": score_f})
    return out


def detect_subjects(path) -> list[dict]:
    """Subject (obj) boxes via an open-vocab detector.

    Not implemented on this CPU box: open-vocab detection (YOLO-World /
    Grounding DINO) is heavy and belongs on the Vast GPU alongside training.
    For the small local validation batch, subject boxes are supplied by the
    vision describer instead. Returns [] here so the pipeline still runs.
    """
    return []
