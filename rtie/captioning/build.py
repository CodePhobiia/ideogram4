"""Assemble + validate Ideogram-4 captions.

Helpers emit keys in the exact schema order the CaptionVerifier expects
(dicts preserve insertion order), so a built caption validates by construction.
The vision describer fills the prose; geometry (text bboxes via OCR, palette)
is merged in.
"""

from __future__ import annotations

import json
from pathlib import Path

from .caption_verifier import CaptionVerifier

_V = CaptionVerifier()


def style_block(aesthetics, lighting, medium, art_style, color_palette=None) -> dict:
    """Non-photo style_description (art_style branch) in canonical key order."""
    sd = {"aesthetics": aesthetics, "lighting": lighting, "medium": medium, "art_style": art_style}
    if color_palette:
        sd["color_palette"] = color_palette
    return sd


def obj(desc, bbox=None, color_palette=None) -> dict:
    e: dict = {"type": "obj"}
    if bbox is not None:
        e["bbox"] = bbox
    e["desc"] = desc
    if color_palette:
        e["color_palette"] = color_palette
    return e


def text(text_str, desc, bbox=None, color_palette=None) -> dict:
    e: dict = {"type": "text"}
    if bbox is not None:
        e["bbox"] = bbox
    e["text"] = text_str
    e["desc"] = desc
    if color_palette:
        e["color_palette"] = color_palette
    return e


def build(high_level: str, style: dict, background: str, elements: list[dict]) -> dict:
    return {
        "high_level_description": high_level,
        "style_description": style,
        "compositional_deconstruction": {"background": background, "elements": elements},
    }


def validate(caption: dict) -> list[str]:
    return _V.verify(caption)


def to_json(caption: dict) -> str:
    """Minified, ensure_ascii=False — exactly how Ideogram expects captions serialized."""
    return json.dumps(caption, ensure_ascii=False, separators=(",", ":"))


def save(caption: dict, path) -> str:
    warnings = validate(caption)
    if warnings:
        raise ValueError("caption failed verification:\n" + "\n".join(warnings))
    Path(path).write_text(to_json(caption), encoding="utf-8")
    return str(path)
