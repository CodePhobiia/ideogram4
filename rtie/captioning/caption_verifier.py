"""Ideogram-4 caption schema validator.

Faithful re-implementation of the verifier shipped in the Ideogram 4 repo
(src/ideogram4/caption_verifier.py), vendored so we can validate captions
locally without cloning the full repo + its heavy CUDA deps. SWAP THIS for the
repo's exact file once the fork is in place, to guarantee train/inference parity.

All `verify*` methods return a list of warning strings (empty == passed).
Mirrors the repo's rules: known keys, style_description exactly-one-of
photo/art_style with strict key order, required compositional_deconstruction
(background before elements), per-element key order by type, bbox
[ymin,xmin,ymax,xmax] ints in 0..1000, uppercase #RRGGBB palettes with size caps.
"""

from __future__ import annotations

import json
from typing import Sequence

_TOP_KEYS = frozenset({"high_level_description", "style_description", "compositional_deconstruction"})
_SD_PHOTO_ORDER = ("aesthetics", "lighting", "photo", "medium", "color_palette")
_SD_NONPHOTO_ORDER = ("aesthetics", "lighting", "medium", "art_style", "color_palette")
_SD_KNOWN = frozenset({"aesthetics", "lighting", "photo", "art_style", "medium", "color_palette"})
_CD_ORDER = ("background", "elements")
_ELEM_OBJ_ORDER = ("type", "bbox", "desc", "color_palette")
_ELEM_TEXT_ORDER = ("type", "bbox", "text", "desc", "color_palette")
_ELEM_KNOWN = frozenset({"type", "bbox", "text", "desc", "color_palette"})
_ELEM_TYPES = frozenset({"obj", "text"})
_BBOX_MIN, _BBOX_MAX = 0, 1000
_SD_PALETTE_MAX, _ELEM_PALETTE_MAX = 16, 5


class CaptionVerifier:
    def verify(self, caption: dict) -> list[str]:
        w: list[str] = []
        if not isinstance(caption, dict):
            return [f"root: expected JSON object, got {type(caption).__name__}"]
        self._unknown(caption, _TOP_KEYS, "root", w)
        if "high_level_description" in caption and not isinstance(
            caption["high_level_description"], str
        ):
            w.append("high_level_description: expected a string")
        if "style_description" in caption:
            self._style(caption["style_description"], w)
        if "compositional_deconstruction" in caption:
            self._comp(caption["compositional_deconstruction"], w)
        else:
            w.append("root: 'compositional_deconstruction' must exist")
        return w

    def verify_raw(self, raw: str) -> list[str]:
        try:
            return self.verify(json.loads(raw))
        except json.JSONDecodeError as e:
            return [f"invalid JSON: {e}"]

    def _style(self, sd, w: list[str]) -> None:
        if not isinstance(sd, dict):
            w.append("style_description: expected a dict")
            return
        self._unknown(sd, _SD_KNOWN, "style_description", w)
        has_photo, has_art = "photo" in sd, "art_style" in sd
        if has_photo and has_art:
            w.append("style_description: contains both 'photo' and 'art_style'; expected exactly one")
            return
        if not has_photo and not has_art:
            w.append("style_description: expected exactly one of 'photo' or 'art_style'")
            return
        order = _SD_PHOTO_ORDER if has_photo else _SD_NONPHOTO_ORDER
        self._order(sd, [k for k in order if k != "color_palette" or "color_palette" in sd],
                    "style_description", w)
        if "color_palette" in sd:
            self._palette(sd["color_palette"], "style_description.color_palette", _SD_PALETTE_MAX, w)

    def _comp(self, cd, w: list[str]) -> None:
        if not isinstance(cd, dict):
            w.append("compositional_deconstruction: expected a dict")
            return
        if "background" not in cd:
            w.append("compositional_deconstruction: 'background' must exist")
            return
        if not isinstance(cd["background"], str):
            w.append("compositional_deconstruction.background: expected a string")
            return
        if "elements" not in cd:
            w.append("compositional_deconstruction: 'elements' must exist")
            return
        self._order(cd, _CD_ORDER, "compositional_deconstruction", w)
        if not isinstance(cd["elements"], list):
            w.append("compositional_deconstruction.elements: expected a list")
            return
        for i, el in enumerate(cd["elements"]):
            self._element(i, el, w)

    def _element(self, i: int, el, w: list[str]) -> None:
        if not isinstance(el, dict):
            w.append(f"elements[{i}]: expected a dict")
            return
        self._unknown(el, _ELEM_KNOWN, f"elements[{i}]", w)
        if "type" not in el:
            w.append(f"elements[{i}]: 'type' must exist")
            return
        if el.get("type") not in _ELEM_TYPES:
            w.append(f"elements[{i}]: 'type' must be one of {set(_ELEM_TYPES)}")
            return
        base = _ELEM_TEXT_ORDER if el["type"] == "text" else _ELEM_OBJ_ORDER
        order = [k for k in base if k in ("type", "desc", "text") or k in el]
        self._order(el, order, f"elements[{i}]", w)
        if "bbox" in el:
            self._bbox(i, el["bbox"], w)
        if "color_palette" in el:
            self._palette(el["color_palette"], f"elements[{i}].color_palette", _ELEM_PALETTE_MAX, w)

    def _bbox(self, i: int, bbox, w: list[str]) -> None:
        if not isinstance(bbox, list) or len(bbox) != 4:
            w.append(f"elements[{i}].bbox: expected [ymin,xmin,ymax,xmax]")
            return
        if not all(isinstance(v, int) for v in bbox):
            w.append(f"elements[{i}].bbox: all values must be int")
            return
        ymin, xmin, ymax, xmax = bbox
        if not all(_BBOX_MIN <= v <= _BBOX_MAX for v in bbox):
            w.append(f"elements[{i}].bbox: values must be in [{_BBOX_MIN},{_BBOX_MAX}], got {bbox}")
        if ymin > ymax:
            w.append(f"elements[{i}].bbox: ymin>{ymax}")
        if xmin > xmax:
            w.append(f"elements[{i}].bbox: xmin>{xmax}")

    def _palette(self, pal, path: str, max_colors: int, w: list[str]) -> None:
        if not isinstance(pal, list):
            w.append(f"{path}: expected a list")
            return
        if len(pal) > max_colors:
            w.append(f"{path}: too many colors ({len(pal)}), max {max_colors}")
            return
        for j, c in enumerate(pal):
            if (
                not isinstance(c, str)
                or len(c) != 7
                or c[0] != "#"
                or not all(ch in "0123456789ABCDEF" for ch in c[1:])
            ):
                w.append(f"{path}[{j}]: '{c}' is not a valid #RRGGBB hex color")

    @staticmethod
    def _order(obj: dict, expected: Sequence[str], path: str, w: list[str]) -> None:
        present = tuple(k for k in obj if k in expected)
        if present != tuple(expected):
            w.append(f"{path}: key order is {present}, expected {tuple(expected)}")

    @staticmethod
    def _unknown(obj: dict, known: frozenset, path: str, w: list[str]) -> None:
        extra = [k for k in obj if k not in known]
        if extra:
            w.append(f"{path}: unknown keys {extra} (not in schema)")
