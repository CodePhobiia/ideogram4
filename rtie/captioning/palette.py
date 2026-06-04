"""Dominant-color palette extraction -> uppercase #RRGGBB (Ideogram schema).

Uses PIL median-cut quantization (no sklearn/numpy clustering needed) and
returns colors ordered by frequency, which is what `color_palette` wants.
"""

from __future__ import annotations

from PIL import Image


def dominant_colors(path, k: int = 5) -> list[str]:
    """Top-`k` dominant colors of an image as uppercase #RRGGBB, most-frequent first."""
    with Image.open(path) as im:
        im = im.convert("RGB").resize((128, 128))  # downscale for speed
        q = im.quantize(colors=k, method=Image.Quantize.MEDIANCUT)
    palette = q.getpalette() or []
    # getcolors() -> list of (count, palette_index); sort by frequency desc
    counts = sorted((q.getcolors() or []), reverse=True)
    out: list[str] = []
    for _, idx in counts[:k]:
        r, g, b = palette[idx * 3 : idx * 3 + 3]
        out.append(f"#{r:02X}{g:02X}{b:02X}")
    return out
