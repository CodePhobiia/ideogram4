"""Tiny (caption, image) dataset for the overfit / SFT loop.

Pairs each saved Ideogram caption (data/captions/<stem>.json) with its image
(data/raw/<stem>.png). Images are resized to the training resolution and mapped
to [-1, 1]; captions are read as the verbatim minified JSON string the model expects.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_pairs(captions_dir: str, raw_dir: str) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for cap in sorted(Path(captions_dir).glob("*.json")):
        img = Path(raw_dir) / f"{cap.stem}.png"
        if img.exists():
            pairs.append((img, cap))
    if not pairs:
        raise FileNotFoundError(f"no (image, caption) pairs under {raw_dir} / {captions_dir}")
    return pairs


def load_image_tensor(path: Path, height: int, width: int) -> torch.Tensor:
    """[3,H,W] float32 in [-1,1] (PIL resize is W,H)."""
    with Image.open(path) as im:
        im = im.convert("RGB").resize((width, height))
        arr = (np.asarray(im, dtype=np.float32) / 127.5) - 1.0  # [H,W,3]
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def read_caption(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def make_batch(pairs, idxs, height: int, width: int) -> tuple[list[str], torch.Tensor]:
    captions = [read_caption(pairs[i][1]) for i in idxs]
    images = torch.stack([load_image_tensor(pairs[i][0], height, width) for i in idxs])
    return captions, images
