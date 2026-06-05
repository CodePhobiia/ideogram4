"""Generate a thumbnail from an Ideogram JSON caption, optionally with a LoRA adapter.

Base render (baseline):
    python -m rtie.training.generate --caption data/_baseline/baseline_chase.json --out base.png
Adapter render (overfit verify / SFT eval):
    python -m rtie.training.generate --caption <file> --adapter checkpoints/overfit_lora/step2000 --out after.png

Same caption + seed + resolution for base and adapter -> the only variable is the weights.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ideogram4 import Ideogram4Pipeline, Ideogram4PipelineConfig, PRESETS

QUANT = {"nf4": "ideogram-ai/ideogram-4-nf4", "fp8": "ideogram-ai/ideogram-4-fp8"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate from an Ideogram JSON caption (+ optional LoRA).")
    ap.add_argument("--caption", required=True, help="path to a caption .json (verbatim prompt, no magic prompt)")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (omit for base model)")
    ap.add_argument("--out", default="out.png")
    ap.add_argument("--height", type=int, default=576)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preset", default="V4_QUALITY_48", choices=list(PRESETS))
    ap.add_argument("--quant", default="nf4", choices=list(QUANT))
    ap.add_argument("--gw", type=float, default=None, help="constant guidance weight (override preset CFG schedule)")
    a = ap.parse_args()

    pipe = Ideogram4Pipeline.from_pretrained(
        config=Ideogram4PipelineConfig(weights_repo=QUANT[a.quant]),
        device="cuda",
        dtype=torch.bfloat16,
    )
    if a.adapter:
        from peft import PeftModel

        pipe.conditional_transformer = PeftModel.from_pretrained(pipe.conditional_transformer, a.adapter)
        pipe.conditional_transformer.train(False)  # inference mode (avoids LoRA dropout)
        print(f"[adapter] loaded {a.adapter}")

    caption = Path(a.caption).read_text(encoding="utf-8")
    p = PRESETS[a.preset]
    gs = tuple([a.gw] * p.num_steps) if a.gw is not None else p.guidance_schedule
    img = pipe(
        caption,
        height=a.height,
        width=a.width,
        num_steps=p.num_steps,
        guidance_schedule=gs,
        mu=p.mu,
        std=p.std,
        seed=a.seed,
        raise_on_caption_issues=False,
    )[0]
    img.save(a.out)
    print(f"[saved] {a.out} {img.size}")


if __name__ == "__main__":
    main()
