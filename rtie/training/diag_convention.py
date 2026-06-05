"""Empirically determine Ideogram's flow-matching convention from the BASE model.

For a real image latent z0 and noise z1, build zt two ways and check which target
the base conditional transformer's velocity aligns with (cosine). The (zt, target)
pair with cosine near +1 is the convention training must use.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ideogram4 import Ideogram4Pipeline, Ideogram4PipelineConfig

from .dataset import load_image_tensor, load_pairs, read_caption
from .flow_matching import encode_pixels_to_tokens

H, W = 256, 448


def cos(a, b):
    return F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


def main():
    pipe = Ideogram4Pipeline.from_pretrained(
        config=Ideogram4PipelineConfig(weights_repo="ideogram-ai/ideogram-4-nf4"),
        device="cuda", dtype=torch.bfloat16,
    )
    img, cap = load_pairs("data/captions", "data/raw")[0]
    px = load_image_tensor(img, H, W).unsqueeze(0).to("cuda")
    z0 = encode_pixels_to_tokens(pipe, px)
    inputs = pipe._build_inputs([read_caption(cap)], height=H, width=W)
    with torch.no_grad():
        llm = pipe._encode_text(inputs["token_ids"], inputs["text_position_ids"], inputs["indicator"])
    mt = inputs["max_text_tokens"]
    g = torch.Generator(device="cuda").manual_seed(0)
    z1 = torch.randn(z0.shape, generator=g, device="cuda", dtype=torch.float32)

    print(f"{'zt_build':24} {'t':>4} {'cos(v,z0-z1)':>13} {'cos(v,z1-z0)':>13}")
    for t_val in (0.3, 0.6, 0.85):
        t = torch.full((1,), t_val, device="cuda")
        t_ = t[:, None, None]
        for name, zt in (("A:(1-t)z0+t z1", (1 - t_) * z0 + t_ * z1),
                         ("B:t z0+(1-t)z1", t_ * z0 + (1 - t_) * z1)):
            pos_z = torch.cat([torch.zeros(1, mt, z0.shape[-1], device="cuda"), zt], dim=1)
            with torch.no_grad():
                pred = pipe.conditional_transformer(
                    llm_features=llm, x=pos_z, t=t,
                    position_ids=inputs["position_ids"], segment_ids=inputs["segment_ids"],
                    indicator=inputs["indicator"],
                )
            v = pred[:, mt:]
            print(f"{name:24} {t_val:>4} {cos(v, z0 - z1):>13.3f} {cos(v, z1 - z0):>13.3f}")


if __name__ == "__main__":
    main()
