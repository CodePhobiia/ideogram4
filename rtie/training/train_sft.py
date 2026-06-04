"""QLoRA flow-matching SFT — overfit test first, then real SFT.

Overfit milestone (validates the whole training plumbing on Vast):
    python -m rtie.training.train_sft \
        --captions-dir data/captions --raw-dir data/raw \
        --height 432 --width 768 --steps 2000 --lr 1e-4 \
        --out checkpoints/overfit_lora

Success = loss drives toward ~0 and a held generation reproduces the training
thumbnails. Then point it at the full caption set for real SFT.

CUDA only (nf4 = bitsandbytes 4-bit). Frozen: text encoder, VAE, unconditional
transformer. Trained: LoRA adapters on the conditional transformer.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch

from ideogram4 import Ideogram4Pipeline, Ideogram4PipelineConfig
from ideogram4.scheduler import get_schedule_for_resolution

from .dataset import load_pairs, make_batch
from .flow_matching import flow_matching_loss
from .lora import add_qlora, trainable_report

QUANT_REPOS = {"nf4": "ideogram-ai/ideogram-4-nf4", "fp8": "ideogram-ai/ideogram-4-fp8"}


def _freeze(module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)


def main() -> None:
    ap = argparse.ArgumentParser(description="QLoRA flow-matching SFT for Ideogram 4.")
    ap.add_argument("--captions-dir", default="data/captions")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--height", type=int, default=432, help="multiple of 16")
    ap.add_argument("--width", type=int, default=768, help="multiple of 16")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--sched-mean", type=float, default=1.0, help="logit-normal mean (pre resolution shift)")
    ap.add_argument("--sched-std", type=float, default=1.0)
    ap.add_argument("--quant", choices=list(QUANT_REPOS), default="nf4")
    ap.add_argument("--out", default="checkpoints/overfit_lora")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--max-pairs", type=int, default=0, help="cap dataset size (0 = all)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required (nf4 = bitsandbytes 4-bit). Run on a GPU box (Vast).")
    for d in (args.height, args.width):
        if d % 16:
            raise SystemExit(f"height/width must be multiples of 16, got {d}")

    print(f"[load] Ideogram 4 ({args.quant}) ...")
    pipe = Ideogram4Pipeline.from_pretrained(
        config=Ideogram4PipelineConfig(weights_repo=QUANT_REPOS[args.quant]),
        device="cuda",
        dtype=torch.bfloat16,
    )

    # Freeze everything except the conditional transformer's LoRA adapters.
    _freeze(pipe.text_encoder)
    _freeze(pipe.autoencoder)
    _freeze(pipe.unconditional_transformer)
    pipe.conditional_transformer = add_qlora(
        pipe.conditional_transformer,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
    )
    pipe.conditional_transformer.train()
    print("[lora]", trainable_report(pipe.conditional_transformer))

    params = [p for p in pipe.conditional_transformer.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    schedule = get_schedule_for_resolution(
        (args.height, args.width), known_mean=args.sched_mean, std=args.sched_std
    )

    pairs = load_pairs(args.captions_dir, args.raw_dir)
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    print(f"[data] {len(pairs)} (image,caption) pairs @ {args.width}x{args.height}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    queue: list[int] = []

    def next_idxs(bs: int) -> list[int]:
        picked: list[int] = []
        while len(picked) < bs:
            if not queue:
                queue.extend(range(len(pairs)))
                random.shuffle(queue)
            picked.append(queue.pop())
        return picked

    t0 = time.time()
    for step in range(1, args.steps + 1):
        captions, images = make_batch(pairs, next_idxs(args.batch_size), args.height, args.width)
        images = images.to("cuda")
        loss = flow_matching_loss(
            pipe, pipe.conditional_transformer, captions, images, args.height, args.width, schedule
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, args.clip)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if step % args.log_every == 0:
            print(f"step {step}/{args.steps}  loss {loss.item():.4f}  {(time.time()-t0)/step:.2f}s/step")
        if step % args.save_every == 0 or step == args.steps:
            ckpt = out / f"step{step}"
            pipe.conditional_transformer.save_pretrained(str(ckpt))
            print(f"[save] adapter -> {ckpt}")

    print("[done]")


if __name__ == "__main__":
    main()
