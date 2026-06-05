"""QLoRA flow-matching SFT — overfit test first, then real SFT.

Precompute-and-free: encode all (caption,image) pairs once with the frozen text
encoder + VAE, free those + the unconditional transformer, then train LoRA on the
conditional transformer alone. Keeps a 9.3B model + backward inside 24GB.

Overfit milestone:
    python -m rtie.training.train_sft --captions-dir data/captions --raw-dir data/raw \
        --height 432 --width 768 --steps 2000 --lr 1e-4 --out checkpoints/overfit_lora

Success = loss -> ~0 and the adapter reproduces the training thumbnails.
CUDA only (nf4 = bitsandbytes 4-bit).
"""

from __future__ import annotations

import argparse
import gc
import random
import time
from pathlib import Path

import torch

from ideogram4 import Ideogram4Pipeline, Ideogram4PipelineConfig
from ideogram4.scheduler import get_schedule_for_resolution

from .dataset import load_pairs, load_image_tensor, read_caption
from .flow_matching import flow_matching_loss_cached, precompute_sample
from .lora import add_qlora, trainable_report

QUANT_REPOS = {"nf4": "ideogram-ai/ideogram-4-nf4", "fp8": "ideogram-ai/ideogram-4-fp8"}


def _free(obj_owner, attr: str) -> None:
    if getattr(obj_owner, attr, None) is not None:
        setattr(obj_owner, attr, None)


def _vram() -> str:
    return f"{torch.cuda.memory_allocated()/1e9:.2f}GB alloc / {torch.cuda.max_memory_allocated()/1e9:.2f}GB peak"


def main() -> None:
    ap = argparse.ArgumentParser(description="QLoRA flow-matching SFT for Ideogram 4.")
    ap.add_argument("--captions-dir", default="data/captions")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--height", type=int, default=432)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=1, help="gradient-accumulation micro-steps per optimizer step")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--sched-mean", type=float, default=1.0)
    ap.add_argument("--sched-std", type=float, default=1.0)
    ap.add_argument("--t-sampling", choices=["uniform", "schedule"], default="uniform",
                    help="uniform = train across the full timestep range (recommended); schedule = logit-normal band")
    ap.add_argument("--quant", choices=list(QUANT_REPOS), default="nf4")
    ap.add_argument("--out", default="checkpoints/overfit_lora")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--max-pairs", type=int, default=0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required (nf4 = bitsandbytes 4-bit).")
    for d in (args.height, args.width):
        if d % 16:
            raise SystemExit(f"height/width must be multiples of 16, got {d}")

    print(f"[load] Ideogram 4 ({args.quant}) ...")
    pipe = Ideogram4Pipeline.from_pretrained(
        config=Ideogram4PipelineConfig(weights_repo=QUANT_REPOS[args.quant]),
        device="cuda",
        dtype=torch.bfloat16,
    )

    # Free the unconditional transformer up front — SFT only trains the conditional one.
    _free(pipe, "unconditional_transformer")
    gc.collect(); torch.cuda.empty_cache()

    # Precompute frozen features for every pair (uses text encoder + VAE).
    pairs = load_pairs(args.captions_dir, args.raw_dir)
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    print(f"[data] precomputing {len(pairs)} (image,caption) pairs @ {args.width}x{args.height} ...")
    samples = []
    for img_path, cap_path in pairs:
        px = load_image_tensor(img_path, args.height, args.width).unsqueeze(0).to("cuda")
        samples.append(precompute_sample(pipe, read_caption(cap_path), px, args.height, args.width))
        del px
    print(f"[precompute] done. {_vram()}")

    # Free the frozen encoders — no longer needed.
    _free(pipe, "text_encoder"); _free(pipe, "autoencoder")
    gc.collect(); torch.cuda.empty_cache()

    # Attach LoRA to the conditional transformer.
    pipe.conditional_transformer = add_qlora(
        pipe.conditional_transformer, rank=args.rank, alpha=args.alpha, dropout=args.dropout
    )
    pipe.conditional_transformer.train()
    print("[lora]", trainable_report(pipe.conditional_transformer))
    torch.cuda.reset_peak_memory_stats()
    print(f"[ready] {_vram()}")

    params = [p for p in pipe.conditional_transformer.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    schedule = get_schedule_for_resolution((args.height, args.width), known_mean=args.sched_mean, std=args.sched_std)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    queue: list[int] = []

    def next_idx() -> int:
        if not queue:
            queue.extend(range(len(samples)))
            random.shuffle(queue)
        return queue.pop()

    t0 = time.time()
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        loss_val = 0.0
        uniform = args.t_sampling == "uniform"
        for _ in range(args.batch_size):
            loss = flow_matching_loss_cached(
                pipe.conditional_transformer, samples[next_idx()], schedule, "cuda", uniform=uniform
            )
            (loss / args.batch_size).backward()
            loss_val += loss.item() / args.batch_size
        torch.nn.utils.clip_grad_norm_(params, args.clip)
        opt.step()

        if step % args.log_every == 0 or step == 1:
            print(f"step {step}/{args.steps}  loss {loss_val:.4f}  {(time.time()-t0)/step:.2f}s/step  {_vram()}", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            ckpt = out / f"step{step}"
            pipe.conditional_transformer.save_pretrained(str(ckpt))
            print(f"[save] adapter -> {ckpt}", flush=True)

    print("[done]")


if __name__ == "__main__":
    main()
