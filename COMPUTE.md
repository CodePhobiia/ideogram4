# RTIE Compute & Cost Plan (Vast.ai)

**Context:** private R&D only — bursty experiments, no production serving. This is why a GPU marketplace (rent, run, tear down) beats a reserved cloud instance.

> Prices below are **ballpark and float with supply** — always check live on Vast before committing. Figures reflect mid-2026 ranges.

---

## Why Vast.ai
Rent consumer (3090/4090) and datacenter (A100/H100) GPUs from third-party hosts, typically **2–5× cheaper than AWS/GCP**. Trade-offs: it's someone else's machine (matters for the HF token and private images — see Gotchas), and instances are ephemeral (provision persistent storage + checkpoint).

## The fact that makes this cheap
**Everything except the conditional transformer is frozen.** Qwen3-VL-8B text encoder and the VAE never train. Memory is dominated by one ~9.3B transformer in 4-bit + a tiny LoRA + tiny optimizer state.

## GPU per task

| Task | Resident | GPU |
|------|----------|-----|
| Inference (Phase 0) | nf4 DiT (~5GB) + VAE + text encoder, one forward | 3090/4090 24GB comfortable |
| Feature precompute (Phase 1) | Qwen3-VL-8B + VAE, batch → disk | 24GB; one-time |
| QLoRA SFT (Phase 2–3) @ 768×432 | 4-bit DiT + LoRA(<100MB) + AdamW(LoRA only) + activations w/ grad-checkpoint | **24GB 4090 enough**; A100 = headroom |
| Higher-res / DPO (Phase 4) | 2 forwards (winner+loser) + frozen reference | **A100 40–80GB** painless |

**Recommendation:** a single **A100 80GB on-demand** is the path of least resistance — keep text encoder + VAE + DiT all resident, skip caching gymnastics. Tight budget → **4090 24GB** for SFT (with feature caching), but DPO gets cramped.

## The caching trade-off (surprise factor)
Caching frozen features cuts VRAM but text features are **huge**: 13 concatenated Qwen3-VL layers → ~53k dims/token. A ~300-token JSON caption ≈ 32MB fp16 → **~64GB disk for 2,000 images.**
- **VAE latents: always cache** (~330KB/image, <1GB total).
- **Text features: usually don't cache** — keep the text encoder resident (~8–16GB VRAM) or recompute. On an A100 80GB, keep everything resident and forget caching.

## Costs (hedge hard — check live)
- RTX 3090 24GB: ~$0.20–0.40/hr
- RTX 4090 24GB: ~$0.30–0.60/hr
- A100 80GB: ~$0.80–1.50/hr
- H100 80GB: ~$1.80–3.00/hr (overkill — skip)

**Per run:** QLoRA SFT, ~2,000 imgs × 4 epochs ≈ 8,000 micro-steps; ~1–3s/step on A100 ≈ **4–6 GPU-hrs ≈ $5–15/run.** Cost is the *number* of runs (caption iterations, sweeps, debugging) + idle time, not any single run.

**Realistic total if disciplined: a few hundred dollars** ($200–800 over the project), dominated by iteration + storage. Balloons only if instances sit idle — **always destroy when done.**

## Vast.ai gotchas
1. **On-demand vs interruptible.** Interruptible is cheaper but killable. Use on-demand for real training, or checkpoint every ~500 steps so a kill costs minutes.
2. **Weights re-download per fresh instance** (~15–20GB gated). Put the HF cache on a **persistent volume** or bake the Docker image so you don't pay GPU-time to re-pull.
3. **One Docker image.** Heavy deps (`torch>=2.11`, `bitsandbytes>=0.49.2`, GB of CUDA wheels). Pin recent CUDA, install once, snapshot. See `infra/vast/Dockerfile`.
4. **Security / data.** Third-party box: use a **scoped HF token and rotate after**, don't upload sensitive data, pull results down + destroy instance.
5. **Provision disk.** Weights + latent cache + checkpoints add up; running out mid-run is a classic failure.

## Suggested flow
1. **Smoke test** — cheap 4090 on-demand, Phase 0 inference (~1–2 hrs, a few $).
2. **Precompute** — cache VAE latents; decide text-feature strategy.
3. **Train** — A100 80GB on-demand (simplest) or 4090 24GB w/ caching (cheapest); checkpoint to persistent volume every 500 steps.
4. **Eval** — cheap card for generation/eval sheets.

## Infra files
- `infra/vast/Dockerfile` — reproducible training image (CUDA + deps + repo + PEFT).
- `infra/vast/setup.sh` — one-shot bootstrap for a fresh instance (deps, HF auth, weight pre-pull).
