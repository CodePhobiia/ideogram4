# RTIE Build Plan — Roblox Thumbnail Intelligence Engine

**Status:** R&D only. Not a revenue-generating product. Will not be released or distributed anywhere. Private experimentation.

This plan describes how to build RTIE by **forking the Ideogram 4 inference repo** and adding the training, data, captioning, ranking, and serving layers it does not ship.

> Reality check: the Ideogram 4 repo is **inference-only** (`pipeline_ideogram4.py`, `modeling_ideogram4.py`, `autoencoder.py`, `caption_verifier.py`, `quantized_loading.py`, `safety.py`, `run_inference.py`). It has **no training code, no LoRA, no dataset loaders, no captioner, no ranker.** Forking gives us the model plumbing (~20%); the other ~80% is net-new code we write on top of it.

---

## Gates (decide before writing code)

| # | Gate | Status |
|---|------|--------|
| 1 | **Hardware / CUDA** | Open. Published weights are `nf4` (bitsandbytes 4-bit, **CUDA-only**) or `fp8` (any device). 9.3B DiT + frozen 8B Qwen3-VL text encoder. Training (even QLoRA) won't fit a typical local box → **rent cloud CUDA (Vast.ai).** |
| 2 | **License** | ✅ **Resolved.** Ideogram 4 is non-commercial; the license explicitly permits "personal use for research, experimentation, testing… private entertainment or hobby project." This is private R&D, no release, no revenue → squarely allowed. (Note: training on third-party copyrighted thumbnails is a separate copyright question, but low practical risk for private, non-distributed R&D.) |
| 3 | **Training data we have rights to** | Open. "Top games have high-CTR thumbnails" = style reference, not a labeled CTR dataset, and likely not rights-cleared. Prefer our own assets / generated data; treat scraped competitor thumbnails as R&D-only style reference, never redistributed. |

---

## Phase 0 — Fork and get vanilla inference working
- Fork `ideogram-oss/ideogram4`. Weights are **gated on Hugging Face** (not in repo): accept gate, `hf auth login`.
- License travels with derivatives (§3: keep Notice file, mark modified files, keep non-commercial terms) — moot unless we ever distribute, which we won't.
- Run `run_inference.py` with a JSON caption (`--no-magic-prompt`) at 1920×1088 to prove the environment and get the `Ideogram4Pipeline` generation primitive (reused everywhere downstream).
- nf4 = CUDA only; fp8 = anywhere. This is the only phase that works out of the box.

## Phase 1 — Data + captioning pipeline (the real bottleneck)
- **Ingest / dedupe (pHash) / rights + policy gates** → `manifest.jsonl`.
- **Auto-captioner that emits Ideogram-schema JSON *with bounding boxes*** — a mini-system: VLM (Claude/GPT-4V) for prose/emotion/archetype + object detector + OCR for `bbox` and `text` fields, assembled in the exact key order Ideogram expects.
- **Validate every caption through the repo's own `CaptionVerifier`** (key order, bbox `[ymin,xmin,ymax,xmax]` 0–1000, uppercase `#RRGGBB`).
- **Keep bboxes** — magic-prompt helpers default to `strip_bboxes=True`, which discards exactly the layout control thumbnails need.
- **Precompute & cache frozen features** (biggest practical win): VAE latents (small, always cache) and Qwen3-VL features via `pipe._encode_text(...)`. Trade-off: text features are large (13-layer concat → ~53k dims/token), so caching all of them costs real disk — alternative is keeping the text encoder resident (see compute notes).

## Phase 2 — Build the training spine the repo lacks
- **`flow_matching.py`** — `zt = (1-t)·z0 + t·z1`, `target = z1 - z0`, `MSE(model(zt,t,caption), target)`, sample `t` from the logit-normal schedule (`scheduler.py`). Reuse `pipe._build_inputs` + the `text_z_padding` concat from the sampling loop.
- **`lora.py` — the trap.** Weights are quantized custom layers:
  - **nf4 (recommended for training):** `swap_linears_to_bnb4bit` makes them `bnb.nn.Linear4bit`; PEFT/QLoRA supports these. Attach LoRA **after** swap+load, targeting `layers.*.attention.qkv|o` and `feed_forward.w1|w2|w3` on the **conditional transformer only**. Freeze text encoder, VAE, and (initially) the unconditional transformer.
  - **fp8:** custom `Fp8Linear` stores weight as a buffer → PEFT doesn't see it; needs a hand-written LoRA wrapper. Easier to **train on nf4 (CUDA), keep fp8 for CPU/MPS inference.**
  - bf16 base would be cleaner but isn't published → QLoRA it is.
- **`train_sft.py`** + checkpointing + a `pipe.load_lora(path)` for inference.

## Phase 3 — Prove the signal cheaply
- **Overfit 5–10 images first** (loss → ~0). If that fails, the data/caption/latent plumbing is broken — find out on 10 images, not 2,000.
- Then a few hundred winners at **768×432**, generate an eval sheet, eyeball base vs SFT. Don't scale until SFT visibly beats base on "Roblox-ness."

## Phase 4 — Preference tuning (makes it a CTR engine, not a style filter)
- Only after SFT works. **Diffusion-DPO** on the SFT adapter using *comparable* winner/loser pairs + a **frozen reference** (SFT snapshot or base). Conservative LR, 1–2 epochs. Data is thinnest here, so this likely lags the generator.

## Phase 5 — Surrounding system (separate from the model)
- **Ranker + QA**: separate SigLIP/CLIP + MLP predicting CTR/qPTR + policy/readability/bottom-safe-zone. Reuse `safety.py` Hive integration for moderation.
- **Candidate-slate orchestration**: archetype × seed × text grid → rank → top-N.
- **Feedback loop** (only if we ever test live, which R&D may skip): metrics → datasets.

---

## The three things that will actually break it
1. **Fine-tuning quantized + custom layers.** QLoRA-on-`Linear4bit` works; `Fp8Linear` needs custom code. Budget time here.
2. **Hardware/cost.** Not a local-machine project. Rent CUDA (Vast.ai). Latent caching keeps the bill sane.
3. **Captions + rights, not training, are the bottleneck.** A clean, bbox-accurate, schema-valid dataset is worth more than any training trick.

---

## First milestone
Phase 0 + skeleton of Phase 1/2 in the fork: one JSON-caption generation working, then `rtie/` scaffolded (data, captioning, training, inference, ranking, qa) with flow-matching loss + QLoRA wrapper stubbed against real module names — so the **first test is "can I overfit 10 images."**
