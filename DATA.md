# RTIE Data Plan

**Context:** private R&D only, no release, no live testing. Decisions locked with the project owner:
- **Train on real, live Roblox thumbnails** (public images) — fine for a private experiment.
- **No live testing on real games** — out of respect for the original image owners. Consequence below.
- **GPT Image 2 distillation** is a **small, supplementary** part of the corpus — not the core.

---

## What dropping live testing means (the bounded claim)

No owned-game A/B → **no access to real CTR/qPTR, ever, in this project.** The experiment is therefore:

> "Can we build a generator that produces authentically Roblox-native, judge-preferred thumbnails, and does preference-tuning measurably shift it toward winning composition grammar?"

Success = **learnability + visual fidelity + measurable preference shift under a proxy judge.** It does **not** claim proven CTR lift — that would require the live test we've ruled out. Keep this line explicit so results aren't over-stated.

---

## The core constraint

The label we actually want — CTR / qPTR — is **private** (owner-only on Roblox). Public data gives us images + metadata + *which games are trending*, never real click rates. So preference is built from **proxies**, not ground truth.

| Public (scrapeable) | Private (owner-only) |
|---|---|
| thumbnail images + order, visits, concurrent players, favorites, like ratio, genre, age rating, created/updated, trending sorts | impressions, CTR, qPTR, personalization A/B results |

Three datasets, very different difficulty:

| Dataset | Teaches | Hard part |
|---|---|---|
| **A — Style/SFT** | what winning Roblox thumbnails look like | captions, not images |
| **B — Preference** | what beats alternatives | true CTR is private → proxy |
| **C — Ranker/eval** | which candidate deserves a slot | derived from B + generated candidates |

---

## Dataset A — real live thumbnails (the anchor)

Ground truth for *authentic Roblox visual grammar* (chunky avatars, plastic shader, exaggerated faces) that base Ideogram won't produce. Acquisition via Roblox public APIs (explore/discovery sorts → `games` metadata → `thumbnails` images; resolve place→universe IDs; verify exact endpoints live, rate-limit politely, cache):

- **Multiple thumbnails per game** — capture the whole carousel, record **order** (position 1 = hero, most "winner" weight).
- **Highest available resolution**, keep/crop to 16:9.
- **Join metadata** per image: universe_id, title, genre, visits, CCU, favorites, like-ratio, created/updated.
- **Momentum** — scrape trending sorts repeatedly over days; a game *surging* in Up-and-Coming is the closest public proxy to "this thumbnail is working now," less brand-confounded than all-time-top.
- **Dedup hard (pHash)** — Roblox is full of clone/template thumbnails.

**Rights posture:** training on these for private, non-distributed R&D is low practical risk; the line never crossed is **redistributing the dataset or the model.**

### Captioner (the real work)
Each image → rich Ideogram-JSON caption with bboxes:
- **Prose / style / archetype / emotion:** vision LLM (Claude vision, GPT-4V, or **Qwen3-VL — already in the stack**).
- **Bboxes:** *fast/R&D* = VLM-estimated (accept noise); *accurate* = object detector (Grounding DINO/YOLO) + OCR (PaddleOCR/EasyOCR) → real bboxes, VLM describes each region.
- **Color palette:** k-means → uppercase `#RRGGBB`.
- **Validate** through the repo's `CaptionVerifier`, with a repair loop.
- **Genre/archetype:** Roblox genre field + VLM classification into the archetype ontology.

Start tiny: **10** images (overfit test) → **~300–500** (SFT-beats-base test). Don't build 20k before the pipeline is proven.

---

## GPT Image 2 distillation — supplementary, kept in its lane

A *small part* of the generator's training, useful for:
- **Controllable gap-filling** — genres/archetypes the scrape under-covers, for balance.
- **Clean exemplars + prior-preservation** data we 100% control.
- **Cheap perfect preference pairs** — matched pairs where one follows every grammar rule, one breaks it (tiny text, clutter, bottom-zone violation): teaches the rules cleanly.

Honest ceiling (why it stays minority):
- **Circular** — teaches Ideogram to imitate GPT Image 2's polished-generic-3D look, not authentic Roblox-ad grammar. Real thumbnails remain the truth.
- **No platform signal** — can't tell us what wins on Roblox.
- **ToS note:** OpenAI restricts using outputs to train competing image models; private, non-released R&D is gray-but-low-risk (same posture as the Ideogram license).

**Discipline:** tag every row `source: live_scrape | gpt_image_2 | base_ideogram`; keep synthetic a clear minority of SFT; **run the ablation** (train with vs. without synthetic) to measure whether it helps or just homogenizes the look.

---

## Dataset B — preference signal (proxy-only)

1. **VLM-as-judge (primary)** — pairwise clickability rubric (readability, emotion, hook clarity, genre fit) over real + generated thumbnails. The DPO workhorse.
2. **Public-momentum weak labels (the only real-world tether)** — within a genre, high-momentum vs stale-same-genre as noisy winner/loser. Confounded but valuable; weight it in.
3. **High-confidence pairs = momentum and VLM agree** on direction; downweight/drop disagreements. Plus the synthetic rule-pairs above.

---

## Eval (proxy-only)
- **Blind human eval** (owner/small group): base vs SFT vs SFT+DPO — "more Roblox-like / more clickable / readable at 320×180."
- **VLM-judge win-rate** at scale.
- **Quantitative:** caption + bbox adherence, OCR text accuracy, subject size, contrast, archetype diversity.
- **Held-out by game/universe** — never the same game in train and eval (measure generalization, not memorized art direction).

---

## Concrete data plan
1. **Synthetic bootstrap (no scraping):** ~20 hand/LLM-written archetype briefs → base Ideogram + GPT Image 2 → caption back → prove the caption→train→generate loop on owned data.
2. **Scrape + caption ~500 live trending thumbnails**, genre-balanced, source-tagged → Dataset A → SFT vs base.
3. **Add GPT Image 2 supplement** (minority, tagged) → re-SFT → ablate.
4. **VLM-judge + momentum preference pairs** → Diffusion-DPO.
5. Eval blind, held-out by game. Claim bounded to generation quality + preference shift.

**Build order:** scraper (`rtie/data/`) + captioner (`rtie/captioning/`) first — everything downstream depends on them.
