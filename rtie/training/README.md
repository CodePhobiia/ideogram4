# RTIE Training Spine (QLoRA flow-matching)

CUDA only — the 9.3B model can't load on the local CPU box. Runs on Vast.

## Files
- `flow_matching.py` — VAE-encode → `zt=(1-t)z0+t·z1`, target `z1-z0`, MSE on image-token velocities; reuses the pipeline's `_build_inputs`/`_encode_text`.
- `lora.py` — QLoRA adapters on the conditional transformer's `bnb.nn.Linear4bit` attn+MLP layers (`qkv,o,w1,w2,w3`).
- `dataset.py` — pairs `data/captions/<stem>.json` with `data/raw/<stem>.png`.
- `train_sft.py` — the loop (overfit test first, then real SFT).

## Run on Vast
```bash
# 1. bring up the box (clones this fork incl. rtie/, installs deps, pre-pulls weights)
export HF_TOKEN=hf_...                 # gate accepted on ideogram-ai/ideogram-4-nf4
bash infra/vast/setup.sh
python -m pip install -r rtie/training/requirements.txt

# 2. upload the overfit data (NOT in git — copyrighted): 10 imgs + captions
#    -> data/raw/*.png  and  data/captions/*.json   (scp/rsync from local)

# 3. overfit test
python -m rtie.training.train_sft \
    --captions-dir data/captions --raw-dir data/raw \
    --height 432 --width 768 --steps 2000 --lr 1e-4 \
    --out checkpoints/overfit_lora
```
**Success = loss → ~0** and the adapter can reproduce the training thumbnails.
If OOM on a 24GB card, drop resolution (e.g. `--height 256 --width 448`) first.

## Verify (generate from the adapter)
```python
import torch
from peft import PeftModel
from ideogram4 import Ideogram4Pipeline, Ideogram4PipelineConfig, PRESETS

pipe = Ideogram4Pipeline.from_pretrained(
    config=Ideogram4PipelineConfig(weights_repo="ideogram-ai/ideogram-4-nf4"),
    device="cuda", dtype=torch.bfloat16)
pipe.conditional_transformer = PeftModel.from_pretrained(
    pipe.conditional_transformer, "checkpoints/overfit_lora/step2000")

caption = open("data/captions/383310974_t0.json", encoding="utf-8").read()  # a training caption
p = PRESETS["V4_QUALITY_48"]
img = pipe(caption, height=432, width=768, num_steps=p.num_steps,
           guidance_schedule=p.guidance_schedule, mu=p.mu, std=p.std,
           seed=0, raise_on_caption_issues=False)[0]
img.save("overfit_check.png")
```
Compare `overfit_check.png` to the original — close ⇒ the spine learns; we then scale to the full caption set for real SFT and move on to preference (DPO).

## Notes / knobs to tune on-GPU
- `--sched-mean/--sched-std`: the logit-normal t-sampling distribution. If loss stalls, this is the first thing to sweep.
- v1 targets attn+MLP only; add `input_proj/llm_cond_proj/final_layer.linear` in `lora.py` if it underfits.
- No gradient checkpointing yet (the custom transformer has no hook) — add manual `torch.utils.checkpoint` per block if memory is tight at higher res.
