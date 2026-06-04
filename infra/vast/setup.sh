#!/usr/bin/env bash
# One-shot bootstrap for a fresh Vast.ai instance.
#
# Usage (on the instance):
#   export HF_TOKEN=hf_xxx              # scoped token; rotate after the project
#   export FORK_URL=https://github.com/<you>/ideogram4.git
#   bash setup.sh
#
# Idempotent: safe to re-run. Pre-pulls gated weights into the HF cache so the
# first training/inference call doesn't pay GPU-time to download ~15-20GB.
#
# Prereqs: the gate on the model repo must already be accepted by the HF account
# behind HF_TOKEN (huggingface.co/ideogram-ai/ideogram-4-nf4 -> "Agree and access").

set -euo pipefail

# ---- config (override via env) ----------------------------------------------
WORKDIR="${WORKDIR:-/workspace}"
FORK_URL="${FORK_URL:-https://github.com/CodePhobiia/ideogram4.git}"
REPO_DIR="${REPO_DIR:-$WORKDIR/ideogram4}"
QUANT="${QUANT:-nf4}"                                  # nf4 (CUDA) or fp8 (any device)
MODEL_REPO="${MODEL_REPO:-ideogram-ai/ideogram-4-$QUANT}"
export HF_HOME="${HF_HOME:-$WORKDIR/.hf}"              # keep on a persistent volume

: "${HF_TOKEN:?set HF_TOKEN (scoped Hugging Face token)}"

echo "==> workdir=$WORKDIR repo=$REPO_DIR model=$MODEL_REPO quant=$QUANT"
mkdir -p "$WORKDIR" "$HF_HOME"

# ---- clone / update the fork -------------------------------------------------
if [ -d "$REPO_DIR/.git" ]; then
  echo "==> updating existing checkout"
  git -C "$REPO_DIR" pull --ff-only || true
else
  echo "==> cloning fork"
  git clone "$FORK_URL" "$REPO_DIR"
fi

# ---- deps --------------------------------------------------------------------
# Skip if you launched the prebuilt infra/vast Docker image (deps already baked).
# Editable install picks up the repo's pyproject deps; add training extras.
echo "==> installing deps"
python -m pip install --upgrade pip
python -m pip install -e "$REPO_DIR"
python -m pip install "peft>=0.13.0" wandb

# ---- HF auth + gated weight pre-pull ----------------------------------------
# huggingface_hub reads HF_TOKEN from the env; this just warms the cache.
echo "==> pre-pulling weights ($MODEL_REPO) into $HF_HOME"
hf download "$MODEL_REPO" --quiet || \
  python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL_REPO')"

# ---- sanity ------------------------------------------------------------------
echo "==> sanity check"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda?", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

echo "==> done. Next: run_inference.py with --no-magic-prompt at 1920x1088 to verify Phase 0."
