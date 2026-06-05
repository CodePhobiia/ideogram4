#!/usr/bin/env bash
# Self-provision the Vast box: clone the fork, install deps, pre-pull nf4 weights.
# Run with HF_TOKEN in the environment (gate must be accepted on the account).
#   HF_TOKEN=hf_xxx nohup bash provision.sh > setup.log 2>&1 &
set -euo pipefail

cd /workspace
if [ ! -d ideogram4 ]; then
  echo "[clone] CodePhobiia/ideogram4"
  git clone https://github.com/CodePhobiia/ideogram4.git
fi
cd ideogram4

echo "[pip] installing ideogram4 + training deps"
# pytorch image's base env is PEP-668 externally-managed; container is ephemeral so override is safe.
python -m pip install --break-system-packages -e .
python -m pip install --break-system-packages -r rtie/training/requirements.txt

echo "[weights] pre-pulling ideogram-ai/ideogram-4-nf4 (~15-20GB)"
python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("ideogram-ai/ideogram-4-nf4")
print("weights cached at", p)
PY

echo PROVISION_DONE
