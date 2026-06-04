# RTIE training/inference image for Vast.ai
#
# Build once, push to a registry, then point your Vast instance at it so you
# don't pay GPU-time re-installing heavy CUDA wheels on every spin-up.
#
#   docker build -t <you>/rtie:latest infra/vast
#   docker push <you>/rtie:latest
#
# Version pins reflect the Ideogram 4 repo's pyproject (torch>=2.11,
# bitsandbytes>=0.49.2). Bump the CUDA tag to match the torch build you install;
# nf4 (bitsandbytes 4-bit) requires a CUDA GPU.

FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Put the HF cache on /workspace so a mounted persistent volume keeps weights
    # across instances (override at runtime if your volume mounts elsewhere).
    HF_HOME=/workspace/.hf

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip git git-lfs curl ca-certificates \
    && git lfs install \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Torch first, from the CUDA index, so the rest resolve against it.
# (Adjust cu124 / version to the live torch>=2.11 build for your CUDA tag.)
RUN python -m pip install --upgrade pip \
    && python -m pip install "torch>=2.11" --index-url https://download.pytorch.org/whl/cu124

# Core Ideogram 4 inference deps (mirrors the repo pyproject) + training extras.
RUN python -m pip install \
        "transformers>=4.49.0" \
        "safetensors>=0.4.5" \
        "accelerate>=1.0.0" \
        "einops>=0.7.0" \
        sentencepiece \
        pillow \
        "huggingface_hub>=0.26.0" \
        "requests>=2.28" \
        "bitsandbytes>=0.49.2" \
        "peft>=0.13.0" \
        wandb

# The fork itself is cloned at runtime by setup.sh (so the image isn't tied to a
# commit). To bake it in instead, COPY it here and `pip install -e .`.

CMD ["/bin/bash"]
