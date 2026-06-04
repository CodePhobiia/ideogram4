"""QLoRA adapters on Ideogram 4's conditional transformer.

The nf4 build swaps every nn.Linear to bitsandbytes `Linear4bit` (see
ideogram4.quantized_loading), which PEFT supports — so this is textbook QLoRA:
attach low-rank adapters to the attention + MLP linears, freeze everything else
(text encoder, VAE, unconditional transformer stay frozen elsewhere).

Module names inside Ideogram4Transformer:
    layers.{i}.attention.qkv, layers.{i}.attention.o
    layers.{i}.feed_forward.{w1,w2,w3}
PEFT matches `target_modules` by name suffix, so the short names below hit all 34 layers.
"""

from __future__ import annotations

import torch.nn as nn
from peft import LoraConfig, get_peft_model

# v1: attention projections + the SwiGLU MLP. (input_proj / llm_cond_proj /
# final_layer.linear can be added later if the adapter underfits.)
DEFAULT_TARGETS = ["qkv", "o", "w1", "w2", "w3"]


def add_qlora(
    transformer: nn.Module,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    targets: list[str] | None = None,
):
    """Wrap the conditional transformer with LoRA adapters; returns the PeftModel.

    The returned module is call-compatible with the original — the pipeline calls
    it with (llm_features=, x=, t=, position_ids=, segment_ids=, indicator=) and PEFT
    forwards those straight through to the base transformer.
    """
    cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=targets or DEFAULT_TARGETS,
        bias="none",
        task_type=None,  # not an HF task head; plain module injection
    )
    peft_model = get_peft_model(transformer, cfg)
    return peft_model


def trainable_report(model: nn.Module) -> str:
    trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in model.parameters())
    return f"trainable params: {trn:,} / {tot:,} ({100 * trn / tot:.3f}%)"
