"""Flow-matching SFT loss for Ideogram 4 (precompute-and-free architecture).

Ideogram 4 predicts a velocity field v(z_t, t). Target:
    z0 = VAE-encoded image latent tokens ; z1 = noise
    zt = (1-t)*z0 + t*z1 ; target_v = z1 - z0
    loss = MSE(transformer(zt, t, caption), target_v)   # over image-token positions

Because the text encoder (Qwen3-VL) and VAE are frozen, we encode every sample
ONCE up front (`precompute_sample`) and cache the tensors, so the big frozen
models can be freed before training. Only the conditional transformer + LoRA stay
resident — this is what keeps a 9.3B model + backward pass inside 24GB.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange


@torch.no_grad()
def encode_pixels_to_tokens(pipe, pixel_values: torch.Tensor) -> torch.Tensor:
    """[B,3,H,W] in [-1,1] -> normalized DiT latent tokens [B, grid_h*grid_w, 128]."""
    moments = pipe.autoencoder.encoder(pixel_values.to(pipe.device, pipe.dtype))
    mean, _logvar = moments.chunk(2, dim=1)
    p = pipe.config.patch_size
    z = rearrange(mean, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=p, p2=p)
    z = (z.float() - pipe.latent_shift[None, None, :]) / pipe.latent_scale[None, None, :]
    return z


@torch.no_grad()
def precompute_sample(pipe, caption_json: str, pixel_values: torch.Tensor, height: int, width: int) -> dict:
    """Encode one (caption, image) pair into cached tensors (needs text encoder + VAE).

    Returns everything the training step needs, so the frozen encoders can be
    freed afterward. Done per-sample (batch of 1) so each keeps its own seq length.
    """
    z0 = encode_pixels_to_tokens(pipe, pixel_values)  # [1, N_img, 128]
    inputs = pipe._build_inputs([caption_json], height=height, width=width)
    llm = pipe._encode_text(inputs["token_ids"], inputs["text_position_ids"], inputs["indicator"])
    return {
        "z0": z0,
        "llm_features": llm,
        "position_ids": inputs["position_ids"],
        "segment_ids": inputs["segment_ids"],
        "indicator": inputs["indicator"],
        "max_text_tokens": inputs["max_text_tokens"],
    }


def sample_timesteps(batch_size: int, schedule, device, uniform: bool = False) -> torch.Tensor:
    """Draw t in (0,1) for training.

    uniform=True samples t ~ Uniform(0,1) so the adapter trains across the FULL
    timestep range the sampler queries at inference (high-noise early steps
    included). The schedule path concentrates t in a narrow band, which leaves the
    LoRA untrained at high t and corrupts the denoising trajectory -> noise output.
    """
    u = torch.rand(batch_size, device=device).clamp_(1e-4, 1.0 - 1e-4)
    return u if uniform else schedule(u)


def flow_matching_loss_cached(transformer, sample: dict, schedule, device, uniform: bool = False) -> torch.Tensor:
    """Flow-matching MSE for one precomputed sample (no text encoder / VAE needed)."""
    z0 = sample["z0"]
    batch_size, _num_img, latent_dim = z0.shape

    t = sample_timesteps(batch_size, schedule, device, uniform=uniform)
    z1 = torch.randn_like(z0)  # noise
    t_ = t[:, None, None]
    # Ideogram convention (read off the sampler): t=0 -> noise, t=1 -> clean.
    # z0 is the clean VAE latent. zt interpolates noise(t=0) -> clean(t=1);
    # velocity v = dz/dt = z0 - z1. (Earlier code had this time-axis FLIPPED,
    # which trained the model against an inverted schedule -> noise at inference.)
    zt = t_ * z0 + (1.0 - t_) * z1
    target_v = z0 - z1

    mt = sample["max_text_tokens"]
    text_pad = torch.zeros(batch_size, mt, latent_dim, dtype=torch.float32, device=device)
    pos_z = torch.cat([text_pad, zt], dim=1)

    pred = transformer(
        llm_features=sample["llm_features"],
        x=pos_z,
        t=t,
        position_ids=sample["position_ids"],
        segment_ids=sample["segment_ids"],
        indicator=sample["indicator"],
    )
    pred_v = pred[:, mt:]
    return F.mse_loss(pred_v.float(), target_v.float())
