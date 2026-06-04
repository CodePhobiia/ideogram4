"""Flow-matching SFT loss for Ideogram 4.

Ideogram 4 predicts a velocity field v(z_t, t) (not noise). Training target:
    z0 = VAE-encoded image latent tokens
    z1 = gaussian noise
    zt = (1 - t) * z0 + t * z1
    target_v = z1 - z0
    loss = MSE(transformer(zt, t, caption), target_v)   # over image-token positions

We reuse the pipeline's own input plumbing (`_build_inputs`, `_encode_text`) so the
packed text+image sequence, position/segment ids, and asymmetric layout exactly match
inference. Only the conditional transformer trains; text encoder + VAE stay frozen.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange


@torch.no_grad()
def encode_pixels_to_tokens(pipe, pixel_values: torch.Tensor) -> torch.Tensor:
    """[B,3,H,W] in [-1,1]  ->  normalized DiT latent tokens [B, grid_h*grid_w, 128]."""
    moments = pipe.autoencoder.encoder(pixel_values.to(pipe.device, pipe.dtype))
    mean, _logvar = moments.chunk(2, dim=1)  # use the mean (no sampling) for a stable target
    p = pipe.config.patch_size
    z = rearrange(mean, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=p, p2=p)
    z = (z.float() - pipe.latent_shift[None, None, :]) / pipe.latent_scale[None, None, :]
    return z


def sample_timesteps(batch_size: int, schedule, device) -> torch.Tensor:
    """Draw t in (0,1) from the model's logit-normal schedule (resolution-aware)."""
    # clamp away from {0,1}: the schedule applies ndtri (probit), inf at the ends.
    u = torch.rand(batch_size, device=device).clamp_(1e-4, 1.0 - 1e-4)
    return schedule(u)


def flow_matching_loss(
    pipe,
    transformer,
    caption_jsons: list[str],
    pixel_values: torch.Tensor,
    height: int,
    width: int,
    schedule,
) -> torch.Tensor:
    """One flow-matching MSE loss over a (caption, image) batch.

    `transformer` is the trainable conditional transformer (LoRA-wrapped). It is
    called with the same kwargs the pipeline uses at inference.
    """
    z0 = encode_pixels_to_tokens(pipe, pixel_values)  # [B, N_img, 128]
    batch_size, _num_img, latent_dim = z0.shape

    inputs = pipe._build_inputs(caption_jsons, height=height, width=width)
    with torch.no_grad():
        llm_features = pipe._encode_text(
            inputs["token_ids"], inputs["text_position_ids"], inputs["indicator"]
        )

    t = sample_timesteps(batch_size, schedule, pipe.device)  # [B]
    z1 = torch.randn_like(z0)
    t_ = t[:, None, None]
    zt = (1.0 - t_) * z0 + t_ * z1
    target_v = z1 - z0

    # The conditional branch sees [text-pad latents | image latents]; only the image
    # positions carry zt, matching the pipeline's pos_z construction.
    max_text = inputs["max_text_tokens"]
    text_pad = torch.zeros(batch_size, max_text, latent_dim, dtype=torch.float32, device=pipe.device)
    pos_z = torch.cat([text_pad, zt], dim=1)

    pred = transformer(
        llm_features=llm_features,
        x=pos_z,
        t=t,
        position_ids=inputs["position_ids"],
        segment_ids=inputs["segment_ids"],
        indicator=inputs["indicator"],
    )
    pred_v = pred[:, max_text:]  # keep only image-token velocities
    return F.mse_loss(pred_v.float(), target_v.float())
