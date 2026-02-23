# src/softargmax.py
from __future__ import annotations
import torch
import torch.nn.functional as F


def soft_argmax_2d(heatmaps: torch.Tensor, beta: float = 50.0, eps: float = 1e-8):
    """
    Soft-argmax 2D per heatmaps.
    Args:
      heatmaps: (B, K, H, W) logits or unnormalized heatmaps
      beta: sharpening (higher -> closer to argmax)
    Returns:
      coords: (B, K, 2)  in heatmap coordinates (x,y) with subpixel precision
      conf:   (B, K)     soft confidence ~ max probability (useful for weighting)
    """
    B, K, H, W = heatmaps.shape

    # Flatten spatial dims
    x = heatmaps.view(B, K, -1) * beta
    p = F.softmax(x, dim=-1)  # (B,K,H*W)

    # coordinate grid in heatmap space
    ys = torch.linspace(0, H - 1, H, device=heatmaps.device,
                        dtype=heatmaps.dtype)
    xs = torch.linspace(0, W - 1, W, device=heatmaps.device,
                        dtype=heatmaps.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)  # (H*W,2)

    coords = torch.matmul(p, grid)  # (B,K,2) expected value
    conf = p.max(dim=-1).values     # (B,K)
    return coords, conf


def hm_to_image_coords(coords_hm: torch.Tensor, heatmap_size=(128, 128), img_size=(512, 512)):
    """
    Convert coords from heatmap pixel space to image pixel space.
    coords_hm: (B,K,2) in heatmap coords (x,y)
    """
    Hm, Wm = heatmap_size
    Wi, Hi = img_size
    sx = Wi / float(Wm)
    sy = Hi / float(Hm)
    out = coords_hm.clone()
    out[..., 0] *= sx
    out[..., 1] *= sy
    return out
