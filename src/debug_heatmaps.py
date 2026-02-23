#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
src/debug_heatmaps.py

Debug qualitative HRNet heatmaps + verify coordinate "space" consistency.

What it does:
- Loads SpeedPlusKeypointDataset (image + GT heatmaps + GT 2D keypoints if available).
- Runs HRNetKeypointModel -> predicted heatmaps (logits).
- Aligns pred heatmaps to GT heatmap size (bilinear).
- Applies sigmoid -> predicted heatmaps in [0,1] (like your original).
- Saves:
  - base image
  - overlay of max-heatmap (GT) on image
  - overlay of max-heatmap (PRED) on image
  - single keypoint heatmaps (GT + PRED auto/fixed)
  - keypoints overlay images:
      * GT only
      * PRED only
      * GT+PRED
- Prints:
  - heatmap sizes
  - per-keypoint heatmap stats
  - coordinate ranges to verify whether GT is in heatmap space (e.g. 0..127) or image space (0..511)

Usage examples:
  python -m src.debug_heatmaps --split val --domain synthetic --ckpt runs/.../best.pt
  python -m src.debug_heatmaps --fixed_k 0,1,2 --method softargmax
"""

import argparse
import random
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.dataset_speedplus import SpeedPlusKeypointDataset
from src.model_hrnet_pose import HRNetKeypointModel

# Optional: softargmax (if you want)
try:
    from src.softargmax import softargmax_2d  # if your file exposes this
except Exception:
    softargmax_2d = None


# -------------------------
# Utilities
# -------------------------
def hm_stats(hm: np.ndarray):
    """Return (peak, mean, peak/mean)."""
    hm = hm.astype(np.float32)
    peak = float(hm.max())
    mean = float(hm.mean())
    p2m = peak / (mean + 1e-8)
    return peak, mean, p2m


def normalize_autostretch(hm: np.ndarray) -> np.ndarray:
    """Normalize heatmap to [0,1] using its own min/max."""
    hm = hm.astype(np.float32)
    hm = hm - hm.min()
    hm = hm / (hm.max() + 1e-8)
    return hm


def normalize_fixed(hm: np.ndarray, vmin: float = 0.0, vmax: float = 1.0) -> np.ndarray:
    """Clamp and map [vmin,vmax] -> [0,1]."""
    hm = hm.astype(np.float32)
    hm = np.clip(hm, vmin, vmax)
    hm = (hm - vmin) / (vmax - vmin + 1e-8)
    return hm


def overlay_heatmap_on_image(
    img_bgr: np.ndarray,
    hm_2d: np.ndarray,
    alpha: float = 0.35,
    fixed_scale: bool = False,
    smooth: bool = True,
) -> np.ndarray:
    """
    Correct visualization:
      - resize in float
      - normalize (auto or fixed)
      - convert to uint8
      - applyColorMap
    """
    H, W = img_bgr.shape[:2]
    interp = cv2.INTER_LINEAR if smooth else cv2.INTER_NEAREST

    # 1) resize FLOAT first (important!)
    hm_resized = cv2.resize(hm_2d.astype(np.float32),
                            (W, H), interpolation=interp)

    # 2) normalize
    if fixed_scale:
        hm01 = normalize_fixed(hm_resized, vmin=0.0, vmax=1.0)
    else:
        hm01 = normalize_autostretch(hm_resized)

    # 3) to uint8
    hm_u8 = (hm01 * 255.0).clip(0, 255).astype(np.uint8)

    # 4) colormap + blend
    hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 1.0 - alpha, hm_color, alpha, 0.0)


def save_single_heatmaps(
    out_dir: Path,
    stem: str,
    hmaps: np.ndarray,
    tag: str,
    k_list,
    fixed_scale: bool = False,
):
    """
    Save selected keypoint heatmaps as color images.
    NOTE: save at heatmap resolution.
    """
    for k in k_list:
        hm = hmaps[k].astype(np.float32)

        if fixed_scale:
            hm01 = normalize_fixed(hm, 0.0, 1.0)
            suffix = "fixed"
        else:
            hm01 = normalize_autostretch(hm)
            suffix = "auto"

        hm_u8 = (hm01 * 255.0).clip(0, 255).astype(np.uint8)
        hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
        cv2.imwrite(
            str(out_dir / f"{stem}_{tag}_hm{k:02d}_{suffix}.jpg"), hm_color)


def load_checkpoint_if_any(model: torch.nn.Module, ckpt_path: str | None, device: str):
    if not ckpt_path:
        print(
            "[INFO] No checkpoint provided -> predicted heatmaps will be from random weights.")
        return

    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    new_state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(new_state, strict=False)

    print(f"[INFO] Loaded checkpoint: {ckpt_path}")
    if missing:
        print("[WARN] Missing keys (up to 10):", missing[:10])
    if unexpected:
        print("[WARN] Unexpected keys (up to 10):", unexpected[:10])


def find_gt_kpts_key(sample: dict) -> Optional[str]:
    """
    Try to find which key in the dataset sample contains GT 2D keypoints.
    We don't assume exact naming; we search common variants.
    """
    candidates = [
        "kpts_2d", "keypoints_2d", "kpts2d", "kpts", "keypoints",
        "kpts_img", "kpts_image", "uv", "kp2d"
    ]
    for k in candidates:
        if k in sample:
            return k
    return None


def extract_pred_kpts_from_heatmaps(
    h_pred_sig: np.ndarray,
    method: str = "argmax",
    beta: float = 20.0,
) -> np.ndarray:
    """
    Returns pred keypoints in HEATMAP pixel space as (K,2) with (x,y).
    method:
      - argmax: integer peak location
      - softargmax: differentiable expectation (requires src.softargmax exposing softargmax_2d)
    """
    K, H, W = h_pred_sig.shape

    if method == "argmax":
        flat_idx = h_pred_sig.reshape(K, -1).argmax(axis=1)
        y = flat_idx // W
        x = flat_idx % W
        return np.stack([x, y], axis=1).astype(np.float32)

    if method == "softargmax":
        if softargmax_2d is None:
            raise RuntimeError(
                "softargmax method requested but src.softargmax.softargmax_2d was not found/importable."
            )
        # softargmax expects torch tensors; we pass logits-like (not necessarily sigmoid)
        # Here we use logit of sigmoid for numeric stability; clamp to avoid inf.
        h = np.clip(h_pred_sig, 1e-6, 1.0 - 1e-6)
        logits = np.log(h / (1.0 - h)).astype(np.float32)  # inverse sigmoid
        t = torch.from_numpy(logits).unsqueeze(0)  # (1,K,H,W)
        xy = softargmax_2d(t, beta=beta)  # expected (1,K,2) in (x,y)
        return xy[0].detach().cpu().numpy().astype(np.float32)

    raise ValueError(f"Unknown method: {method}")


def hm_to_img_coords(
    xy_hm: np.ndarray,
    heatmap_size: Tuple[int, int],
    img_size: Tuple[int, int],
) -> np.ndarray:
    """
    Convert coords from heatmap space (W_hm,H_hm) to image space (W_img,H_img).
    Uses linear scaling based on [0..W-1] mapping.

    xy_hm: (K,2) in (x,y)
    """
    W_hm, H_hm = heatmap_size
    W_img, H_img = img_size

    sx = (W_img - 1) / max(1, (W_hm - 1))
    sy = (H_img - 1) / max(1, (H_hm - 1))

    xy_img = xy_hm.copy().astype(np.float32)
    xy_img[:, 0] *= sx
    xy_img[:, 1] *= sy
    return xy_img


def draw_kpts(
    img_bgr: np.ndarray,
    xy: np.ndarray,
    color: Tuple[int, int, int],
    label: str = "",
    radius: int = 3,
    font_scale: float = 0.45,
) -> np.ndarray:
    """
    Draw keypoints (x,y) with index labels.
    """
    out = img_bgr.copy()
    for k, (x, y) in enumerate(xy):
        xi, yi = int(round(float(x))), int(round(float(y)))
        cv2.circle(out, (xi, yi), radius, color, -1)
        cv2.putText(
            out,
            str(k),
            (xi + 4, yi - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            1,
            cv2.LINE_AA,
        )
    if label:
        cv2.putText(
            out,
            label,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


# -------------------------
# Main
# -------------------------
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="val",
                    choices=["train", "val", "test"])
    ap.add_argument("--domain", type=str, default="synthetic")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--out", type=str, default="outputs/debug_heatmaps")
    ap.add_argument("--num_single", type=int, default=8)
    ap.add_argument("--fixed_k", type=str, default="")
    ap.add_argument("--nearest", action="store_true",
                    help="nearest resize for overlays (blocky)")
    ap.add_argument("--alpha", type=float, default=0.40)

    # NEW: kpt extraction method
    ap.add_argument("--method", type=str, default="argmax",
                    choices=["argmax", "softargmax"])
    ap.add_argument("--beta", type=float, default=20.0,
                    help="softargmax temperature beta (if method=softargmax)")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] device:", device)

    # Keep your sizes explicit
    # (W,H) conceptually, but dataset uses (H,W); we store tuple anyway
    IMG_SIZE = (512, 512)
    HEATMAP_SIZE = (128, 128)

    ds = SpeedPlusKeypointDataset(
        split=args.split,
        domain=args.domain,
        img_size=(IMG_SIZE[1], IMG_SIZE[0]),          # (H,W)
        heatmap_size=(HEATMAP_SIZE[1], HEATMAP_SIZE[0]),
        sigma=2.0,
        return_heatmaps=True,
        normalize_rgb=False,
    )
    print("[INFO] dataset size:", len(ds))

    sample0 = ds[0]
    print("[INFO] sample keys:", list(sample0.keys()))
    gt_kpts_key = find_gt_kpts_key(sample0)
    if gt_kpts_key:
        print(f"[INFO] Found GT 2D keypoints key: '{gt_kpts_key}'")
    else:
        print("[WARN] No GT 2D keypoints key found in sample. Will only debug heatmaps.")

    K = int(sample0["heatmaps"].shape[0])
    print("[INFO] num keypoints:", K)

    model = HRNetKeypointModel(num_keypoints=K).to(device).eval()
    load_checkpoint_if_any(model, args.ckpt, device)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    idxs = random.sample(range(len(ds)), k=min(args.n, len(ds)))

    # fixed selection or random selection of keypoints for heatmap saving
    if args.fixed_k.strip():
        k_list_global = [int(x)
                         for x in args.fixed_k.split(",") if x.strip() != ""]
        k_list_global = [k for k in k_list_global if 0 <= k < K]
        if not k_list_global:
            raise ValueError("fixed_k provided but empty after filtering.")
        print("[INFO] Using fixed keypoints:", k_list_global)
    else:
        k_list_global = None

    smooth_overlay = not args.nearest

    for i, idx in enumerate(idxs):
        s = ds[idx]

        # Image: dataset gives RGB float [0,1] CHW
        img = s["image"].numpy().transpose(1, 2, 0)  # HWC RGB float
        img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
        H_img, W_img = img_bgr.shape[:2]

        # GT heatmaps (K,Hg,Wg)
        h_gt = s["heatmaps"].numpy().astype(np.float32)
        Hg, Wg = int(h_gt.shape[1]), int(h_gt.shape[2])
        hm_gt_max = h_gt.max(axis=0)

        # Forward
        x = s["image"].unsqueeze(0).to(device)   # (1,3,H,W)
        pred_logits = model(x)                   # (1,K,Hp,Wp)
        Hp, Wp = int(pred_logits.shape[-2]), int(pred_logits.shape[-1])

        # Align logits -> GT heatmap size (correct)
        if (Hp, Wp) != (Hg, Wg):
            pred_logits = F.interpolate(
                pred_logits, size=(Hg, Wg), mode="bilinear", align_corners=False
            )

        pred_logits_np = pred_logits[0].detach().cpu(
        ).numpy().astype(np.float32)  # (K,Hg,Wg)
        print(
            f"[DBG] idx={idx} GT_hm={(Hg, Wg)} | PRED_raw={(Hp, Wp)} | PRED_aligned={tuple(pred_logits.shape[-2:])}")

        # Sigmoid -> predicted heatmaps in [0,1]
        h_pred_sig = torch.sigmoid(
            pred_logits[0]).detach().cpu().numpy().astype(np.float32)
        hm_pred_max = h_pred_sig.max(axis=0)

        stem = Path(s.get("img_name", f"idx{idx:06d}")).stem

        # Save base image
        cv2.imwrite(str(out_dir / f"{stem}_img.jpg"), img_bgr)

        # Overlays of max-heatmap
        ov_gt = overlay_heatmap_on_image(
            img_bgr, hm_gt_max, alpha=args.alpha, fixed_scale=False, smooth=smooth_overlay
        )
        cv2.imwrite(str(out_dir / f"{stem}_GT_overlay.jpg"), ov_gt)

        ov_pr = overlay_heatmap_on_image(
            img_bgr, hm_pred_max, alpha=args.alpha, fixed_scale=True, smooth=smooth_overlay
        )
        cv2.imwrite(str(out_dir / f"{stem}_PRED_overlay_fixed.jpg"), ov_pr)

        # Keypoints selection for single heatmaps
        if k_list_global is None:
            num_single = min(args.num_single, K)
            k_list = random.sample(range(K), k=num_single)
        else:
            k_list = k_list_global

        # Save GT single (auto)
        save_single_heatmaps(out_dir, stem, h_gt, "GT",
                             k_list, fixed_scale=False)

        # Save PRED single: auto + fixed
        save_single_heatmaps(out_dir, stem, h_pred_sig,
                             "PRED", k_list, fixed_scale=False)
        save_single_heatmaps(out_dir, stem, h_pred_sig,
                             "PRED", k_list, fixed_scale=True)

        # ---- NEW: coordinate space debug ----
        # Pred coords in HEATMAP space (x,y)
        pred_xy_hm = extract_pred_kpts_from_heatmaps(
            h_pred_sig, method=args.method, beta=args.beta)

        print(
            f"[DBG] pred_xy_hm x range: {pred_xy_hm[:, 0].min():.1f}-{pred_xy_hm[:, 0].max():.1f} (expect 0-{Wg-1})")
        print(
            f"[DBG] pred_xy_hm y range: {pred_xy_hm[:, 1].min():.1f}-{pred_xy_hm[:, 1].max():.1f} (expect 0-{Hg-1})")

        # If GT keypoints exist, read and decide their space
        if gt_kpts_key is not None:
            gt_xy = s[gt_kpts_key]
            if torch.is_tensor(gt_xy):
                gt_xy = gt_xy.detach().cpu().numpy()
            gt_xy = np.asarray(gt_xy, dtype=np.float32)

            # Some datasets store shape (K,2) or (2,K) - handle basic case
            if gt_xy.ndim == 2 and gt_xy.shape[0] == 2 and gt_xy.shape[1] == K:
                gt_xy = gt_xy.T

            if gt_xy.shape[0] != K or gt_xy.shape[1] != 2:
                print(
                    f"[WARN] GT keypoints shape unexpected: {gt_xy.shape} (expected Kx2). Skipping kpt overlay.")
                gt_xy = None

            if gt_xy is not None:
                print(
                    f"[DBG] gt_xy x range: {gt_xy[:, 0].min():.1f}-{gt_xy[:, 0].max():.1f}")
                print(
                    f"[DBG] gt_xy y range: {gt_xy[:, 1].min():.1f}-{gt_xy[:, 1].max():.1f}")

                # Decide if GT is in heatmap space or image space by checking max range
                gt_max = max(float(gt_xy[:, 0].max()),
                             float(gt_xy[:, 1].max()))
                # heuristic threshold: if > max(Hg,Wg)*1.5 => image space
                gt_is_image_space = gt_max > max(Hg, Wg) * 1.5

                if gt_is_image_space:
                    # Convert pred to image space for overlay
                    pred_xy_img = hm_to_img_coords(
                        pred_xy_hm,
                        heatmap_size=(Wg, Hg),
                        img_size=(W_img, H_img),
                    )
                    gt_xy_img = gt_xy
                    print(
                        "[INFO] GT appears to be in IMAGE space -> overlay uses pred rescaled HM->IMG")
                else:
                    # Convert both to image for overlay (better visualization)
                    pred_xy_img = hm_to_img_coords(
                        pred_xy_hm,
                        heatmap_size=(Wg, Hg),
                        img_size=(W_img, H_img),
                    )
                    gt_xy_img = hm_to_img_coords(
                        gt_xy,
                        heatmap_size=(Wg, Hg),
                        img_size=(W_img, H_img),
                    )
                    print(
                        "[INFO] GT appears to be in HEATMAP space -> overlay uses both rescaled HM->IMG")

                # Save kpt overlay images
                gt_only = draw_kpts(img_bgr, gt_xy_img,
                                    (0, 255, 0), label="GT (green)")
                pr_only = draw_kpts(
                    img_bgr, pred_xy_img, (0, 0, 255), label=f"PRED {args.method} (red)")
                both = img_bgr.copy()
                both = draw_kpts(both, gt_xy_img, (0, 255, 0), label="")
                both = draw_kpts(both, pred_xy_img, (0, 0, 255),
                                 label="GT (green) + PRED (red)")

                cv2.imwrite(str(out_dir / f"{stem}_kpts_GT_only.jpg"), gt_only)
                cv2.imwrite(
                    str(out_dir / f"{stem}_kpts_PRED_only_{args.method}.jpg"), pr_only)
                cv2.imwrite(
                    str(out_dir / f"{stem}_kpts_GT_PRED_{args.method}.jpg"), both)

                # Also print an approximate scale ratio (helps spot mismatch)
                sx = (gt_xy[:, 0].max() + 1e-6) / \
                    (pred_xy_hm[:, 0].max() + 1e-6)
                sy = (gt_xy[:, 1].max() + 1e-6) / \
                    (pred_xy_hm[:, 1].max() + 1e-6)
                print(
                    f"[DBG] approx scale gt/pred (raw spaces): sx={sx:.2f}, sy={sy:.2f}")

        # Print stats for selected heatmaps
        print(f"\n[{i+1}/{len(idxs)}] {stem} (idx={idx})")
        for k in k_list:
            gt_peak, gt_mean, gt_p2m = hm_stats(h_gt[k])
            pr_peak, pr_mean, pr_p2m = hm_stats(h_pred_sig[k])
            print(
                f"  GT   hm{k:02d}: peak={gt_peak:.4f}  mean={gt_mean:.6f}  peak/mean={gt_p2m:.1f}")
            print(
                f"  PRED hm{k:02d}: peak={pr_peak:.4f} mean={pr_mean:.6f} peak/mean={pr_p2m:.1f}")

    print("\n[DONE] saved to:", out_dir.resolve())


if __name__ == "__main__":
    main()
