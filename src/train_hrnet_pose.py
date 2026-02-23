# src/train_hrnet_pose.py
from __future__ import annotations

from src.config import KEYPOINTS_3D_NPY, assert_dataset_layout
from src.dataset_speedplus import SpeedPlusKeypointDataset
from src.model_hrnet_pose import HRNetKeypointModel
from src.softargmax import soft_argmax_2d

from tqdm import tqdm
from torch.utils.data import DataLoader

import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

import argparse
import csv
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import matplotlib
matplotlib.use("Agg")


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass
class TrainCfg:
    # data
    img_size: Tuple[int, int] = (512, 512)      # (W,H)
    heatmap_size: Tuple[int, int] = (128, 128)  # (Wm,Hm)
    sigma: float = 2.0
    domain: str = "synthetic"

    # training
    epochs: int = 55
    batch_size: int = 8
    lr: float = 1e-4
    num_workers: int = 4

    # losses
    use_coord_loss: bool = True

    # lambda schedule (IMPORTANT)
    lambda_coord_max: float = 0.3
    lambda_warmup_epochs: int = 5     # epochs with lambda=0
    lambda_ramp_epochs: int = 10      # epochs to reach lambda_max

    beta_softarg: float = 30.0        # softargmax temperature

    # coord loss settings
    # SmoothL1 beta in normalized [0,1] space
    coord_smoothl1_beta: float = 0.02
    use_coord_peak_gate: bool = True
    # only apply coord loss for kpts with peak(sigmoid(logit)) >= thr
    coord_peak_thr: float = 0.35

    hm_loss_type: str = "weighted_mse"  # ["weighted_mse", "mse"]
    pos_weight: float = 20.0
    thr: float = 0.2

    # optional diagnostics: gradient ratio logging
    log_grad_norm_every: int = 0       # 0 disables; else every N steps log grad norms
    grad_layer_name: str = "final_layer"  # which submodule to probe

    # runs/logging
    runs_dir: str = "runs"
    run_tag: str = "hrnet_hmCoord_weightedMSE_pw20_lambdaSched"
    save_every_epochs: int = 1

    # debug heatmaps dump
    debug_hm_every: int = 1   # 0 disables
    debug_hm_n: int = 6
    debug_hm_k: str = "0,3,8"
    debug_hm_seed: int = 0


def parse_args():
    ap = argparse.ArgumentParser()

    # training
    ap.add_argument("--epochs", type=int, default=TrainCfg.epochs)
    ap.add_argument("--batch_size", type=int, default=TrainCfg.batch_size)
    ap.add_argument("--lr", type=float, default=TrainCfg.lr)
    ap.add_argument("--num_workers", type=int, default=TrainCfg.num_workers)

    # losses
    ap.add_argument("--use_coord_loss", type=int, default=1,
                    help="1=use coord loss, 0=HM-only")
    ap.add_argument("--lambda_coord_max", type=float,
                    default=TrainCfg.lambda_coord_max)
    ap.add_argument("--lambda_warmup_epochs", type=int,
                    default=TrainCfg.lambda_warmup_epochs)
    ap.add_argument("--lambda_ramp_epochs", type=int,
                    default=TrainCfg.lambda_ramp_epochs)
    ap.add_argument("--beta_softarg", type=float,
                    default=TrainCfg.beta_softarg)

    ap.add_argument("--coord_smoothl1_beta", type=float,
                    default=TrainCfg.coord_smoothl1_beta)
    ap.add_argument("--use_coord_peak_gate", type=int,
                    default=int(TrainCfg.use_coord_peak_gate))
    ap.add_argument("--coord_peak_thr", type=float,
                    default=TrainCfg.coord_peak_thr)

    ap.add_argument("--hm_loss_type", type=str,
                    default=TrainCfg.hm_loss_type, choices=["weighted_mse", "mse"])
    ap.add_argument("--pos_weight", type=float, default=TrainCfg.pos_weight)
    ap.add_argument("--thr", type=float, default=TrainCfg.thr)

    ap.add_argument("--log_grad_norm_every", type=int,
                    default=TrainCfg.log_grad_norm_every)
    ap.add_argument("--grad_layer_name", type=str,
                    default=TrainCfg.grad_layer_name)

    # runs
    ap.add_argument("--runs_dir", type=str, default=TrainCfg.runs_dir)
    ap.add_argument("--run_tag", type=str, default=TrainCfg.run_tag)
    ap.add_argument("--save_every_epochs", type=int,
                    default=TrainCfg.save_every_epochs)

    # debug heatmaps
    ap.add_argument("--debug_hm_every", type=int,
                    default=TrainCfg.debug_hm_every)
    ap.add_argument("--debug_hm_n", type=int, default=TrainCfg.debug_hm_n)
    ap.add_argument("--debug_hm_k", type=str, default=TrainCfg.debug_hm_k)
    ap.add_argument("--debug_hm_seed", type=int,
                    default=TrainCfg.debug_hm_seed)

    # checkpoint
    ap.add_argument("--resume_ckpt", type=str, default="")
    ap.add_argument("--resume_run_dir", type=str, default="")

    args = ap.parse_args()

    cfg = TrainCfg(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,

        use_coord_loss=bool(args.use_coord_loss),

        lambda_coord_max=float(args.lambda_coord_max),
        lambda_warmup_epochs=int(args.lambda_warmup_epochs),
        lambda_ramp_epochs=int(args.lambda_ramp_epochs),
        beta_softarg=float(args.beta_softarg),

        coord_smoothl1_beta=float(args.coord_smoothl1_beta),
        use_coord_peak_gate=bool(args.use_coord_peak_gate),
        coord_peak_thr=float(args.coord_peak_thr),

        hm_loss_type=str(args.hm_loss_type),
        pos_weight=float(args.pos_weight),
        thr=float(args.thr),

        log_grad_norm_every=int(args.log_grad_norm_every),
        grad_layer_name=str(args.grad_layer_name),

        runs_dir=str(args.runs_dir),
        run_tag=str(args.run_tag),
        save_every_epochs=int(args.save_every_epochs),

        debug_hm_every=int(args.debug_hm_every),
        debug_hm_n=int(args.debug_hm_n),
        debug_hm_k=str(args.debug_hm_k),
        debug_hm_seed=int(args.debug_hm_seed),
    )
    return cfg, args


# -----------------------------------------------------------------------------
# Run utils
# -----------------------------------------------------------------------------
SCRIPT_SIGNATURE = "train_hrnet_pose.py :: v3 :: lambda schedule + coord peak-gate + logging lambda*coord + optional grad norms"


def make_run_dir(runs_dir: Path, run_tag: str) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = runs_dir / f"{ts}_{run_tag}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    (run_dir / "debug_heatmaps").mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def append_csv_row(csv_path: Path, header: List[str], row: List):
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(header)
        w.writerow(row)


def plot_loss(out_png: Path, epochs, train_vals, val_vals, title: str, logy: bool = False):
    plt.figure()
    plt.plot(epochs, train_vals, label="train")
    plt.plot(epochs, val_vals, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title)
    if logy:
        plt.yscale("log")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_all_losses_log(out_png: Path, epochs, series: Dict[str, List[float]]):
    plt.figure()
    for name, vals in series.items():
        plt.plot(epochs, vals, label=name)
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss (log)")
    plt.title("All losses (log scale)")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=220)
    plt.close()


# -----------------------------------------------------------------------------
# Heatmap debug dump (unchanged, kept for your workflow)
# -----------------------------------------------------------------------------
def hm_stats(hm: np.ndarray):
    hm = hm.astype(np.float32)
    peak = float(hm.max())
    mean = float(hm.mean())
    p2m = peak / (mean + 1e-8)
    return peak, mean, p2m


def to_uint8_hm_autostretch(hm: np.ndarray) -> np.ndarray:
    hm = hm.astype(np.float32)
    hm = hm - hm.min()
    hm = hm / (hm.max() + 1e-8)
    return (hm * 255.0).astype(np.uint8)


def to_uint8_hm_fixed(hm: np.ndarray, vmin: float = 0.0, vmax: float = 1.0) -> np.ndarray:
    hm = hm.astype(np.float32)
    hm = np.clip(hm, vmin, vmax)
    hm = (hm - vmin) / (vmax - vmin + 1e-8)
    return (hm * 255.0).astype(np.uint8)


def overlay_heatmap_on_image(img_bgr: np.ndarray, hm_2d: np.ndarray, alpha=0.40, fixed_scale: bool = False) -> np.ndarray:
    if fixed_scale:
        hm_u8 = to_uint8_hm_fixed(hm_2d, vmin=0.0, vmax=1.0)
    else:
        hm_u8 = to_uint8_hm_autostretch(hm_2d)

    hm_u8 = cv2.resize(
        hm_u8, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 1.0 - alpha, hm_color, alpha, 0.0)


def save_single_heatmaps(out_dir: Path, stem: str, hmaps: np.ndarray, tag: str, k_list: List[int], fixed_scale: bool, vmax: float = 1.0):
    for k in k_list:
        hm = hmaps[k]
        if fixed_scale:
            hm_u8 = to_uint8_hm_fixed(hm, vmin=0.0, vmax=vmax)
            suffix = "fixed"
        else:
            hm_u8 = to_uint8_hm_autostretch(hm)
            suffix = "auto"
        hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
        cv2.imwrite(
            str(out_dir / f"{stem}_{tag}_hm{k:02d}_{suffix}.jpg"), hm_color)


@torch.no_grad()
def dump_debug_heatmaps(model: torch.nn.Module, val_ds: SpeedPlusKeypointDataset, device: torch.device, run_dir: Path, epoch: int, cfg: TrainCfg, num_kpts: int):
    if cfg.debug_hm_every <= 0:
        return
    if (epoch % cfg.debug_hm_every) != 0:
        return

    k_list: List[int] = []
    for s in cfg.debug_hm_k.split(","):
        s = s.strip()
        if not s:
            continue
        try:
            k = int(s)
        except ValueError:
            continue
        if 0 <= k < num_kpts:
            k_list.append(k)
    if not k_list:
        k_list = [0, min(3, num_kpts - 1), min(8, num_kpts - 1)]

    rng = random.Random(cfg.debug_hm_seed)
    n = min(cfg.debug_hm_n, len(val_ds))
    idxs = rng.sample(range(len(val_ds)), k=n)

    out_dir = run_dir / "debug_heatmaps" / f"epoch_{epoch:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[DEBUG] dumping heatmaps to: {out_dir.resolve()} (n={n}, k_list={k_list})")

    stats_lines = [f"epoch={epoch}", f"k_list={k_list}", ""]
    model.eval()

    for idx in idxs:
        s = val_ds[idx]

        img = s["image"].numpy().transpose(1, 2, 0)
        img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)

        h_gt = s["heatmaps"].numpy().astype(np.float32)
        hm_gt_max = h_gt.max(axis=0)

        x = s["image"].unsqueeze(0).to(device)
        pred_logits = model(x)[0].detach().cpu(
        ).numpy().astype(np.float32)  # (K,Hm,Wm)
        pred_sig = 1.0 / (1.0 + np.exp(-pred_logits))
        hm_pred_max = pred_sig.max(axis=0)

        stem = Path(s.get("img_name", f"idx{idx:06d}")).stem

        cv2.imwrite(str(out_dir / f"{stem}_img.jpg"), img_bgr)

        ov_gt = overlay_heatmap_on_image(
            img_bgr, hm_gt_max, alpha=0.40, fixed_scale=False)
        cv2.imwrite(str(out_dir / f"{stem}_GT_overlay.jpg"), ov_gt)

        ov_pr = overlay_heatmap_on_image(
            img_bgr, hm_pred_max, alpha=0.40, fixed_scale=True)
        cv2.imwrite(str(out_dir / f"{stem}_PRED_overlay_fixed.jpg"), ov_pr)

        save_single_heatmaps(out_dir, stem, h_gt, "GT",
                             k_list, fixed_scale=False)
        save_single_heatmaps(out_dir, stem, pred_sig, "PRED",
                             k_list, fixed_scale=True, vmax=1.0)

        stats_lines.append(f"[{stem}] idx={idx}")
        for k in k_list:
            gt_peak, gt_mean, gt_p2m = hm_stats(h_gt[k])
            pr_peak, pr_mean, pr_p2m = hm_stats(pred_sig[k])
            stats_lines.append(
                f"  GT   hm{k:02d}: peak={gt_peak:.4f} mean={gt_mean:.6f} peak/mean={gt_p2m:.1f}")
            stats_lines.append(
                f"  PRED hm{k:02d}: peak={pr_peak:.4f} mean={pr_mean:.6f} peak/mean={pr_p2m:.1f}")
        stats_lines.append("")

    (out_dir / "stats.txt").write_text("\n".join(stats_lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Loss helpers
# -----------------------------------------------------------------------------
def weighted_mse_loss(pred: torch.Tensor, target: torch.Tensor, pos_weight: float, thr: float):
    w = torch.ones_like(target)
    w = w + (pos_weight - 1.0) * (target >= thr).float()
    return ((pred - target) ** 2 * w).mean()


def kpts_img_to_hm_normalized(
    kpts_img: torch.Tensor,
    img_size: Tuple[int, int],      # (W,H)
    heatmap_size: Tuple[int, int],  # (Wm,Hm)
) -> torch.Tensor:
    """
    Convert GT keypoints from image space (px) to normalized heatmap space [0,1].
    kpts_img: (B,K,2) in image pixels
    returns:  (B,K,2) in [0,1] (heatmap normalized)
    """
    W_img, H_img = img_size
    W_hm, H_hm = heatmap_size

    x_hm = kpts_img[..., 0] * (W_hm - 1) / (W_img - 1)
    y_hm = kpts_img[..., 1] * (H_hm - 1) / (H_img - 1)

    x_n = x_hm / (W_hm - 1)
    y_n = y_hm / (H_hm - 1)
    return torch.stack([x_n, y_n], dim=-1)


def lambda_schedule(epoch: int, cfg: TrainCfg) -> float:
    """
    epoch is 1-based.
    warmup: lambda=0
    ramp: linear to lambda_coord_max
    """
    if (not cfg.use_coord_loss) or (cfg.lambda_coord_max <= 0):
        return 0.0

    if epoch <= cfg.lambda_warmup_epochs:
        return 0.0

    e0 = cfg.lambda_warmup_epochs + 1
    if cfg.lambda_ramp_epochs <= 0:
        return float(cfg.lambda_coord_max)

    t = (epoch - e0 + 1) / float(cfg.lambda_ramp_epochs)
    t = max(0.0, min(1.0, t))
    return float(cfg.lambda_coord_max * t)


def compute_losses(
    hmaps_pred_logits: torch.Tensor,
    hmaps_gt: torch.Tensor,
    kpts_2d_gt: torch.Tensor,
    cfg: TrainCfg,
    lambda_eff: float,
):
    """
    Returns:
      loss_total, loss_hm, loss_coord, loss_coord_weighted, coord_mask_frac
    """
    pred_hm = torch.sigmoid(hmaps_pred_logits)

    # heatmap loss
    if cfg.hm_loss_type == "mse":
        loss_hm = torch.mean((pred_hm - hmaps_gt) ** 2)
    elif cfg.hm_loss_type == "weighted_mse":
        loss_hm = weighted_mse_loss(
            pred_hm, hmaps_gt, pos_weight=cfg.pos_weight, thr=cfg.thr)
    else:
        raise ValueError(cfg.hm_loss_type)

    loss_coord = torch.zeros(
        (), device=hmaps_pred_logits.device, dtype=hmaps_pred_logits.dtype)
    coord_mask_frac = torch.tensor(
        0.0, device=hmaps_pred_logits.device, dtype=hmaps_pred_logits.dtype)

    if cfg.use_coord_loss and lambda_eff > 0.0:
        # softargmax coords in heatmap pixel coordinates
        coords_hm_pred, _conf = soft_argmax_2d(
            hmaps_pred_logits, beta=cfg.beta_softarg)  # (B,K,2) in [0..W_hm)

        # normalize pred to [0,1]
        H_hm, W_hm = hmaps_pred_logits.shape[-2], hmaps_pred_logits.shape[-1]
        coords_hm_pred_n = torch.stack(
            [coords_hm_pred[..., 0] / (W_hm - 1),
             coords_hm_pred[..., 1] / (H_hm - 1)],
            dim=-1,
        )

        # GT -> normalized heatmap space
        coords_hm_gt_n = kpts_img_to_hm_normalized(
            kpts_2d_gt, img_size=cfg.img_size, heatmap_size=(W_hm, H_hm)
        )

        # optional gating by predicted heatmap peak per keypoint
        if cfg.use_coord_peak_gate:
            # peak of sigmoid heatmap per (B,K)
            peak = pred_hm.amax(dim=(-2, -1))  # (B,K)
            mask = (peak >= cfg.coord_peak_thr).float()  # (B,K)
            coord_mask_frac = mask.mean()

            # apply mask to SmoothL1: (B,K,2)
            diff = F.smooth_l1_loss(
                coords_hm_pred_n, coords_hm_gt_n,
                beta=cfg.coord_smoothl1_beta,
                reduction="none",
            )  # (B,K,2)

            diff = diff.mean(dim=-1)  # (B,K)
            # avoid all-zero mask -> keep stable
            denom = mask.sum().clamp_min(1.0)
            loss_coord = (diff * mask).sum() / denom
        else:
            loss_coord = F.smooth_l1_loss(
                coords_hm_pred_n, coords_hm_gt_n,
                beta=cfg.coord_smoothl1_beta,
                reduction="mean",
            )
            coord_mask_frac = torch.tensor(
                1.0, device=hmaps_pred_logits.device, dtype=hmaps_pred_logits.dtype)

    loss_coord_weighted = loss_coord * float(lambda_eff)
    loss_total = loss_hm + loss_coord_weighted
    return loss_total, loss_hm, loss_coord, loss_coord_weighted, coord_mask_frac


def get_layer_params(model: nn.Module, layer_name: str) -> Optional[List[torch.Tensor]]:
    """
    Returns a list of parameters for a submodule with given name, if found.
    """
    if not hasattr(model, layer_name):
        return None
    m = getattr(model, layer_name)
    params = [p for p in m.parameters() if p.requires_grad]
    return params if params else None


def grad_norm(loss: torch.Tensor, params: List[torch.Tensor]) -> float:
    """
    Compute L2 norm of gradients of loss wrt params (without stepping).
    This is expensive -> use sparsely.
    """
    grads = torch.autograd.grad(
        loss, params, retain_graph=True, create_graph=False, allow_unused=True)
    s = 0.0
    for g in grads:
        if g is None:
            continue
        s += float(torch.sum(g.detach() ** 2).item())
    return float(np.sqrt(max(s, 0.0)))


# -----------------------------------------------------------------------------
# Epoch loops
# -----------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scaler, device, cfg: TrainCfg, epoch: int):
    model.train()
    tot = hm = coord = coord_w = maskf = 0.0
    n = 0

    lambda_eff = lambda_schedule(epoch, cfg)
    layer_params = None
    if cfg.log_grad_norm_every > 0:
        layer_params = get_layer_params(model, cfg.grad_layer_name)
        if layer_params is None:
            print(
                f"[WARN] grad layer '{cfg.grad_layer_name}' not found or has no params -> disabling grad logging.")
            layer_params = None

    pbar = tqdm(loader, desc=f"train (λ={lambda_eff:.4f})", leave=False)
    for step, batch in enumerate(pbar, start=1):
        imgs = batch["image"].to(device, non_blocking=True)
        hmaps_gt = batch["heatmaps"].to(device, non_blocking=True)
        kpts_2d_gt = batch["kpts_2d"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            hmaps_pred = model(imgs)
            if hmaps_pred.shape[2:] != hmaps_gt.shape[2:]:
                hmaps_pred = F.interpolate(
                    hmaps_pred, size=hmaps_gt.shape[2:], mode="bilinear", align_corners=False)

            loss_total, loss_hm, loss_coord, loss_coord_w, coord_mask_frac = compute_losses(
                hmaps_pred, hmaps_gt, kpts_2d_gt, cfg, lambda_eff=lambda_eff
            )

        # optional gradient diagnostics (very sparse)
        grad_ratio = None
        if layer_params is not None and cfg.log_grad_norm_every > 0 and (step % cfg.log_grad_norm_every == 0):
            with torch.cuda.amp.autocast(enabled=False):
                # recompute in fp32 for stable grad norms
                hmaps_pred_fp = hmaps_pred.float()
                hmaps_gt_fp = hmaps_gt.float()
                kpts_fp = kpts_2d_gt.float()
                lt, lhm, lcoord, lcoord_w2, _mf = compute_losses(
                    hmaps_pred_fp, hmaps_gt_fp, kpts_fp, cfg, lambda_eff=lambda_eff
                )
                g_hm = grad_norm(lhm, layer_params)
                # weighted contribution
                g_coord = grad_norm(lcoord_w2, layer_params)
                grad_ratio = (g_coord / (g_hm + 1e-12))

        if device.type == "cuda":
            scaler.scale(loss_total).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_total.backward()
            optimizer.step()

        bs = imgs.size(0)
        tot += float(loss_total.item()) * bs
        hm += float(loss_hm.item()) * bs
        coord += float(loss_coord.item()) * bs
        coord_w += float(loss_coord_w.item()) * bs
        maskf += float(coord_mask_frac.item()) * bs
        n += bs

        postfix = dict(
            t=float(loss_total.item()),
            hm=float(loss_hm.item()),
            c=float(loss_coord.item()),
            lc=float(loss_coord_w.item()),
            mf=float(coord_mask_frac.item()),
        )
        if grad_ratio is not None:
            postfix["g(c)/g(hm)"] = float(grad_ratio)
        pbar.set_postfix(postfix)

    denom = max(n, 1)
    return (tot / denom, hm / denom, coord / denom, coord_w / denom, maskf / denom, lambda_eff)


@torch.no_grad()
def val_one_epoch(model, loader, device, cfg: TrainCfg, epoch: int):
    model.eval()
    tot = hm = coord = coord_w = maskf = 0.0
    n = 0

    lambda_eff = lambda_schedule(epoch, cfg)

    for batch in tqdm(loader, desc=f"val (λ={lambda_eff:.4f})", leave=False):
        imgs = batch["image"].to(device, non_blocking=True)
        hmaps_gt = batch["heatmaps"].to(device, non_blocking=True)
        kpts_2d_gt = batch["kpts_2d"].to(device, non_blocking=True)

        hmaps_pred = model(imgs)
        if hmaps_pred.shape[2:] != hmaps_gt.shape[2:]:
            hmaps_pred = F.interpolate(
                hmaps_pred, size=hmaps_gt.shape[2:], mode="bilinear", align_corners=False)

        loss_total, loss_hm, loss_coord, loss_coord_w, coord_mask_frac = compute_losses(
            hmaps_pred, hmaps_gt, kpts_2d_gt, cfg, lambda_eff=lambda_eff
        )

        bs = imgs.size(0)
        tot += float(loss_total.item()) * bs
        hm += float(loss_hm.item()) * bs
        coord += float(loss_coord.item()) * bs
        coord_w += float(loss_coord_w.item()) * bs
        maskf += float(coord_mask_frac.item()) * bs
        n += bs

    denom = max(n, 1)
    return (tot / denom, hm / denom, coord / denom, coord_w / denom, maskf / denom, lambda_eff)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    assert_dataset_layout()
    cfg, args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)
    if device.type == "cuda":
        print("[INFO] GPU:", torch.cuda.get_device_name(0))

    kpts_3d = np.load(KEYPOINTS_3D_NPY)
    num_kpts = int(kpts_3d.shape[0])
    print("[INFO] num_kpts:", num_kpts)
    print("[INFO] epochs:", cfg.epochs)

    if args.resume_run_dir:
        run_dir = Path(args.resume_run_dir)
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (run_dir / "plots").mkdir(parents=True, exist_ok=True)
        (run_dir / "debug_heatmaps").mkdir(parents=True, exist_ok=True)
    else:
        run_dir = make_run_dir(Path(cfg.runs_dir), cfg.run_tag)

    ckpt_dir = run_dir / "checkpoints"
    plots_dir = run_dir / "plots"
    csv_path = run_dir / "loss_history.csv"

    (run_dir / "TRAIN_SCRIPT_SIGNATURE.txt").write_text(SCRIPT_SIGNATURE +
                                                        "\n", encoding="utf-8")

    env_info = {
        "script_signature": SCRIPT_SIGNATURE,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
    }
    save_json(run_dir / "run_config.json", {**asdict(cfg), **env_info})
    print("[INFO] run_dir:", run_dir.resolve())

    train_ds = SpeedPlusKeypointDataset(
        split="train",
        domain=cfg.domain,
        img_size=cfg.img_size,
        heatmap_size=cfg.heatmap_size,
        sigma=cfg.sigma,
        return_heatmaps=True,
    )
    val_ds = SpeedPlusKeypointDataset(
        split="val",
        domain=cfg.domain,
        img_size=cfg.img_size,
        heatmap_size=cfg.heatmap_size,
        sigma=cfg.sigma,
        return_heatmaps=True,
    )

    pin = (device.type == "cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        persistent_workers=(cfg.num_workers > 0),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        persistent_workers=(cfg.num_workers > 0),
        drop_last=False,
    )

    model = HRNetKeypointModel(num_keypoints=num_kpts).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    start_epoch = 1
    best_val = float("inf")

    # resume
    if args.resume_ckpt:
        ckpt_path = Path(args.resume_ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"resume_ckpt not found: {ckpt_path}")

        ckpt = torch.load(str(ckpt_path), map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=True)
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scaler_state" in ckpt and device.type == "cuda":
            scaler.load_state_dict(ckpt["scaler_state"])

        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val", best_val))
        print(
            f"[RESUME] ckpt={ckpt_path} -> start_epoch={start_epoch}, best_val={best_val}")

    header = [
        "epoch",
        "lambda_eff",
        "train_total", "train_hm", "train_coord", "train_lambda_coord", "train_mask_frac",
        "val_total", "val_hm", "val_coord", "val_lambda_coord", "val_mask_frac",
        "epoch_time_s",
        "use_coord_loss",
        "lambda_coord_max", "lambda_warmup_epochs", "lambda_ramp_epochs",
        "beta_softarg",
        "coord_smoothl1_beta", "use_coord_peak_gate", "coord_peak_thr",
        "hm_loss_type", "pos_weight", "thr",
        "debug_hm_every", "debug_hm_n", "debug_hm_k", "debug_hm_seed",
    ]

    hist = {
        "train_total": [], "val_total": [],
        "train_hm": [], "val_hm": [],
        "train_coord": [], "val_coord": [],
        # weighted coord contribution (lambda*coord)
        "train_lc": [], "val_lc": [],
        "train_mf": [], "val_mf": [],  # mask fraction
        "lambda_eff": [],
    }

    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.time()

        tr_tot, tr_hm, tr_coord, tr_lc, tr_mf, lam = train_one_epoch(
            model, train_loader, optimizer, scaler, device, cfg, epoch
        )
        va_tot, va_hm, va_coord, va_lc, va_mf, lam_v = val_one_epoch(
            model, val_loader, device, cfg, epoch
        )

        dt = time.time() - t0

        hist["train_total"].append(tr_tot)
        hist["val_total"].append(va_tot)
        hist["train_hm"].append(tr_hm)
        hist["val_hm"].append(va_hm)
        hist["train_coord"].append(tr_coord)
        hist["val_coord"].append(va_coord)
        hist["train_lc"].append(tr_lc)
        hist["val_lc"].append(va_lc)
        hist["train_mf"].append(tr_mf)
        hist["val_mf"].append(va_mf)
        hist["lambda_eff"].append(lam)

        print(
            f"[Epoch {epoch:03d}/{cfg.epochs}] "
            f"λ={lam:.4f} | "
            f"train tot={tr_tot:.6f} hm={tr_hm:.6f} coord={tr_coord:.6f} (λ*coord={tr_lc:.6f}, mask={tr_mf:.3f}) | "
            f"val tot={va_tot:.6f} hm={va_hm:.6f} coord={va_coord:.6f} (λ*coord={va_lc:.6f}, mask={va_mf:.3f}) | "
            f"time={dt:.1f}s"
        )

        append_csv_row(
            csv_path,
            header=header,
            row=[
                epoch,
                lam,
                tr_tot, tr_hm, tr_coord, tr_lc, tr_mf,
                va_tot, va_hm, va_coord, va_lc, va_mf,
                dt,
                int(cfg.use_coord_loss),
                cfg.lambda_coord_max, cfg.lambda_warmup_epochs, cfg.lambda_ramp_epochs,
                cfg.beta_softarg,
                cfg.coord_smoothl1_beta, int(
                    cfg.use_coord_peak_gate), cfg.coord_peak_thr,
                cfg.hm_loss_type, cfg.pos_weight, cfg.thr,
                cfg.debug_hm_every, cfg.debug_hm_n, cfg.debug_hm_k, cfg.debug_hm_seed,
            ],
        )

        # checkpoints
        if (epoch % max(cfg.save_every_epochs, 1)) == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scaler_state": (scaler.state_dict() if device.type == "cuda" else None),
                    "cfg": asdict(cfg),

                    "train_loss_total": tr_tot,
                    "train_loss_hm": tr_hm,
                    "train_loss_coord": tr_coord,
                    "train_loss_lambda_coord": tr_lc,
                    "train_coord_mask_frac": tr_mf,

                    "val_loss_total": va_tot,
                    "val_loss_hm": va_hm,
                    "val_loss_coord": va_coord,
                    "val_loss_lambda_coord": va_lc,
                    "val_coord_mask_frac": va_mf,

                    "lambda_eff": lam,
                    "run_dir": str(run_dir),
                    "script_signature": SCRIPT_SIGNATURE,
                },
                ckpt_dir / "hrnet_kpts_last.pth",
            )

        if va_tot < best_val:
            best_val = va_tot
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scaler_state": (scaler.state_dict() if device.type == "cuda" else None),
                    "cfg": asdict(cfg),

                    "train_loss_total": tr_tot,
                    "train_loss_hm": tr_hm,
                    "train_loss_coord": tr_coord,
                    "train_loss_lambda_coord": tr_lc,
                    "train_coord_mask_frac": tr_mf,

                    "val_loss_total": va_tot,
                    "val_loss_hm": va_hm,
                    "val_loss_coord": va_coord,
                    "val_loss_lambda_coord": va_lc,
                    "val_coord_mask_frac": va_mf,

                    "lambda_eff": lam,
                    "run_dir": str(run_dir),
                    "script_signature": SCRIPT_SIGNATURE,
                    "best_val_total": best_val,
                },
                ckpt_dir / "hrnet_kpts_best.pth",
            )
            print(f"  -> saved BEST (val_tot={best_val:.6f})")

        # plots
        epochs_axis = list(range(1, len(hist["train_total"]) + 1))
        plot_loss(plots_dir / "loss_total.png", epochs_axis,
                  hist["train_total"], hist["val_total"], "Total loss")
        plot_loss(plots_dir / "loss_hm.png", epochs_axis,
                  hist["train_hm"], hist["val_hm"], "Heatmap loss")
        if cfg.use_coord_loss:
            plot_loss(plots_dir / "loss_coord.png", epochs_axis,
                      hist["train_coord"], hist["val_coord"], "Coord loss (unweighted)")
            plot_loss(plots_dir / "loss_lambda_coord.png", epochs_axis,
                      hist["train_lc"], hist["val_lc"], "Lambda*Coord contribution")

        # log-scale summary
        plot_all_losses_log(
            plots_dir / "loss_all_log.png",
            epochs_axis,
            {
                "train_total": hist["train_total"],
                "val_total": hist["val_total"],
                "train_hm": hist["train_hm"],
                "val_hm": hist["val_hm"],
                "train_coord": hist["train_coord"],
                "val_coord": hist["val_coord"],
                "train_lambda_coord": hist["train_lc"],
                "val_lambda_coord": hist["val_lc"],
            },
        )

        # debug heatmaps dump
        dump_debug_heatmaps(
            model=model,
            val_ds=val_ds,
            device=device,
            run_dir=run_dir,
            epoch=epoch,
            cfg=cfg,
            num_kpts=num_kpts,
        )

    print("[DONE] Training completed.")
    print("[DONE] run_dir:", run_dir.resolve())
    print("[DONE] best_val_total:", best_val)


if __name__ == "__main__":
    main()
