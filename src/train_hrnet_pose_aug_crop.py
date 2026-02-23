# src/train_hrnet_pose_aug.py
from __future__ import annotations
import inspect
import albumentations as A

from src.config import KEYPOINTS_3D_NPY, assert_dataset_layout
from src.dataset_speedplus import SpeedPlusKeypointDataset
from src.model_hrnet_pose import HRNetKeypointModel
from src.softargmax import soft_argmax_2d
from tqdm import tqdm
from torch.utils.data import DataLoader

import torch.optim as optim
import torch.nn as nn
import torch
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
    img_size: Tuple[int, int] = (512, 512)
    heatmap_size: Tuple[int, int] = (128, 128)
    sigma: float = 2.0
    domain: str = "synthetic"

    # training
    epochs: int = 100
    batch_size: int = 8
    lr: float = 1e-4
    num_workers: int = 4

    # losses (your normalized coord-loss setup)
    use_coord_loss: bool = True
    lambda_coord: float = 1.0
    beta_softarg: float = 30.0

    hm_loss_type: str = "weighted_mse"  # ["weighted_mse", "mse"]
    pos_weight: float = 20.0
    thr: float = 0.2

    # bbox crop (pre-processing)
    use_bbox_crop: bool = False
    bbox_margin: float = 0.15
    bbox_square: bool = True
    bbox_min_size: int = 64

    # augmentations (domain shift)
    use_aug: bool = True
    aug_preset: str = "strong"  # light/medium/strong
    aug_seed: int = 0
    p_geom: float = 0.30
    p_photo: float = 0.95
    p_noise: float = 0.85
    p_occl: float = 0.30
    p_flare: float = 0.20

    # runs/logging
    runs_dir: str = "runs"
    run_tag: str = "hrnet_augDomain_weightedMSE_pw20"
    save_every_epochs: int = 1

    # debug heatmaps dump
    debug_hm_every: int = 1  # 0 disables
    debug_hm_n: int = 6
    debug_hm_k: str = "0,3,8"
    debug_hm_seed: int = 0


def parse_args() -> TrainCfg:
    ap = argparse.ArgumentParser()

    # training
    ap.add_argument("--epochs", type=int, default=TrainCfg.epochs)
    ap.add_argument("--batch_size", type=int, default=TrainCfg.batch_size)
    ap.add_argument("--lr", type=float, default=TrainCfg.lr)
    ap.add_argument("--num_workers", type=int, default=TrainCfg.num_workers)

    # losses
    ap.add_argument("--use_coord_loss", type=int, default=1,
                    help="1=use coord loss, 0=HM-only")
    ap.add_argument("--lambda_coord", type=float,
                    default=TrainCfg.lambda_coord)
    ap.add_argument("--beta_softarg", type=float,
                    default=TrainCfg.beta_softarg)
    ap.add_argument("--hm_loss_type", type=str,
                    default=TrainCfg.hm_loss_type, choices=["weighted_mse", "mse"])
    ap.add_argument("--pos_weight", type=float, default=TrainCfg.pos_weight)
    ap.add_argument("--thr", type=float, default=TrainCfg.thr)

    # bbox crop
    ap.add_argument("--use_bbox_crop", type=int, default=0,
                    help="1=enable crop on GT keypoints bbox before aug")
    ap.add_argument("--bbox_margin", type=float, default=TrainCfg.bbox_margin)
    ap.add_argument("--bbox_square", type=int, default=1)
    ap.add_argument("--bbox_min_size", type=int,
                    default=TrainCfg.bbox_min_size)

    # augment
    ap.add_argument("--use_aug", type=int, default=1,
                    help="1=enable albumentations on train, 0=off")
    ap.add_argument("--aug_preset", type=str, default="strong",
                    choices=["light", "medium", "strong"])
    ap.add_argument("--aug_seed", type=int, default=0)
    ap.add_argument("--p_geom", type=float, default=TrainCfg.p_geom)
    ap.add_argument("--p_photo", type=float, default=TrainCfg.p_photo)
    ap.add_argument("--p_noise", type=float, default=TrainCfg.p_noise)
    ap.add_argument("--p_occl", type=float, default=TrainCfg.p_occl)
    ap.add_argument("--p_flare", type=float, default=TrainCfg.p_flare)

    # runs
    ap.add_argument("--runs_dir", type=str, default=TrainCfg.runs_dir)
    ap.add_argument("--run_tag", type=str, default=TrainCfg.run_tag)
    ap.add_argument("--save_every_epochs", type=int,
                    default=TrainCfg.save_every_epochs)

    # debug heatmaps
    ap.add_argument("--debug_hm_every", type=int, default=TrainCfg.debug_hm_every,
                    help="save debug heatmaps every N epochs (0 disables)")
    ap.add_argument("--debug_hm_n", type=int, default=TrainCfg.debug_hm_n,
                    help="how many val images to dump")
    ap.add_argument("--debug_hm_k", type=str, default=TrainCfg.debug_hm_k,
                    help="comma-separated keypoint ids to dump")
    ap.add_argument("--debug_hm_seed", type=int, default=TrainCfg.debug_hm_seed,
                    help="seed for selecting debug samples")

    # checkpoint
    ap.add_argument("--resume_ckpt", type=str, default="",
                    help="path to hrnet_kpts_last.pth to resume (optional)")
    ap.add_argument("--resume_run_dir", type=str, default="",
                    help="existing run_dir to continue (optional)")

    args = ap.parse_args()
    cfg = TrainCfg(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,

        use_coord_loss=bool(args.use_coord_loss),
        lambda_coord=args.lambda_coord,
        beta_softarg=args.beta_softarg,
        hm_loss_type=args.hm_loss_type,
        pos_weight=args.pos_weight,
        thr=args.thr,

        use_bbox_crop=bool(args.use_bbox_crop),
        bbox_margin=float(args.bbox_margin),
        bbox_square=bool(args.bbox_square),
        bbox_min_size=int(args.bbox_min_size),

        use_aug=bool(args.use_aug),
        aug_preset=args.aug_preset,
        aug_seed=args.aug_seed,
        p_geom=args.p_geom,
        p_photo=args.p_photo,
        p_noise=args.p_noise,
        p_occl=args.p_occl,
        p_flare=args.p_flare,

        runs_dir=args.runs_dir,
        run_tag=args.run_tag,
        save_every_epochs=args.save_every_epochs,

        debug_hm_every=args.debug_hm_every,
        debug_hm_n=args.debug_hm_n,
        debug_hm_k=args.debug_hm_k,
        debug_hm_seed=args.debug_hm_seed,
    )
    return cfg, args


# -----------------------------------------------------------------------------
# Run utils
# -----------------------------------------------------------------------------
SCRIPT_SIGNATURE = "train_hrnet_pose_aug.py :: v1 :: domain-shift albumentations + normalized coord loss + bbox crop"


def kpts_bbox_xyxy(
    kpts: np.ndarray,
    img_wh: Tuple[int, int],
    margin: float = 0.10,
    make_square: bool = True,
    min_size_px: int = 32,
):
    W, H = img_wh
    k = np.asarray(kpts, dtype=np.float32).reshape(-1, 2)

    m = (k[:, 0] >= 0) & (k[:, 0] < W) & (k[:, 1] >= 0) & (k[:, 1] < H)
    if not np.any(m):
        return None

    xs = k[m, 0]
    ys = k[m, 1]
    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())

    bw = max(x2 - x1, 1e-6)
    bh = max(y2 - y1, 1e-6)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    bw *= (1.0 + margin)
    bh *= (1.0 + margin)

    if make_square:
        s = max(bw, bh)
        bw = bh = s

    bw = max(bw, float(min_size_px))
    bh = max(bh, float(min_size_px))

    x1 = cx - 0.5 * bw
    x2 = cx + 0.5 * bw
    y1 = cy - 0.5 * bh
    y2 = cy + 0.5 * bh

    x1 = int(np.floor(max(0.0, x1)))
    y1 = int(np.floor(max(0.0, y1)))
    x2 = int(np.ceil(min(float(W), x2)))
    y2 = int(np.ceil(min(float(H), y2)))

    if (x2 - x1) < 2 or (y2 - y1) < 2:
        return None
    return (x1, y1, x2, y2)


def crop_and_resize(
    img_rgb_u8: np.ndarray,
    kpts: np.ndarray,
    bbox_xyxy: Tuple[int, int, int, int],
    out_wh: Tuple[int, int],
):
    x1, y1, x2, y2 = bbox_xyxy
    crop = img_rgb_u8[y1:y2, x1:x2, :]

    ch, cw = crop.shape[:2]
    Wout, Hout = out_wh

    sx = Wout / float(max(cw, 1))
    sy = Hout / float(max(ch, 1))

    k = kpts.copy().astype(np.float32)
    k[:, 0] = (k[:, 0] - x1) * sx
    k[:, 1] = (k[:, 1] - y1) * sy

    crop_res = cv2.resize(crop, (Wout, Hout), interpolation=cv2.INTER_LINEAR)
    return crop_res, k


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
# Keypoints normalization helper (your version)
# -----------------------------------------------------------------------------
def kpts_img_to_hm_normalized(
    kpts_img: torch.Tensor,
    img_size: Tuple[int, int],
    heatmap_size: Tuple[int, int],
):
    """
    Convert GT keypoints from image space (px) to normalized heatmap space [0,1].
    kpts_img: (B,K,2) in image pixels
    returns: (B,K,2) in [0,1]
    """
    W_img, H_img = img_size
    W_hm, H_hm = heatmap_size

    x_hm = kpts_img[..., 0] * (W_hm - 1) / (W_img - 1)
    y_hm = kpts_img[..., 1] * (H_hm - 1) / (H_img - 1)

    x_n = x_hm / (W_hm - 1)
    y_n = y_hm / (H_hm - 1)
    return torch.stack([x_n, y_n], dim=-1)


# -----------------------------------------------------------------------------
# Albumentations (domain shift)
# -----------------------------------------------------------------------------
def safe_aug(name: str, ctor, **kwargs):
    """Create augmentation ignoring unsupported kwargs; returns None on failure."""
    try:
        sig = inspect.signature(ctor)
        filt = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return ctor(**filt)
    except Exception as e:
        print(f"[SKIP] {name}: {e}")
        return None


def build_aug_pipeline(
    preset: str,
    seed: int,
    p_geom: float,
    p_photo: float,
    p_noise: float,
    p_occl: float,
    p_flare: float,
) -> A.Compose:
    """
    Augmentations tuned for lightbox/sunlamp:
      - heavy photometric: gamma, equalize/CLAHE, tone curve, exposure, color shifts
      - sensor artifacts: noise, blur, jpeg
      - occasional flare (sunlamp)
      - mild geometry jitter
      - mild occlusions (dropout)
    """
    preset = preset.lower().strip()

    # Geometry (mild)
    geom_light = A.OneOf(
        [
            A.Affine(
                translate_percent=(-0.04, 0.04),
                scale=(0.92, 1.08),
                rotate=(-8, 8),
                shear=(-4, 4),
                p=1.0,
            ),
            A.Perspective(scale=(0.02, 0.05), keep_size=True, p=1.0),
        ],
        p=1.0,
    )
    geom_min = A.Affine(
        translate_percent=(-0.02, 0.02),
        scale=(0.96, 1.04),
        rotate=(-5, 5),
        shear=(-2, 2),
        p=1.0,
    )

    # Photometric core
    photo_core = A.OneOf(
        [
            A.RandomBrightnessContrast(
                brightness_limit=0.6, contrast_limit=0.6, p=1.0),
            A.RandomGamma(gamma_limit=(70, 160), p=1.0),
            A.CLAHE(clip_limit=(1.0, 4.0), tile_grid_size=(8, 8), p=1.0),
            A.Equalize(mode="cv", by_channels=True, p=1.0),
            A.RandomToneCurve(scale=0.6, p=1.0),
        ],
        p=1.0,
    )

    photo_color = A.OneOf(
        [
            A.ColorJitter(brightness=0.25, contrast=0.30,
                          saturation=0.20, hue=0.06, p=1.0),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=1.0),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15,
                       b_shift_limit=15, p=1.0),
        ],
        p=1.0,
    )

    # Noise / compression / blur
    noise_blk = A.OneOf(
        [
            A.GaussNoise(std_range=(0.02, 0.12), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05),
                       intensity=(0.10, 0.55), p=1.0),
            A.ImageCompression(quality_range=(25, 85), p=1.0),
        ],
        p=1.0,
    )

    blur_blk = A.OneOf(
        [
            A.GaussianBlur(blur_limit=(3, 9), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
            A.Sharpen(alpha=(0.15, 0.45), lightness=(0.9, 1.6), p=1.0),
        ],
        p=1.0,
    )

    sun_flare = safe_aug(
        "RandomSunFlare",
        A.RandomSunFlare,
        flare_roi=(0.0, 0.0, 1.0, 0.6),
        angle_lower=0.0,
        angle_upper=1.0,
        num_flare_circles_lower=6,
        num_flare_circles_upper=10,
        src_radius=90,
        p=1.0,
    )

    occl_blk = A.CoarseDropout(
        num_holes_range=(3, 9),
        hole_height_range=(14, 90),
        hole_width_range=(14, 90),
        fill=0,
        p=1.0,
    )

    styleish = A.OneOf(
        [
            A.ToGray(p=1.0),
            A.Posterize(num_bits=(4, 6), p=1.0),
        ],
        p=1.0,
    )

    # Presets
    if preset == "light":
        use_geom = geom_min
        p_geom = min(p_geom, 0.20)
        p_photo = min(p_photo, 0.75)
        p_noise = min(p_noise, 0.55)
        p_occl = min(p_occl, 0.20)
        p_flare = min(p_flare, 0.10)
        p_style = 0.10
    elif preset == "medium":
        use_geom = geom_light
        p_style = 0.15
    else:
        use_geom = geom_light
        p_style = 0.20

    blocks = [
        A.OneOf([use_geom], p=float(p_geom)),
        A.OneOf([photo_core], p=float(p_photo)),
        A.OneOf([photo_color], p=float(min(0.55, p_photo))),
        A.OneOf([noise_blk], p=float(p_noise)),
        A.OneOf([blur_blk], p=float(min(0.75, p_noise))),
        A.OneOf([occl_blk], p=float(p_occl)),
        A.OneOf([styleish], p=float(p_style)),
    ]
    if sun_flare is not None:
        blocks.insert(3, A.OneOf([sun_flare], p=float(p_flare)))

    kp_params = A.KeypointParams(format="xy", remove_invisible=False)
    return A.Compose(blocks, keypoint_params=kp_params)


class AugmentedSpeedPlusDataset(SpeedPlusKeypointDataset):
    def __init__(
        self,
        *args,
        aug: Optional[A.Compose] = None,
        use_bbox_crop: bool = False,
        bbox_margin: float = 0.15,
        bbox_square: bool = True,
        bbox_min_size: int = 64,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.aug = aug
        self.use_bbox_crop = bool(use_bbox_crop)
        self.bbox_margin = float(bbox_margin)
        self.bbox_square = bool(bbox_square)
        self.bbox_min_size = int(bbox_min_size)

    def __getitem__(self, idx: int) -> Dict:
        out = super().__getitem__(idx)

        # Keypoints in image pixel space (K,2)
        kpts = out["kpts_2d"].numpy().astype(np.float32)

        # Image tensor CHW [0,1] -> uint8 HWC RGB
        img = out["image"].numpy().transpose(1, 2, 0)
        img_u8 = (np.clip(img, 0, 1) * 255.0).astype(np.uint8)

        W, H = img_u8.shape[1], img_u8.shape[0]

        # 1) BBOX CROP (deterministico su GT)
        if self.use_bbox_crop:
            bbox = kpts_bbox_xyxy(
                kpts,
                img_wh=(W, H),
                margin=self.bbox_margin,
                make_square=self.bbox_square,
                min_size_px=self.bbox_min_size,
            )
            if bbox is not None:
                img_u8, kpts = crop_and_resize(
                    img_u8, kpts, bbox, out_wh=(W, H)
                )

        # 2) ALBUMENTATIONS (se attivo)
        if self.aug is not None:
            res = self.aug(image=img_u8, keypoints=[tuple(xy) for xy in kpts])
            img_u8 = res["image"]
            kpts = np.array(res["keypoints"], dtype=np.float32)

        # Back to tensor CHW [0,1]
        x = img_u8.astype(np.float32) / 255.0
        out["image"] = torch.from_numpy(x).permute(2, 0, 1).contiguous()
        out["kpts_2d"] = torch.from_numpy(kpts)

        # Rebuild GT heatmaps from final kpts
        if self.return_heatmaps:
            out["heatmaps"] = self._make_heatmaps(kpts)

        return out


# -----------------------------------------------------------------------------
# Heatmap debug dump (unchanged)
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
def dump_debug_heatmaps(model: torch.nn.Module, val_ds: SpeedPlusKeypointDataset, device: torch.device,
                        run_dir: Path, epoch: int, cfg: TrainCfg, num_kpts: int):
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
        pred_logits = model(x)[0].detach().cpu().numpy().astype(np.float32)
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
# Losses (your normalized coord loss)
# -----------------------------------------------------------------------------
def weighted_mse_loss(pred: torch.Tensor, target: torch.Tensor, pos_weight: float, thr: float):
    w = torch.ones_like(target)
    w = w + (pos_weight - 1.0) * (target >= thr).float()
    return ((pred - target) ** 2 * w).mean()


def compute_losses(hmaps_pred_logits: torch.Tensor, hmaps_gt: torch.Tensor, kpts_2d_gt: torch.Tensor, cfg: TrainCfg):
    pred_hm = torch.sigmoid(hmaps_pred_logits)

    if cfg.hm_loss_type == "mse":
        loss_hm = torch.mean((pred_hm - hmaps_gt) ** 2)
    elif cfg.hm_loss_type == "weighted_mse":
        loss_hm = weighted_mse_loss(
            pred_hm, hmaps_gt, pos_weight=cfg.pos_weight, thr=cfg.thr)
    else:
        raise ValueError(cfg.hm_loss_type)

    loss_coord = torch.zeros(
        (), device=hmaps_pred_logits.device, dtype=hmaps_pred_logits.dtype)

    if cfg.use_coord_loss:
        # Pred coords in HEATMAP space (pixels)
        coords_hm_pred, _conf = soft_argmax_2d(
            hmaps_pred_logits, beta=cfg.beta_softarg)  # (B,K,2)

        # Normalize pred to [0,1]
        W_hm, H_hm = hmaps_pred_logits.shape[-1], hmaps_pred_logits.shape[-2]
        coords_hm_pred_n = torch.stack(
            [coords_hm_pred[..., 0] / (W_hm - 1),
             coords_hm_pred[..., 1] / (H_hm - 1)],
            dim=-1
        )

        # GT coords: image px -> normalized HM
        coords_hm_gt_n = kpts_img_to_hm_normalized(
            kpts_2d_gt,
            img_size=cfg.img_size,
            heatmap_size=(W_hm, H_hm),
        )

        # Smooth L1 in normalized space
        loss_coord = nn.SmoothL1Loss(beta=0.02)(
            coords_hm_pred_n, coords_hm_gt_n)

        loss_total = loss_hm + cfg.lambda_coord * loss_coord
    else:
        loss_total = loss_hm

    return loss_total, loss_hm, loss_coord


# -----------------------------------------------------------------------------
# Epoch loops
# -----------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scaler, device, cfg: TrainCfg):
    model.train()
    tot = hm = coord = 0.0
    n = 0

    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        imgs = batch["image"].to(device, non_blocking=True)
        hmaps_gt = batch["heatmaps"].to(device, non_blocking=True)
        kpts_2d_gt = batch["kpts_2d"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            hmaps_pred = model(imgs)
            if hmaps_pred.shape[2:] != hmaps_gt.shape[2:]:
                hmaps_pred = nn.functional.interpolate(
                    hmaps_pred, size=hmaps_gt.shape[2:], mode="bilinear", align_corners=False
                )
            loss_total, loss_hm, loss_coord = compute_losses(
                hmaps_pred, hmaps_gt, kpts_2d_gt, cfg)

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
        n += bs

        pbar.set_postfix(t=float(loss_total.item()), hm=float(
            loss_hm.item()), c=float(loss_coord.item()))

    denom = max(n, 1)
    return tot / denom, hm / denom, coord / denom


@torch.no_grad()
def val_one_epoch(model, loader, device, cfg: TrainCfg):
    model.eval()
    tot = hm = coord = 0.0
    n = 0

    for batch in tqdm(loader, desc="val", leave=False):
        imgs = batch["image"].to(device, non_blocking=True)
        hmaps_gt = batch["heatmaps"].to(device, non_blocking=True)
        kpts_2d_gt = batch["kpts_2d"].to(device, non_blocking=True)

        hmaps_pred = model(imgs)
        if hmaps_pred.shape[2:] != hmaps_gt.shape[2:]:
            hmaps_pred = nn.functional.interpolate(
                hmaps_pred, size=hmaps_gt.shape[2:], mode="bilinear", align_corners=False
            )
        loss_total, loss_hm, loss_coord = compute_losses(
            hmaps_pred, hmaps_gt, kpts_2d_gt, cfg)

        bs = imgs.size(0)
        tot += float(loss_total.item()) * bs
        hm += float(loss_hm.item()) * bs
        coord += float(loss_coord.item()) * bs
        n += bs

    denom = max(n, 1)
    return tot / denom, hm / denom, coord / denom


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

    # run dir
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

    # build augmentation
    aug = None
    if cfg.use_aug:
        aug = build_aug_pipeline(
            preset=cfg.aug_preset,
            seed=cfg.aug_seed,
            p_geom=cfg.p_geom,
            p_photo=cfg.p_photo,
            p_noise=cfg.p_noise,
            p_occl=cfg.p_occl,
            p_flare=cfg.p_flare,
        )
        print(f"[INFO] AUG enabled preset={cfg.aug_preset} seed={cfg.aug_seed} "
              f"p_geom={cfg.p_geom} p_photo={cfg.p_photo} p_noise={cfg.p_noise} "
              f"p_occl={cfg.p_occl} p_flare={cfg.p_flare}")
    else:
        print("[INFO] AUG disabled")

    # datasets
    train_ds = AugmentedSpeedPlusDataset(
        split="train",
        domain=cfg.domain,
        img_size=cfg.img_size,
        heatmap_size=cfg.heatmap_size,
        sigma=cfg.sigma,
        return_heatmaps=True,
        aug=aug,
        use_bbox_crop=cfg.use_bbox_crop,
        bbox_margin=cfg.bbox_margin,
        bbox_square=cfg.bbox_square,
        bbox_min_size=cfg.bbox_min_size,
    )
    val_ds = AugmentedSpeedPlusDataset(
        split="val",
        domain=cfg.domain,
        img_size=cfg.img_size,
        heatmap_size=cfg.heatmap_size,
        sigma=cfg.sigma,
        return_heatmaps=True,
        aug=None,
        use_bbox_crop=cfg.use_bbox_crop,
        bbox_margin=cfg.bbox_margin,
        bbox_square=cfg.bbox_square,
        bbox_min_size=cfg.bbox_min_size,
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
        "train_total", "train_hm", "train_coord",
        "val_total", "val_hm", "val_coord",
        "epoch_time_s",
        "use_aug", "aug_preset", "aug_seed", "p_geom", "p_photo", "p_noise", "p_occl", "p_flare",
        "use_coord_loss", "lambda_coord", "beta_softarg",
        "hm_loss_type", "pos_weight", "thr",
        "debug_hm_every", "debug_hm_n", "debug_hm_k", "debug_hm_seed",
    ]

    hist = {
        "train_total": [], "val_total": [],
        "train_hm": [], "val_hm": [],
        "train_coord": [], "val_coord": [],
    }

    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.time()

        tr_tot, tr_hm, tr_coord = train_one_epoch(
            model, train_loader, optimizer, scaler, device, cfg)
        va_tot, va_hm, va_coord = val_one_epoch(model, val_loader, device, cfg)

        dt = time.time() - t0

        hist["train_total"].append(tr_tot)
        hist["val_total"].append(va_tot)
        hist["train_hm"].append(tr_hm)
        hist["val_hm"].append(va_hm)
        hist["train_coord"].append(tr_coord)
        hist["val_coord"].append(va_coord)

        print(
            f"[Epoch {epoch:03d}/{cfg.epochs}] "
            f"train tot={tr_tot:.6f} hm={tr_hm:.6f} coord={tr_coord:.6f} | "
            f"val tot={va_tot:.6f} hm={va_hm:.6f} coord={va_coord:.6f} | "
            f"time={dt:.1f}s"
        )

        append_csv_row(
            csv_path,
            header=header,
            row=[
                epoch, tr_tot, tr_hm, tr_coord, va_tot, va_hm, va_coord, dt,
                int(cfg.use_aug), cfg.aug_preset, cfg.aug_seed, cfg.p_geom, cfg.p_photo, cfg.p_noise, cfg.p_occl, cfg.p_flare,
                int(cfg.use_coord_loss), cfg.lambda_coord, cfg.beta_softarg,
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
                    "scaler_state": scaler.state_dict() if device.type == "cuda" else None,
                    "cfg": asdict(cfg),
                    "train_loss_total": tr_tot,
                    "train_loss_hm": tr_hm,
                    "train_loss_coord": tr_coord,
                    "val_loss_total": va_tot,
                    "val_loss_hm": va_hm,
                    "val_loss_coord": va_coord,
                    "best_val": best_val,
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
                    "scaler_state": scaler.state_dict() if device.type == "cuda" else None,
                    "cfg": asdict(cfg),
                    "train_loss_total": tr_tot,
                    "train_loss_hm": tr_hm,
                    "train_loss_coord": tr_coord,
                    "val_loss_total": va_tot,
                    "val_loss_hm": va_hm,
                    "val_loss_coord": va_coord,
                    "best_val": best_val,
                    "run_dir": str(run_dir),
                    "script_signature": SCRIPT_SIGNATURE,
                },
                ckpt_dir / "hrnet_kpts_best.pth",
            )
            print(f"  -> saved BEST (val_tot={best_val:.6f})")

        # plots
        epochs_axis = list(range(1, epoch + 1))
        plot_loss(plots_dir / "loss_total.png", epochs_axis,
                  hist["train_total"], hist["val_total"], "Total loss")
        plot_loss(plots_dir / "loss_hm.png", epochs_axis,
                  hist["train_hm"], hist["val_hm"], "Heatmap loss")
        if cfg.use_coord_loss:
            plot_loss(plots_dir / "loss_coord.png", epochs_axis,
                      hist["train_coord"], hist["val_coord"], "Coord loss")
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
            },
        )

        # debug heatmaps dump (clean val!)
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
