# src/eval_pnp_crop_dynamic.py
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm
from scipy.spatial.transform import Rotation as SciRot

from src.softargmax import soft_argmax_2d, hm_to_image_coords
from src.config import KEYPOINTS_3D_NPY, load_camera_matrix, scale_intrinsics
from src.dataset_speedplus import SpeedPlusKeypointDataset
from src.model_hrnet_pose import HRNetKeypointModel


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(p: str) -> Path:
    pth = Path(p).expanduser()
    if pth.is_absolute():
        return pth
    return (project_root() / pth).resolve()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_txt(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def is_finite_arr(x: np.ndarray) -> bool:
    x = np.asarray(x)
    return bool(np.all(np.isfinite(x)))


def summarize(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return dict(N=0, mean=None, median=None, p90=None)
    return dict(
        N=int(arr.size),
        mean=float(np.mean(arr)),
        median=float(np.median(arr)),
        p90=float(np.percentile(arr, 90)),
    )


def summarize_times_ms(times_ms: List[float]) -> Optional[dict]:
    t = np.asarray(times_ms, dtype=np.float64)
    if t.size == 0:
        return None
    mean_ms = float(t.mean())
    med_ms = float(np.median(t))
    p90_ms = float(np.percentile(t, 90))
    return dict(
        mean_ms=mean_ms,
        median_ms=med_ms,
        p90_ms=p90_ms,
        fps_mean=(1000.0 / mean_ms if mean_ms > 0 else None),
        fps_median=(1000.0 / med_ms if med_ms > 0 else None),
    )


def heatmaps_argmax_2d(hmaps: torch.Tensor) -> torch.Tensor:
    """
    hmaps: (B,K,H,W) logits or probs
    returns: (B,K,2) coords in heatmap pixel space (x,y) as float32
    """
    B, K, H, W = hmaps.shape
    flat = hmaps.view(B, K, -1)
    idx = torch.argmax(flat, dim=-1)  # (B,K)
    y = (idx // W).to(torch.float32)
    x = (idx % W).to(torch.float32)
    return torch.stack([x, y], dim=-1)


def reprojection_points(kpts_3d: np.ndarray, K: np.ndarray, R_CT: np.ndarray, t_C: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(R_CT.astype(np.float64))
    tvec = t_C.reshape(3, 1).astype(np.float64)
    proj, _ = cv2.projectPoints(
        kpts_3d.astype(np.float64),
        rvec,
        tvec,
        K.astype(np.float64),
        None
    )
    return proj.reshape(-1, 2).astype(np.float32)


def reprojection_rmse_px(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.float32) - b.astype(np.float32)
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


def draw_points(img_bgr: np.ndarray, pts: np.ndarray, color, prefix: str = "") -> np.ndarray:
    out = img_bgr.copy()
    for i, (x, y) in enumerate(pts):
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        cv2.circle(out, (int(round(x)), int(round(y))),
                   3, color, -1, lineType=cv2.LINE_AA)
        cv2.putText(out, f"{prefix}{i}", (int(round(x)) + 4, int(round(y)) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


def draw_bbox(img_bgr: np.ndarray, bbox_xyxy, color=(255, 0, 0)) -> np.ndarray:
    out = img_bgr.copy()
    x1, y1, x2, y2 = bbox_xyxy
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
    return out


def parse_int_list_csv(s: str) -> List[int]:
    s = (s or "").strip()
    if not s:
        return []
    out: List[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


# ---------------------------------------------------------------------
# BBox crop utils (same as training logic)
# ---------------------------------------------------------------------
def kpts_bbox_xyxy(
    kpts: np.ndarray,
    img_wh: Tuple[int, int],
    margin: float = 0.10,
    make_square: bool = True,
    min_size_px: int = 32,
):
    """
    kpts: (K,2) in pixel coords in the resized image space (img_wh)
    returns bbox as (x1,y1,x2,y2) inclusive-exclusive in int, clipped.
    Uses only "valid" keypoints inside image.
    """
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
    pts2d: np.ndarray,
    bbox_xyxy: Tuple[int, int, int, int],
    out_wh: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    img_rgb_u8: (H,W,3) uint8 (FULL resized)
    pts2d: (K,2) in FULL resized coords
    bbox: (x1,y1,x2,y2) on FULL resized image
    out_wh: (Wout,Hout) output size (same as cfg.img_size)
    returns: img_out_u8, pts2d_out (in CROPPED resized coords), and transform dict
    """
    x1, y1, x2, y2 = bbox_xyxy
    crop = img_rgb_u8[y1:y2, x1:x2, :]

    ch, cw = crop.shape[:2]
    Wout, Hout = out_wh

    sx = Wout / float(max(cw, 1))
    sy = Hout / float(max(ch, 1))

    k = pts2d.copy().astype(np.float32)
    k[:, 0] = (k[:, 0] - x1) * sx
    k[:, 1] = (k[:, 1] - y1) * sy

    crop_res = cv2.resize(crop, (Wout, Hout), interpolation=cv2.INTER_LINEAR)

    tfm = dict(
        x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
        cw=int(cw), ch=int(ch),
        sx=float(sx), sy=float(sy),
        Wout=int(Wout), Hout=int(Hout),
    )
    return crop_res, k, tfm


def map_crop_to_full(pts_crop: np.ndarray, tfm: dict) -> np.ndarray:
    """
    Inverse of crop_and_resize points transform.
    pts_crop: (K,2) in CROPPED resized coords (Wout,Hout)
    return pts_full: (K,2) in FULL resized coords
    """
    sx = float(tfm["sx"])
    sy = float(tfm["sy"])
    x1 = float(tfm["x1"])
    y1 = float(tfm["y1"])
    p = np.asarray(pts_crop, dtype=np.float32).copy()
    p[:, 0] = p[:, 0] / max(sx, 1e-12) + x1
    p[:, 1] = p[:, 1] / max(sy, 1e-12) + y1
    return p


def crop_intrinsics(K_full: np.ndarray, tfm: dict) -> np.ndarray:
    """
    Given FULL resized intrinsics K_full, produce intrinsics for CROPPED resized image.
    Crop: subtract (x1,y1), then resize by (sx,sy).
    """
    K = np.asarray(K_full, dtype=np.float32).copy()
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    x1 = float(tfm["x1"])
    y1 = float(tfm["y1"])
    sx = float(tfm["sx"])
    sy = float(tfm["sy"])

    fx2 = fx * sx
    fy2 = fy * sy
    cx2 = (cx - x1) * sx
    cy2 = (cy - y1) * sy

    K2 = np.array(
        [[fx2, 0.0, cx2],
         [0.0, fy2, cy2],
         [0.0, 0.0, 1.0]],
        dtype=np.float32
    )
    return K2


# ---------------------------------------------------------------------
# Metrics (SPEED+/ESA-like)
# ---------------------------------------------------------------------
def orientation_error_rad(q_gt_wxyz: np.ndarray, q_est_wxyz: np.ndarray) -> float:
    qg = np.asarray(q_gt_wxyz, dtype=np.float64)
    qe = np.asarray(q_est_wxyz, dtype=np.float64)
    qg = qg / (np.linalg.norm(qg) + 1e-12)
    qe = qe / (np.linalg.norm(qe) + 1e-12)
    dot = float(np.abs(np.dot(qg, qe)))
    dot = max(-1.0, min(1.0, dot))
    return float(2.0 * np.arccos(dot))


def esa_score_components(
    t_gt: np.ndarray,
    t_est: np.ndarray,
    q_gt_wxyz: np.ndarray,
    q_est_wxyz: np.ndarray,
):
    denom = float(np.linalg.norm(t_gt) + 1e-12)
    err_pos = float(np.linalg.norm(t_gt - t_est) / denom)
    err_ori = float(orientation_error_rad(q_gt_wxyz, q_est_wxyz))

    thr_pos = 0.002173
    thr_ori = np.deg2rad(0.169)

    score_pos = 0.0 if err_pos < thr_pos else err_pos
    score_ori = 0.0 if err_ori < thr_ori else err_ori

    score = (score_pos + score_ori)
    return err_pos, err_ori, score_pos, score_ori, score


# ---------------------------------------------------------------------
# PnP RANSAC
# ---------------------------------------------------------------------
def solve_pnp_ransac(
    kpts_3d: np.ndarray,
    kpts_2d: np.ndarray,
    K: np.ndarray,
    reprojection_error_px: float,
    confidence: float,
    iterations: int,
    pnp_flag: int,
):
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=kpts_3d.astype(np.float32),
        imagePoints=kpts_2d.astype(np.float32),
        cameraMatrix=K.astype(np.float32),
        distCoeffs=None,
        flags=int(pnp_flag),
        reprojectionError=float(reprojection_error_px),
        confidence=float(confidence),
        iterationsCount=int(iterations),
    )
    if not ok:
        return None, None, None
    R_CT, _ = cv2.Rodrigues(rvec)
    t_C = tvec.reshape(3).astype(np.float32)
    return R_CT.astype(np.float32), t_C, inliers


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass
class EvalCfg:
    ckpt: str
    split: str = "validation"
    domain: str = "synthetic"
    img_size: Tuple[int, int] = (512, 512)
    heatmap_size: Tuple[int, int] = (128, 128)
    sigma: float = 2.0
    max_samples: int = -1
    kpt_extractor: str = "softargmax"  # ["softargmax","argmax"]

    # softargmax
    beta_softarg: float = 50.0

    # PnP
    pnp_method: str = "EPNP"
    reprojection_error_px: float = 12.0
    ransac_confidence: float = 0.999
    ransac_iterations: int = 500

    # Gates/filters
    min_inliers: int = 6
    min_inliers_schedule: str = ""        # NEW: e.g. "11,9,8,6,4"
    rmse_inliers_thr_px: float = 15.0
    min_kpt_area_px2: float = 5000.0       # anti-cluster gate
    t_ratio_max: float = 10.0              # ||t_est|| / ||t_gt|| gate

    # Disable gates (challenge aligned): keep as many samples as possible
    disable_too_few_inliers_gate: int = 0
    disable_reproj_gate: int = 0
    disable_area_gate: int = 0
    disable_t_ratio_gate: int = 0

    # BBox-crop eval (must match training)
    use_bbox_crop: int = 1
    bbox_margin: float = 0.15
    bbox_square: int = 1
    bbox_min_size: int = 64

    # Output
    out_dir: str = ""
    out_root: str = "runs_eval"
    run_tag: str = "epnp"

    # Compatibility flags (your job passes these)
    viz: int = 0
    viz_n: int = 0
    viz_stride: int = 100
    save_per_sample_json: int = 0
    save_kpts_txt: int = 0
    save_pose_json: int = 0


def parse_args() -> EvalCfg:
    ap = argparse.ArgumentParser()

    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--domain", type=str, default="synthetic",
                    choices=["synthetic", "lightbox", "sunlamp"])
    ap.add_argument("--split", type=str, default="validation")
    ap.add_argument("--img_size", type=int, nargs=2,
                    default=[512, 512], metavar=("W", "H"))
    ap.add_argument("--heatmap_size", type=int, nargs=2,
                    default=[128, 128], metavar=("Wm", "Hm"))
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--max_samples", type=int, default=-1)

    ap.add_argument("--beta_softarg", type=float, default=50.0)
    ap.add_argument("--kpt_extractor", type=str, default="softargmax",
                    choices=["softargmax", "argmax"])

    ap.add_argument("--pnp_method", type=str, default="EPNP",
                    choices=["EPNP", "ITERATIVE"])
    ap.add_argument("--reprojection_error_px", type=float, default=8.0)
    ap.add_argument("--confidence", type=float, default=0.999)
    ap.add_argument("--iterations", type=int, default=100)

    # Gates
    ap.add_argument("--min_inliers", type=int, default=6)
    ap.add_argument("--min_inliers_schedule", type=str, default="")  # NEW
    ap.add_argument("--rmse_inliers_thr_px", type=float, default=15.0)
    ap.add_argument("--min_kpt_area_px2", type=float, default=5000.0)
    ap.add_argument("--t_ratio_max", type=float, default=10.0)
    ap.add_argument("--disable_too_few_inliers_gate", type=int, default=0)
    ap.add_argument("--disable_reproj_gate", type=int, default=0)
    ap.add_argument("--disable_area_gate", type=int, default=0)
    ap.add_argument("--disable_t_ratio_gate", type=int, default=0)

    # BBox-crop eval
    ap.add_argument("--use_bbox_crop", type=int, default=1)
    ap.add_argument("--bbox_margin", type=float, default=0.15)
    ap.add_argument("--bbox_square", type=int, default=1)
    ap.add_argument("--bbox_min_size", type=int, default=64)

    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--out_root", type=str, default="runs_eval")
    ap.add_argument("--run_tag", type=str, default="epnp")

    # Compatibility with your job
    ap.add_argument("--viz", type=int, default=0)
    ap.add_argument("--viz_n", type=int, default=0)
    ap.add_argument("--viz_stride", type=int, default=100)
    ap.add_argument("--save_per_sample_json", type=int, default=0)
    ap.add_argument("--save_kpts_txt", type=int, default=0)
    ap.add_argument("--save_pose_json", type=int, default=0)

    a = ap.parse_args()
    return EvalCfg(
        ckpt=a.ckpt,
        domain=a.domain,
        split=a.split,
        img_size=(int(a.img_size[0]), int(a.img_size[1])),
        heatmap_size=(int(a.heatmap_size[0]), int(a.heatmap_size[1])),
        sigma=float(a.sigma),
        max_samples=int(a.max_samples),
        beta_softarg=float(a.beta_softarg),
        pnp_method=str(a.pnp_method),
        reprojection_error_px=float(a.reprojection_error_px),
        ransac_confidence=float(a.confidence),
        ransac_iterations=int(a.iterations),
        min_inliers=int(a.min_inliers),
        min_inliers_schedule=str(a.min_inliers_schedule),
        rmse_inliers_thr_px=float(a.rmse_inliers_thr_px),
        min_kpt_area_px2=float(a.min_kpt_area_px2),
        t_ratio_max=float(a.t_ratio_max),
        disable_too_few_inliers_gate=int(a.disable_too_few_inliers_gate),
        disable_reproj_gate=int(a.disable_reproj_gate),
        disable_area_gate=int(a.disable_area_gate),
        disable_t_ratio_gate=int(a.disable_t_ratio_gate),

        use_bbox_crop=int(a.use_bbox_crop),
        bbox_margin=float(a.bbox_margin),
        bbox_square=int(a.bbox_square),
        bbox_min_size=int(a.bbox_min_size),
        out_dir=str(a.out_dir),
        out_root=str(a.out_root),
        run_tag=str(a.run_tag),
        viz=int(a.viz),
        viz_n=int(a.viz_n),
        viz_stride=int(a.viz_stride),
        save_per_sample_json=int(a.save_per_sample_json),
        save_kpts_txt=int(a.save_kpts_txt),
        save_pose_json=int(a.save_pose_json),
        kpt_extractor=str(a.kpt_extractor),
    )


def make_out_dir(cfg: EvalCfg) -> Path:
    if cfg.out_dir.strip():
        out_dir = resolve_path(cfg.out_dir)
        ensure_dir(out_dir)
        return out_dir
    out_root = resolve_path(cfg.out_root)
    ensure_dir(out_root)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = out_root / f"{ts}_{cfg.run_tag}_{cfg.domain}_{cfg.split}"
    ensure_dir(out_dir)
    return out_dir


def pick_dynamic_min_inliers(n_inliers: int, schedule_desc: List[int]) -> Optional[int]:
    """
    Return the highest threshold in schedule that is <= n_inliers.
    schedule_desc must be sorted descending (e.g., [11,9,8,6,4]).
    If none is satisfied, returns None.
    """
    for thr in schedule_desc:
        if n_inliers >= thr:
            return int(thr)
    return None


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    cfg = parse_args()
    ckpt_path = resolve_path(cfg.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_dir = make_out_dir(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] PROJECT ROOT:", project_root())
    print("[INFO] CWD         :", Path.cwd())
    print("[INFO] CUDA avail  :", torch.cuda.is_available())
    print("[INFO] device      :", device)
    if device.type == "cuda":
        print("[INFO] GPU         :", torch.cuda.get_device_name(0))

    env_info = {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "gpu_name": (torch.cuda.get_device_name(0) if device.type == "cuda" else None),
    }
    save_json(out_dir / "eval_config.json", {**asdict(cfg), **env_info})

    # Dynamic schedule (if provided). Otherwise fallback to fixed cfg.min_inliers.
    schedule = parse_int_list_csv(cfg.min_inliers_schedule)
    schedule = sorted(set(schedule), reverse=True) if schedule else []
    schedule_min = int(min(schedule)) if schedule else int(cfg.min_inliers)

    # Data
    ds = SpeedPlusKeypointDataset(
        split=cfg.split,
        domain=cfg.domain,
        img_size=cfg.img_size,
        heatmap_size=cfg.heatmap_size,
        sigma=cfg.sigma,
        return_heatmaps=True,
    )

    n_eval = len(ds) if cfg.max_samples < 0 else min(len(ds), cfg.max_samples)
    print("[INFO] Dataset size:", len(ds), "| Evaluating:", n_eval)
    if schedule:
        print("[INFO] Dynamic min_inliers schedule:",
              schedule, "| min:", schedule_min)
    else:
        print("[INFO] Fixed min_inliers:", cfg.min_inliers)

    # 3D keypoints and K_full
    kpts_3d_all = np.load(KEYPOINTS_3D_NPY).astype(np.float32)
    num_kpts = int(kpts_3d_all.shape[0])

    K0, W0, H0 = load_camera_matrix()
    K_full = scale_intrinsics(K0, (W0, H0), cfg.img_size).astype(np.float32)

    # Model
    model = HRNetKeypointModel(num_keypoints=num_kpts).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state"] if isinstance(
        ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    if cfg.pnp_method == "EPNP":
        pnp_flag = cv2.SOLVEPNP_EPNP
    else:
        pnp_flag = cv2.SOLVEPNP_ITERATIVE

    # Metrics accumulators
    E_t, e_t, e_q = [], [], []
    rmse_px = []
    esa, esa_pos, esa_ori = [], [], []

    # Timing
    pnp_times_ms_attempted: List[float] = []
    pnp_times_ms_success: List[float] = []

    # Counts
    failed_pnp = 0
    too_few_inliers = 0
    rejected_reproj = 0
    rejected_area = 0
    rejected_t_ratio = 0
    missing_gt = 0
    missing_gt_kpts_for_crop = 0

    # NEW: tier counts
    tier_counts: Dict[str, int] = {}
    tier_counts["fixed"] = 0  # used when schedule not provided

    per_sample: List[Dict[str, Any]] = []
    viz_saved = 0

    t_total0 = time.perf_counter()
    amp_device = "cuda" if device.type == "cuda" else "cpu"

    Wout, Hout = cfg.img_size

    for i in tqdm(range(n_eval), desc=f"eval {cfg.domain}/{cfg.split}"):
        sample = ds[i]
        img_name = sample.get("img_name", f"idx{i:06d}")

        # FULL resized image for crop (uint8 RGB)
        img_full_rgb = (sample["image"].detach().cpu().numpy().transpose(
            1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)

        # FULL GT keypoints (used for bbox + metrics in full coords)
        kpts_2d_gt_full = to_numpy(sample.get("kpts_2d", np.full(
            (num_kpts, 2), -1.0, np.float32))).astype(np.float32)

        # Decide crop
        use_crop = bool(cfg.use_bbox_crop)
        tfm = None
        bbox = None
        K_crop = K_full
        img_in_rgb = img_full_rgb  # default: full

        if use_crop:
            if (kpts_2d_gt_full is None) or (not np.all(np.isfinite(kpts_2d_gt_full))) or (not np.any(kpts_2d_gt_full[:, 0] >= 0)):
                missing_gt_kpts_for_crop += 1
                use_crop = False
            else:
                bbox = kpts_bbox_xyxy(
                    kpts_2d_gt_full,
                    img_wh=(Wout, Hout),
                    margin=float(cfg.bbox_margin),
                    make_square=bool(cfg.bbox_square),
                    min_size_px=int(cfg.bbox_min_size),
                )
                if bbox is None:
                    missing_gt_kpts_for_crop += 1
                    use_crop = False
                else:
                    img_in_rgb, _, tfm = crop_and_resize(
                        img_full_rgb,
                        kpts_2d_gt_full,
                        bbox,
                        out_wh=(Wout, Hout),
                    )
                    K_crop = crop_intrinsics(K_full, tfm)

        # Prepare tensor for model
        img_in_t = torch.from_numpy(img_in_rgb.astype(
            np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

        # forward -> heatmaps
        with torch.no_grad():
            with torch.amp.autocast(amp_device, enabled=(device.type == "cuda")):
                hmaps = model(img_in_t)  # (1,K,Hm,Wm)

        # kpts pred in CROPPED resized coords (coords of the image fed to model)
        if cfg.kpt_extractor == "softargmax":
            coords_hm, _ = soft_argmax_2d(
                hmaps, beta=cfg.beta_softarg)  # (1,K,2)
        else:
            probs = torch.sigmoid(hmaps)
            coords_hm = heatmaps_argmax_2d(probs)  # (1,K,2)

        coords_img = hm_to_image_coords(
            coords_hm,
            heatmap_size=(hmaps.shape[-2], hmaps.shape[-1]),
            img_size=cfg.img_size,
        )
        kpts_2d_pred_crop = coords_img[0].detach(
        ).cpu().numpy().astype(np.float32)

        # Map predicted keypoints back to FULL resized coords for 2D metrics / viz
        if use_crop and (tfm is not None):
            kpts_2d_pred_full = map_crop_to_full(kpts_2d_pred_crop, tfm)
        else:
            kpts_2d_pred_full = kpts_2d_pred_crop.copy()

        # -----------------------------
        # PnP on CROPPED resized coords
        # -----------------------------
        t0 = time.perf_counter()
        R_CT_est, t_est, inliers = solve_pnp_ransac(
            kpts_3d=kpts_3d_all,
            kpts_2d=kpts_2d_pred_crop,
            K=K_crop,
            reprojection_error_px=cfg.reprojection_error_px,
            confidence=cfg.ransac_confidence,
            iterations=cfg.ransac_iterations,
            pnp_flag=pnp_flag,
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        pnp_times_ms_attempted.append(dt_ms)

        if R_CT_est is None or t_est is None or inliers is None:
            failed_pnp += 1
            if cfg.save_per_sample_json:
                per_sample.append({
                    "i": i, "img_name": img_name,
                    "success": False, "reason": "pnp_fail",
                    "pnp_ms": float(dt_ms),
                    "use_crop": bool(use_crop),
                })
            continue

        idx_inl = inliers.reshape(-1)
        ninl = int(idx_inl.size)

        # ============================
        # DYNAMIC MIN-INLIERS ACCEPT
        # ============================
        if schedule:
            used_min_inliers = pick_dynamic_min_inliers(ninl, schedule)
            if used_min_inliers is None:
                if cfg.disable_too_few_inliers_gate:
                    used_min_inliers = 0
                    tier_key = "tier_0"
                    tier_counts[tier_key] = int(tier_counts.get(tier_key, 0)+1)
                else:
                    too_few_inliers += 1
                    if cfg.save_per_sample_json:
                        per_sample.append({
                            "i": i, "img_name": img_name,
                            "success": False, "reason": "too_few_inliers",
                            "pnp_ms": float(dt_ms),
                            "n_inliers": ninl,
                            "min_inliers_schedule": schedule,
                            "used_min_inliers": None,
                            "use_crop": bool(use_crop),
                        })
                    continue
            else:
                tier_key = f"tier_{used_min_inliers}"
                tier_counts[tier_key] = int(tier_counts.get(tier_key, 0) + 1)
        else:
            used_min_inliers = int(cfg.min_inliers)
            tier_counts["fixed"] = int(tier_counts.get("fixed", 0) + 1)
            if ninl < used_min_inliers:
                if not cfg.disable_too_few_inliers_gate:
                    too_few_inliers += 1
                    if cfg.save_per_sample_json:
                        per_sample.append({
                            "i": i, "img_name": img_name,
                            "success": False, "reason": "too_few_inliers",
                            "pnp_ms": float(dt_ms),
                            "n_inliers": ninl,
                            "min_inliers": int(cfg.min_inliers),
                            "use_crop": bool(use_crop),
                        })
                    continue
                else:
                    used_min_inliers = 0

        # Reprojection on inliers gate (CROP coords)
        proj_crop_all = reprojection_points(
            kpts_3d_all, K_crop, R_CT_est, t_est)  # (K,2)
        rmse_inl = reprojection_rmse_px(
            proj_crop_all[idx_inl], kpts_2d_pred_crop[idx_inl])
        if (not cfg.disable_reproj_gate) and (rmse_inl > cfg.rmse_inliers_thr_px):
            rejected_reproj += 1
            if cfg.save_per_sample_json:
                per_sample.append({
                    "i": i, "img_name": img_name,
                    "success": False, "reason": "reproj_gate",
                    "pnp_ms": float(dt_ms),
                    "n_inliers": ninl,
                    "used_min_inliers": used_min_inliers,
                    "rmse_inliers_px": float(rmse_inl),
                    "use_crop": bool(use_crop),
                })
            continue

        # Anti-cluster gate (in CROP coords)
        xy = kpts_2d_pred_crop[idx_inl]
        w = float(xy[:, 0].max() - xy[:, 0].min())
        h = float(xy[:, 1].max() - xy[:, 1].min())
        area = float(w * h)
        if (not cfg.disable_area_gate) and (area < cfg.min_kpt_area_px2):
            rejected_area += 1
            if cfg.save_per_sample_json:
                per_sample.append({
                    "i": i, "img_name": img_name,
                    "success": False, "reason": "kpt_area_gate",
                    "pnp_ms": float(dt_ms),
                    "n_inliers": ninl,
                    "used_min_inliers": used_min_inliers,
                    "rmse_inliers_px": float(rmse_inl),
                    "kpt_bbox_w": w, "kpt_bbox_h": h, "kpt_area_px2": area,
                    "use_crop": bool(use_crop),
                })
            continue

        # PnP accepted
        pnp_times_ms_success.append(dt_ms)

        # Load GT pose for metrics + t_ratio gate
        q_gt_wxyz = to_numpy(sample["q"]).astype(np.float32)
        t_gt = to_numpy(sample["t"]).astype(np.float32)

        if (not is_finite_arr(q_gt_wxyz)) or (not is_finite_arr(t_gt)):
            missing_gt += 1
            if cfg.save_per_sample_json:
                per_sample.append({
                    "i": i, "img_name": img_name,
                    "success": True, "accepted_pose": True, "metrics_computed": False,
                    "reason": "missing_gt",
                    "pnp_ms": float(dt_ms),
                    "n_inliers": ninl,
                    "used_min_inliers": used_min_inliers,
                    "rmse_inliers_px": float(rmse_inl),
                    "kpt_area_px2": area,
                    "use_crop": bool(use_crop),
                })
            continue

        t_est_norm = float(np.linalg.norm(t_est))
        t_gt_norm = float(np.linalg.norm(t_gt) + 1e-12)
        t_ratio = float(t_est_norm / t_gt_norm)

        if (not cfg.disable_t_ratio_gate) and (t_ratio > cfg.t_ratio_max):
            rejected_t_ratio += 1
            if cfg.save_per_sample_json:
                per_sample.append({
                    "i": i, "img_name": img_name,
                    "success": True, "accepted_pose": False, "metrics_computed": False,
                    "reason": "t_ratio_gate",
                    "pnp_ms": float(dt_ms),
                    "n_inliers": ninl,
                    "used_min_inliers": used_min_inliers,
                    "rmse_inliers_px": float(rmse_inl),
                    "kpt_area_px2": area,
                    "t_est_norm": t_est_norm,
                    "t_gt_norm": t_gt_norm,
                    "t_ratio": t_ratio,
                    "use_crop": bool(use_crop),
                })
            continue

        # Quaternion from estimated R (SciPy xyzw)
        q_est_xyzw = SciRot.from_matrix(R_CT_est).as_quat().astype(np.float32)
        q_est_wxyz = np.array(
            [q_est_xyzw[3], q_est_xyzw[0], q_est_xyzw[1], q_est_xyzw[2]], dtype=np.float32)

        # For 2D metrics: project with K_full into FULL coords
        proj_full = reprojection_points(kpts_3d_all, K_full, R_CT_est, t_est)

        rmse_vs_gt = float("nan")
        if is_finite_arr(kpts_2d_gt_full) and np.all(kpts_2d_gt_full >= 0):
            rmse_vs_gt = reprojection_rmse_px(
                kpts_2d_pred_full, kpts_2d_gt_full)

        # Pose metrics
        Et_i = float(np.linalg.norm(t_gt - t_est))
        et_i = float(Et_i / (np.linalg.norm(t_gt) + 1e-12))
        eq_i = float(orientation_error_rad(q_gt_wxyz, q_est_wxyz))
        _, _, sc_pos, sc_ori, sc = esa_score_components(
            t_gt, t_est, q_gt_wxyz, q_est_wxyz)

        E_t.append(Et_i)
        e_t.append(et_i)
        e_q.append(eq_i)
        if np.isfinite(rmse_vs_gt):
            rmse_px.append(float(rmse_vs_gt))
        esa.append(sc)
        esa_pos.append(sc_pos)
        esa_ori.append(sc_ori)

        # Optional saves
        if cfg.save_kpts_txt:
            kp_path = out_dir / "kpts_txt" / f"{Path(img_name).stem}.txt"
            ensure_dir(kp_path.parent)
            with open(kp_path, "w", encoding="utf-8") as f:
                for (x, y) in kpts_2d_pred_full:
                    f.write(f"{x:.6f} {y:.6f}\n")

        if cfg.save_pose_json:
            pose_path = out_dir / "pose_json" / f"{Path(img_name).stem}.json"
            ensure_dir(pose_path.parent)
            save_json(pose_path, {
                "img_name": img_name,
                "R_CT": R_CT_est.tolist(),
                "t_C": t_est.tolist(),
                "q_est_wxyz": q_est_wxyz.tolist(),
                "n_inliers": ninl,
                "used_min_inliers": used_min_inliers,
                "rmse_inliers_px": float(rmse_inl),
                "kpt_area_px2": float(area),
                "t_est_norm": float(t_est_norm),
                "t_gt_norm": float(t_gt_norm),
                "t_ratio": float(t_ratio),
                "use_crop": bool(use_crop),
                "bbox": (None if bbox is None else list(map(int, bbox))),
            })

        if cfg.viz and (cfg.viz_n > 0) and (i % cfg.viz_stride == 0) and (viz_saved < cfg.viz_n):
            img_bgr = cv2.cvtColor(img_full_rgb, cv2.COLOR_RGB2BGR)

            if use_crop and (bbox is not None):
                img_bgr = draw_bbox(img_bgr, bbox, color=(255, 0, 0))

            vis = draw_points(img_bgr, kpts_2d_pred_full,
                              (255, 0, 0), prefix="p")  # blue
            vis = draw_points(vis, proj_full, (0, 0, 255),
                              prefix="r")              # red

            if is_finite_arr(kpts_2d_gt_full) and np.all(kpts_2d_gt_full >= 0):
                vis = draw_points(vis, kpts_2d_gt_full,
                                  (0, 255, 0), prefix="g")    # green

            vpath = out_dir / "viz" / f"{Path(img_name).stem}_overlay_full.jpg"
            ensure_dir(vpath.parent)
            cv2.imwrite(str(vpath), vis)
            viz_saved += 1

        if cfg.save_per_sample_json:
            per_sample.append({
                "i": i,
                "img_name": img_name,
                "success": True,
                "accepted_pose": True,
                "metrics_computed": True,
                "Et_m": Et_i,
                "et": et_i,
                "eq_rad": eq_i,
                "rmse_px": (None if not np.isfinite(rmse_vs_gt) else float(rmse_vs_gt)),
                "esa": sc,
                "esa_pos": sc_pos,
                "esa_ori": sc_ori,
                "pnp_ms": float(dt_ms),
                "n_inliers": ninl,
                "used_min_inliers": used_min_inliers,
                "rmse_inliers_px": float(rmse_inl),
                "kpt_bbox_w": w,
                "kpt_bbox_h": h,
                "kpt_area_px2": area,
                "t_est_norm": t_est_norm,
                "t_gt_norm": t_gt_norm,
                "t_ratio": t_ratio,
                "use_crop": bool(use_crop),
                "bbox": (None if bbox is None else list(map(int, bbox))),
            })

    total_s = float(time.perf_counter() - t_total0)
    total_fps = float(n_eval / total_s) if total_s > 0 else None

    summary = {
        "cfg": asdict(cfg),
        "env": env_info,
        "dynamic": {
            "min_inliers_schedule": schedule if schedule else None,
            "tier_counts": tier_counts,
        },
        "counts": {
            "requested": int(n_eval),
            "metrics_computed": int(len(E_t)),
            "failed_pnp": int(failed_pnp),
            "too_few_inliers": int(too_few_inliers),
            "rejected_reproj": int(rejected_reproj),
            "rejected_area": int(rejected_area),
            "rejected_t_ratio": int(rejected_t_ratio),
            "missing_gt_pose": int(missing_gt),
            "missing_gt_kpts_for_crop": int(missing_gt_kpts_for_crop),
        },
        "metrics": {
            "E_t_m": summarize(np.array(E_t)),
            "e_t": summarize(np.array(e_t)),
            "e_q_rad": summarize(np.array(e_q)),
            "rmse_px": summarize(np.array(rmse_px)),
            "esa": summarize(np.array(esa)),
            "esa_pos": summarize(np.array(esa_pos)),
            "esa_ori": summarize(np.array(esa_ori)),
        },
        "timing": {
            "total_s": total_s,
            "total_fps": total_fps,
            "pnp_attempted": summarize_times_ms(pnp_times_ms_attempted),
            "pnp_success": summarize_times_ms(pnp_times_ms_success),
        },
    }

    # Pretty text
    lines: List[str] = []
    lines.append(
        "=== RESULTS (HRNet -> kpts -> (CROP) PnP) with robust gates ===")
    lines.append(f"ckpt       : {ckpt_path}")
    lines.append(f"out_dir    : {out_dir}")
    lines.append(f"domain/split: {cfg.domain} / {cfg.split}")
    lines.append("")
    lines.append("=== CROP ===")
    lines.append(f"use_bbox_crop: {bool(cfg.use_bbox_crop)}")
    lines.append(f"bbox_margin  : {cfg.bbox_margin}")
    lines.append(f"bbox_square  : {bool(cfg.bbox_square)}")
    lines.append(f"bbox_min_size: {cfg.bbox_min_size}")
    lines.append("")
    lines.append("=== FILTERS / GATES ===")
    if schedule:
        lines.append(f"min_inliers_schedule : {schedule}")
        lines.append(f"tier_counts          : {tier_counts}")
    else:
        lines.append(f"min_inliers         : {cfg.min_inliers}")
    lines.append(f"rmse_inliers_thr_px : {cfg.rmse_inliers_thr_px}")
    lines.append(f"min_kpt_area_px2    : {cfg.min_kpt_area_px2}")
    lines.append(f"t_ratio_max         : {cfg.t_ratio_max}")
    lines.append("")

    lines.append("=== COUNTS ===")
    for k, v in summary["counts"].items():
        lines.append(f"{k:>24s}: {v}")
    lines.append("")

    def fmt(name: str, key: str) -> str:
        s = summary["metrics"][key]
        if s["N"] == 0:
            return f"{name}: N=0"
        return f"{name}: N={s['N']} mean={s['mean']:.6f} median={s['median']:.6f} p90={s['p90']:.6f}"

    lines.append(fmt("E_t [m]", "E_t_m"))
    lines.append(fmt("e_t [-]", "e_t"))
    lines.append(fmt("e_q [rad]", "e_q_rad"))
    lines.append(fmt("RMSE(kpts pred vs gt) [px]", "rmse_px"))
    lines.append(fmt("ESA score [-]", "esa"))
    lines.append(fmt("ESA score_pos [-]", "esa_pos"))
    lines.append(fmt("ESA score_ori [rad]", "esa_ori"))
    lines.append("")
    lines.append("=== TIMING ===")
    lines.append(
        f"Total wall time (end-to-end): {summary['timing']['total_s']:.3f} s")
    if summary["timing"]["total_fps"] is not None:
        lines.append(
            f"Total throughput (end-to-end): {summary['timing']['total_fps']:.2f} FPS")
    if summary["timing"]["pnp_attempted"] is not None:
        t = summary["timing"]["pnp_attempted"]
        lines.append(
            f"PnP attempted: mean={t['mean_ms']:.3f} ms | median={t['median_ms']:.3f} ms | p90={t['p90_ms']:.3f} ms")
    if summary["timing"]["pnp_success"] is not None:
        t = summary["timing"]["pnp_success"]
        lines.append(
            f"PnP success  : mean={t['mean_ms']:.3f} ms | median={t['median_ms']:.3f} ms | p90={t['p90_ms']:.3f} ms")

    text = "\n".join(lines)
    print("\n" + text)

    save_json(out_dir / "summary.json", summary)
    save_txt(out_dir / "metrics.txt", text)
    if cfg.save_per_sample_json:
        save_json(out_dir / "per_sample.json", {"samples": per_sample})

    print("\n[DONE] Saved:")
    print(" -", (out_dir / "eval_config.json").resolve())
    print(" -", (out_dir / "summary.json").resolve())
    print(" -", (out_dir / "metrics.txt").resolve())
    if cfg.save_per_sample_json:
        print(" -", (out_dir / "per_sample.json").resolve())
    if cfg.save_kpts_txt:
        print(" -", (out_dir / "kpts_txt").resolve())
    if cfg.save_pose_json:
        print(" -", (out_dir / "pose_json").resolve())
    if cfg.viz and cfg.viz_n > 0:
        print(" -", (out_dir / "viz").resolve())


if __name__ == "__main__":
    main()
