# src/eval_pnp.py
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
    proj, _ = cv2.projectPoints(kpts_3d.astype(
        np.float64), rvec, tvec, K.astype(np.float64), None)
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
    reprojection_error_px: float = 8.0
    ransac_confidence: float = 0.999
    ransac_iterations: int = 100

    # Gates/filters
    min_inliers: int = 6
    rmse_inliers_thr_px: float = 15.0
    min_kpt_area_px2: float = 5000.0       # anti-cluster gate
    t_ratio_max: float = 10.0              # ||t_est|| / ||t_gt|| gate

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

    # Gates requested + suggested
    ap.add_argument("--min_inliers", type=int, default=6)
    ap.add_argument("--rmse_inliers_thr_px", type=float, default=15.0)
    ap.add_argument("--min_kpt_area_px2", type=float, default=5000.0)
    ap.add_argument("--t_ratio_max", type=float, default=10.0)

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
        rmse_inliers_thr_px=float(a.rmse_inliers_thr_px),
        min_kpt_area_px2=float(a.min_kpt_area_px2),
        t_ratio_max=float(a.t_ratio_max),
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

    # Data
    ds = SpeedPlusKeypointDataset(
        split=cfg.split,
        domain=cfg.domain,
        img_size=cfg.img_size,
        heatmap_size=cfg.heatmap_size,
        sigma=cfg.sigma,
        return_heatmaps=True,  # serve per pred
    )

    n_eval = len(ds) if cfg.max_samples < 0 else min(len(ds), cfg.max_samples)
    print("[INFO] Dataset size:", len(ds), "| Evaluating:", n_eval)

    # 3D keypoints and K
    kpts_3d_all = np.load(KEYPOINTS_3D_NPY).astype(np.float32)
    num_kpts = int(kpts_3d_all.shape[0])

    K0, W0, H0 = load_camera_matrix()
    K = scale_intrinsics(K0, (W0, H0), cfg.img_size).astype(np.float32)

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

    per_sample: List[Dict[str, Any]] = []
    viz_saved = 0

    t_total0 = time.perf_counter()
    amp_device = "cuda" if device.type == "cuda" else "cpu"

    for i in tqdm(range(n_eval), desc=f"eval {cfg.domain}/{cfg.split}"):
        sample = ds[i]
        img_name = sample.get("img_name", f"idx{i:06d}")
        img_t = sample["image"].unsqueeze(0).to(device)

        # forward -> heatmaps
        with torch.no_grad():
            with torch.amp.autocast(amp_device, enabled=(device.type == "cuda")):
                hmaps = model(img_t)  # (1,K,Hm,Wm)

        if cfg.kpt_extractor == "softargmax":
            coords_hm, _ = soft_argmax_2d(
                hmaps, beta=cfg.beta_softarg)  # (1,K,2)
        else:
            # Argmax on PROB maps is safer than logits
            probs = torch.sigmoid(hmaps)
            coords_hm = heatmaps_argmax_2d(probs)  # (1,K,2)

        coords_img = hm_to_image_coords(
            coords_hm,
            heatmap_size=(hmaps.shape[-2], hmaps.shape[-1]),
            img_size=cfg.img_size,
        )
        kpts_2d_pred_all = coords_img[0].detach(
        ).cpu().numpy().astype(np.float32)

        # Attempt PnP (all kpts; robust gates will reject degeneracies)
        t0 = time.perf_counter()
        R_CT_est, t_est, inliers = solve_pnp_ransac(
            kpts_3d=kpts_3d_all,
            kpts_2d=kpts_2d_pred_all,
            K=K,
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
                })
            continue

        idx_inl = inliers.reshape(-1)
        ninl = int(idx_inl.size)
        if ninl < cfg.min_inliers:
            too_few_inliers += 1
            if cfg.save_per_sample_json:
                per_sample.append({
                    "i": i, "img_name": img_name,
                    "success": False, "reason": "too_few_inliers",
                    "pnp_ms": float(dt_ms),
                    "n_inliers": ninl,
                })
            continue

        # Reprojection on inliers gate
        proj_all = reprojection_points(
            kpts_3d_all, K, R_CT_est, t_est)  # (K,2)
        rmse_inl = reprojection_rmse_px(
            proj_all[idx_inl], kpts_2d_pred_all[idx_inl])
        if rmse_inl > cfg.rmse_inliers_thr_px:
            rejected_reproj += 1
            if cfg.save_per_sample_json:
                per_sample.append({
                    "i": i, "img_name": img_name,
                    "success": False, "reason": "reproj_gate",
                    "pnp_ms": float(dt_ms),
                    "n_inliers": ninl,
                    "rmse_inliers_px": float(rmse_inl),
                })
            continue

        # Anti-cluster gate on inlier keypoints (spread must be sufficient)
        xy = kpts_2d_pred_all[idx_inl]
        w = float(xy[:, 0].max() - xy[:, 0].min())
        h = float(xy[:, 1].max() - xy[:, 1].min())
        area = float(w * h)
        if area < cfg.min_kpt_area_px2:
            rejected_area += 1
            if cfg.save_per_sample_json:
                per_sample.append({
                    "i": i, "img_name": img_name,
                    "success": False, "reason": "kpt_area_gate",
                    "pnp_ms": float(dt_ms),
                    "n_inliers": ninl,
                    "rmse_inliers_px": float(rmse_inl),
                    "kpt_bbox_w": w, "kpt_bbox_h": h, "kpt_area_px2": area,
                })
            continue

        # Now we can count as "PnP success" (geometry accepted)
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
                    "rmse_inliers_px": float(rmse_inl),
                    "kpt_area_px2": area,
                })
            continue

        t_est_norm = float(np.linalg.norm(t_est))
        t_gt_norm = float(np.linalg.norm(t_gt) + 1e-12)
        t_ratio = float(t_est_norm / t_gt_norm)

        # Physical plausibility gate: t_ratio
        if t_ratio > cfg.t_ratio_max:
            rejected_t_ratio += 1
            if cfg.save_per_sample_json:
                per_sample.append({
                    "i": i, "img_name": img_name,
                    "success": True, "accepted_pose": False, "metrics_computed": False,
                    "reason": "t_ratio_gate",
                    "pnp_ms": float(dt_ms),
                    "n_inliers": ninl,
                    "rmse_inliers_px": float(rmse_inl),
                    "kpt_area_px2": area,
                    "t_est_norm": t_est_norm,
                    "t_gt_norm": t_gt_norm,
                    "t_ratio": t_ratio,
                })
            continue

        # Quaternion from estimated R (SciPy xyzw)
        q_est_xyzw = SciRot.from_matrix(R_CT_est).as_quat().astype(np.float32)
        q_est_wxyz = np.array(
            [q_est_xyzw[3], q_est_xyzw[0], q_est_xyzw[1], q_est_xyzw[2]], dtype=np.float32)

        # RMSE vs GT 2D if available and valid
        rmse_vs_gt = float("nan")
        kpts_2d_gt_all = to_numpy(sample.get("kpts_2d", np.full(
            (num_kpts, 2), -1.0, np.float32))).astype(np.float32)
        if is_finite_arr(kpts_2d_gt_all) and np.all(kpts_2d_gt_all >= 0):
            rmse_vs_gt = reprojection_rmse_px(proj_all, kpts_2d_gt_all)

        # Metrics
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
            # Save predicted 2D keypoints (all 11) as simple txt
            kp_path = out_dir / "kpts_txt" / f"{Path(img_name).stem}.txt"
            ensure_dir(kp_path.parent)
            with open(kp_path, "w", encoding="utf-8") as f:
                for (x, y) in kpts_2d_pred_all:
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
                "rmse_inliers_px": float(rmse_inl),
                "kpt_area_px2": float(area),
                "t_est_norm": float(t_est_norm),
                "t_gt_norm": float(t_gt_norm),
                "t_ratio": float(t_ratio),
            })

        if cfg.viz and (cfg.viz_n > 0) and (i % cfg.viz_stride == 0) and (viz_saved < cfg.viz_n):
            # Minimal viz: overlay GT (green) + pred (blue) + reproj (red)
            # Rebuild image for viz
            img_rgb = (sample["image"].detach().cpu().numpy(
            ).transpose(1, 2, 0) * 255.0).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            vis = draw_points(img_bgr, kpts_2d_pred_all,
                              (255, 0, 0), prefix="p")   # blue
            vis = draw_points(vis, proj_all, (0, 0, 255),
                              prefix="r")               # red

            if is_finite_arr(kpts_2d_gt_all) and np.all(kpts_2d_gt_all >= 0):
                vis = draw_points(vis, kpts_2d_gt_all,
                                  (0, 255, 0), prefix="g")     # green

            vpath = out_dir / "viz" / f"{Path(img_name).stem}_overlay.jpg"
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
                "rmse_inliers_px": float(rmse_inl),
                "kpt_bbox_w": w,
                "kpt_bbox_h": h,
                "kpt_area_px2": area,
                "t_est_norm": t_est_norm,
                "t_gt_norm": t_gt_norm,
                "t_ratio": t_ratio,
            })

    total_s = float(time.perf_counter() - t_total0)
    total_fps = float(n_eval / total_s) if total_s > 0 else None

    summary = {
        "cfg": asdict(cfg),
        "env": env_info,
        "counts": {
            "requested": int(n_eval),
            "metrics_computed": int(len(E_t)),
            "failed_pnp": int(failed_pnp),
            "too_few_inliers": int(too_few_inliers),
            "rejected_reproj": int(rejected_reproj),
            "rejected_area": int(rejected_area),
            "rejected_t_ratio": int(rejected_t_ratio),
            "missing_gt": int(missing_gt),
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
        "=== RESULTS (HRNet -> softargmax -> PnP) with robust gates ===")
    lines.append(f"ckpt       : {ckpt_path}")
    lines.append(f"out_dir    : {out_dir}")
    lines.append(f"domain/split: {cfg.domain} / {cfg.split}")
    lines.append("")
    lines.append("=== FILTERS / GATES ===")
    lines.append(f"min_inliers         : {cfg.min_inliers}")
    lines.append(f"rmse_inliers_thr_px : {cfg.rmse_inliers_thr_px}")
    lines.append(f"min_kpt_area_px2    : {cfg.min_kpt_area_px2}")
    lines.append(f"t_ratio_max         : {cfg.t_ratio_max}")
    lines.append("")
    lines.append("=== COUNTS ===")
    for k, v in summary["counts"].items():
        lines.append(f"{k:>18s}: {v}")
    lines.append("")

    def fmt(name: str, key: str) -> str:
        s = summary["metrics"][key]
        if s["N"] == 0:
            return f"{name}: N=0"
        return f"{name}: N={s['N']} mean={s['mean']:.6f} median={s['median']:.6f} p90={s['p90']:.6f}"

    lines.append(fmt("E_t [m]", "E_t_m"))
    lines.append(fmt("e_t [-]", "e_t"))
    lines.append(fmt("e_q [rad]", "e_q_rad"))
    lines.append(fmt("RMSE [px]", "rmse_px"))
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
