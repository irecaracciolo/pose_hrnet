#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trade-off analysis for PnP acceptance gates (accuracy vs coverage) on SPEED+ domains.

Pipeline:
  1) cache: runs HRNet inference -> saves predicted 2D keypoints (+ GT pose if available) into .npz
  2) sweep: runs PnP + gating grid -> writes results.csv + best.csv + best_point.json + plots

BEST point selection:
  - select_mode=ideal_point  (DEFAULT): closest-to-ideal (3D) using mean metrics:
      maximize accept_rate, minimize rmse_inliers_mean, minimize slab_mean (if available)
  - select_mode=pareto_knee: 2D pareto-front knee on (accept_rate vs objective_median)
  - select_mode=weighted:    2D weighted score on (accept_rate vs objective_median/p90)

NOTE:
  - The "ideal point" selection does NOT depend on have_gt. It uses columns if present.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.transform import Rotation as SciRot

from src.softargmax import soft_argmax_2d, hm_to_image_coords
from src.config import KEYPOINTS_3D_NPY, load_camera_matrix, scale_intrinsics
from src.dataset_speedplus import SpeedPlusKeypointDataset
from src.model_hrnet_pose import HRNetKeypointModel


# -----------------------------
# Utils
# -----------------------------
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


def summarize(arr: np.ndarray) -> Dict[str, Optional[float]]:
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"N": 0, "mean": None, "median": None, "p90": None}
    return {
        "N": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
    }


def orientation_error_rad(q_gt_wxyz: np.ndarray, q_est_wxyz: np.ndarray) -> float:
    qg = np.asarray(q_gt_wxyz, dtype=np.float64)
    qe = np.asarray(q_est_wxyz, dtype=np.float64)

    ng = np.linalg.norm(qg)
    ne = np.linalg.norm(qe)
    if not np.isfinite(ng) or not np.isfinite(ne) or ng < 1e-9 or ne < 1e-9:
        return np.nan

    qg = qg / (ng + 1e-12)
    qe = qe / (ne + 1e-12)

    dot = float(np.abs(np.dot(qg, qe)))
    dot = max(-1.0, min(1.0, dot))
    return float(2.0 * np.arccos(dot))


def reprojection_points(kpts_3d: np.ndarray, K: np.ndarray, R_CT: np.ndarray, t_C: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(R_CT.astype(np.float64))
    tvec = t_C.reshape(3, 1).astype(np.float64)
    proj, _ = cv2.projectPoints(
        kpts_3d.astype(np.float64),
        rvec,
        tvec,
        K.astype(np.float64),
        None,
    )
    return proj.reshape(-1, 2).astype(np.float32)


def reprojection_rmse_px(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.float32) - b.astype(np.float32)
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


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
    if not ok or inliers is None or len(inliers) == 0:
        return None, None, None
    R_CT, _ = cv2.Rodrigues(rvec)
    t_C = tvec.reshape(3).astype(np.float32)
    return R_CT.astype(np.float32), t_C, inliers.astype(np.int32)


def parse_list_int(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_list_float(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def has_valid_gt(t: np.ndarray, q: np.ndarray) -> bool:
    if t is None or q is None:
        return False
    t = np.asarray(t, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    if t.shape != (3,) or q.shape != (4,):
        return False
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(q)):
        return False
    if np.linalg.norm(t) < 1e-9:
        return False
    if np.linalg.norm(q) < 1e-9:
        return False
    return True


# -----------------------------
# CACHE stage
# -----------------------------
@dataclass
class CacheCfg:
    ckpt: str
    domain: str
    split: str
    out_npz: str
    img_size: Tuple[int, int] = (512, 512)
    heatmap_size: Tuple[int, int] = (128, 128)
    sigma: float = 2.0
    beta_softarg: float = 50.0
    max_samples: int = -1


def run_cache(cfg: CacheCfg) -> None:
    ckpt_path = resolve_path(cfg.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_npz = resolve_path(cfg.out_npz)
    ensure_dir(out_npz.parent)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[CACHE] device:", device, "| cuda:", torch.cuda.is_available())
    if device.type == "cuda":
        print("[CACHE] GPU:", torch.cuda.get_device_name(0))

    ds = SpeedPlusKeypointDataset(
        split=cfg.split,
        domain=cfg.domain,
        img_size=cfg.img_size,
        heatmap_size=cfg.heatmap_size,
        sigma=cfg.sigma,
        return_heatmaps=True,
    )
    n = len(ds) if cfg.max_samples < 0 else min(len(ds), cfg.max_samples)

    kpts_3d = np.load(KEYPOINTS_3D_NPY).astype(np.float32)
    K0, W0, H0 = load_camera_matrix()
    K = scale_intrinsics(K0, (W0, H0), cfg.img_size).astype(np.float32)

    model = HRNetKeypointModel(num_keypoints=kpts_3d.shape[0]).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    img_names: List[str] = []
    kpts2d_pred = np.zeros((n, kpts_3d.shape[0], 2), dtype=np.float32)

    t_gt = np.zeros((n, 3), dtype=np.float32)
    q_gt = np.zeros((n, 4), dtype=np.float32)
    gt_valid = np.zeros((n,), dtype=np.uint8)

    amp_device = "cuda" if device.type == "cuda" else "cpu"
    t0 = time.perf_counter()

    for i in tqdm(range(n), desc=f"[CACHE] {cfg.domain}/{cfg.split}"):
        s = ds[i]
        img_names.append(s.get("img_name", f"idx{i:06d}"))

        img = s["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            with torch.amp.autocast(amp_device, enabled=(device.type == "cuda")):
                hmaps = model(img)  # (1,K,Hm,Wm)

        coords_hm, _ = soft_argmax_2d(hmaps, beta=cfg.beta_softarg)
        coords_img = hm_to_image_coords(
            coords_hm,
            heatmap_size=(hmaps.shape[-2], hmaps.shape[-1]),
            img_size=cfg.img_size,
        )
        kpts2d_pred[i] = coords_img[0].detach().cpu().numpy().astype(np.float32)

        t = s.get("t", None)
        q = s.get("q", None)
        if t is not None and q is not None:
            t = np.asarray(t, dtype=np.float32).reshape(3,)
            q = np.asarray(q, dtype=np.float32).reshape(4,)
            t_gt[i] = t
            q_gt[i] = q
            gt_valid[i] = 1 if has_valid_gt(t, q) else 0
        else:
            gt_valid[i] = 0

    dt = time.perf_counter() - t0
    print(f"[CACHE] saved {n} samples in {dt:.2f}s -> {out_npz}")
    print(f"[CACHE] GT valid fraction: {gt_valid.mean():.3f}")

    np.savez_compressed(
        out_npz,
        img_names=np.array(img_names),
        kpts2d_pred=kpts2d_pred,
        t_gt=t_gt,
        q_gt=q_gt,
        gt_valid=gt_valid,
        K=K,
        kpts_3d=kpts_3d,
        meta=np.array([json.dumps(asdict(cfg))]),
    )


# -----------------------------
# SWEEP stage
# -----------------------------
@dataclass
class SweepCfg:
    npz: str
    out_dir: str

    # PnP settings
    pnp_method: str = "EPNP"
    reprojection_error_px: float = 8.0
    ransac_confidence: float = 0.999
    ransac_iterations: int = 100

    # Gate grids
    min_inliers_list: Optional[List[int]] = None
    rmse_inliers_thr_list: Optional[List[float]] = None
    min_kpt_area_list: Optional[List[float]] = None
    t_ratio_max_list: Optional[List[float]] = None

    # Guardrail
    min_accept_rate: float = 0.10
    top_k: int = 20

    # Selection mode
    select_mode: str = "ideal_point"   # ideal_point | pareto_knee | weighted

    # Weighted params (only for weighted)
    w_accept: float = 0.60
    w_objective: float = 0.40
    w_robust: float = 0.10

    # Plot preferences (OptionB scatter only)
    x_axis: str = "discard_pct"
    y_stat: str = "median"


def _objective_columns(have_gt: bool) -> Tuple[str, str]:
    # legacy 2D objective for pareto_knee/weighted (kept for compatibility)
    if have_gt:
        return ("slab_median", "slab_p90")
    return ("rmse_inliers_median", "rmse_inliers_p90")


def pareto_front_max_min(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = len(x)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        dom = (x >= x[i]) & (y <= y[i]) & ((x > x[i]) | (y < y[i]))
        if np.any(dom):
            is_pareto[i] = False
    return is_pareto


def choose_pareto_knee(df: pd.DataFrame, accept_col: str, obj_med_col: str) -> pd.Series:
    d = df.copy()
    d = d[np.isfinite(d[accept_col].astype(float))].copy()
    d = d[np.isfinite(d[obj_med_col].astype(float))].copy()
    if len(d) == 0:
        return df.iloc[0]

    x = d[accept_col].to_numpy(dtype=float)
    y = d[obj_med_col].to_numpy(dtype=float)

    pf = pareto_front_max_min(x, y)
    d_pf = d.loc[pf].copy()
    if len(d_pf) == 0:
        return d.sort_values([obj_med_col, accept_col], ascending=[True, False]).iloc[0]

    x_pf = d_pf[accept_col].to_numpy(dtype=float)
    y_pf = d_pf[obj_med_col].to_numpy(dtype=float)

    x_min, x_max = float(np.min(x_pf)), float(np.max(x_pf))
    y_min, y_max = float(np.min(y_pf)), float(np.max(y_pf))

    xn = (x_pf - x_min) / (x_max - x_min + 1e-12)
    yn = (y_pf - y_min) / (y_max - y_min + 1e-12)

    dist = np.sqrt((1.0 - xn) ** 2 + (0.0 - yn) ** 2)
    idx = int(np.argmin(dist))
    return d_pf.iloc[idx]


def choose_weighted(
    df: pd.DataFrame,
    accept_col: str,
    obj_med_col: str,
    obj_p90_col: str,
    w_accept: float,
    w_obj: float,
    w_robust: float,
) -> pd.Series:
    d = df.copy()
    for c in [accept_col, obj_med_col, obj_p90_col]:
        if c not in d.columns:
            return df.iloc[0]
        d = d[np.isfinite(d[c].astype(float))].copy()
    if len(d) == 0:
        return df.iloc[0]

    acc = d[accept_col].to_numpy(dtype=float)
    obj = d[obj_med_col].to_numpy(dtype=float)
    obj90 = d[obj_p90_col].to_numpy(dtype=float)

    def norm01(v: np.ndarray) -> np.ndarray:
        vmin, vmax = float(np.min(v)), float(np.max(v))
        return (v - vmin) / (vmax - vmin + 1e-12)

    acc_n = norm01(acc)
    obj_n = norm01(obj)
    obj90_n = norm01(obj90)

    score = w_accept * (1.0 - acc_n) + w_obj * obj_n + w_robust * obj90_n
    idx = int(np.argmin(score))
    return d.iloc[idx]


def choose_ideal_point_3d_mean(
    df: pd.DataFrame,
    acc_col: str = "accept_rate",
    rmse_col: str = "rmse_inliers_mean",
    slab_col: str = "slab_mean",
) -> Tuple[pd.Series, str]:
    """
    Closest-to-ideal / ideal-point criterion (3D, mean metrics):
      maximize acc, minimize rmse_mean, minimize slab_mean (if available).
    If slab_mean is missing or all NaN -> uses only (acc, rmse).
    """
    d = df.copy()

    # always need acc + rmse_mean
    for c in [acc_col, rmse_col]:
        if c not in d.columns:
            return df.iloc[0], f"ideal_point_3d_mean (missing {c} -> fallback first row)"
        d = d[np.isfinite(d[c].astype(float))].copy()
    if len(d) == 0:
        return df.iloc[0], "ideal_point_3d_mean (no finite acc/rmse -> fallback first row)"

    have_slab = slab_col in d.columns and np.isfinite(d[slab_col].astype(float)).any()
    if have_slab:
        d = d[np.isfinite(d[slab_col].astype(float))].copy()
        if len(d) == 0:
            have_slab = False  # fallback to 2D

    acc = d[acc_col].to_numpy(float)
    rmse = d[rmse_col].to_numpy(float)
    slab = d[slab_col].to_numpy(float) if have_slab else None

    def norm01(v: np.ndarray) -> np.ndarray:
        vmin, vmax = float(np.min(v)), float(np.max(v))
        return (v - vmin) / (vmax - vmin + 1e-12)

    acc_n = norm01(acc)
    rmse_n = norm01(rmse)

    if have_slab:
        slab_n = norm01(slab)
        dist = np.sqrt((1.0 - acc_n) ** 2 + (rmse_n) ** 2 + (slab_n) ** 2)
        criterion = "ideal_point_3d_mean on (maximize accept_rate, minimize rmse_inliers_mean, minimize slab_mean)"
    else:
        dist = np.sqrt((1.0 - acc_n) ** 2 + (rmse_n) ** 2)
        criterion = "ideal_point_2d_mean on (maximize accept_rate, minimize rmse_inliers_mean) [slab_mean unavailable]"

    best = d.iloc[int(np.argmin(dist))]
    return best, criterion


def rank_by_ideal_point_3d_mean(
    df: pd.DataFrame,
    acc_col: str = "accept_rate",
    rmse_col: str = "rmse_inliers_mean",
    slab_col: str = "slab_mean",
) -> pd.DataFrame:
    d = df.copy()
    # keep rows where we can compute distance
    for c in [acc_col, rmse_col]:
        d = d[np.isfinite(d[c].astype(float))].copy()
    if len(d) == 0:
        d["mo_score"] = np.nan
        return d

    have_slab = slab_col in d.columns and np.isfinite(d[slab_col].astype(float)).any()
    if have_slab:
        d = d[np.isfinite(d[slab_col].astype(float))].copy()
        have_slab = len(d) > 0

    acc = d[acc_col].to_numpy(float)
    rmse = d[rmse_col].to_numpy(float)
    slab = d[slab_col].to_numpy(float) if have_slab else None

    def norm01(v: np.ndarray) -> np.ndarray:
        vmin, vmax = float(np.min(v)), float(np.max(v))
        return (v - vmin) / (vmax - vmin + 1e-12)

    acc_n = norm01(acc)
    rmse_n = norm01(rmse)

    if have_slab:
        slab_n = norm01(slab)
        dist = np.sqrt((1.0 - acc_n) ** 2 + rmse_n ** 2 + slab_n ** 2)
    else:
        dist = np.sqrt((1.0 - acc_n) ** 2 + rmse_n ** 2)

    d["mo_score"] = dist
    d.sort_values(["mo_score", "accept_rate"], ascending=[True, False], inplace=True)
    return d


def save_optionB_plots(
    df: pd.DataFrame,
    out_dir: Path,
    x_axis: str,
    y_col: str,
    best_row: pd.Series,
    best_label: Optional[str],
    color_cols: List[str],
) -> None:
    ensure_dir(out_dir)

    if x_axis == "discard_pct":
        x = 100.0 * (1.0 - df["accept_rate"].to_numpy(dtype=float))
        x_label = "Discarded samples [%]"
        x_name = "discard_pct"
        best_x = 100.0 * (1.0 - float(best_row["accept_rate"]))
    else:
        x = df["accept_rate"].to_numpy(dtype=float)
        x_label = "Accept rate"
        x_name = "accept_rate"
        best_x = float(best_row["accept_rate"])

    y = df[y_col].to_numpy(dtype=float)
    best_y = float(best_row[y_col])

    for color_col in color_cols:
        if color_col not in df.columns:
            continue
        c = df[color_col].to_numpy(dtype=float)

        plt.figure()
        sc = plt.scatter(x, y, c=c, s=35, alpha=0.9)
        plt.scatter([best_x], [best_y], marker="X", s=180, linewidths=2, edgecolors="k")
        if best_label:
            plt.annotate(best_label, (best_x, best_y),
                         textcoords="offset points", xytext=(10, 10), fontsize=9)

        plt.xlabel(x_label)
        plt.ylabel(y_col)
        plt.title(f"Trade-off: {x_name} vs {y_col} (color={color_col})")
        plt.grid(True, linewidth=0.5)
        plt.colorbar(sc, label=color_col)
        plt.savefig(out_dir / f"scatter_{x_name}_vs_{y_col}_color_{color_col}.png", dpi=220, bbox_inches="tight")
        plt.close()


def run_sweep(cfg: SweepCfg) -> None:
    npz_path = resolve_path(cfg.npz)
    if not npz_path.exists():
        raise FileNotFoundError(f"Cache not found: {npz_path}")

    out_dir = resolve_path(cfg.out_dir)
    ensure_dir(out_dir)

    data = np.load(npz_path, allow_pickle=True)
    kpts2d_pred = data["kpts2d_pred"].astype(np.float32)  # (N,K,2)
    t_gt = data["t_gt"].astype(np.float32)               # (N,3)
    q_gt = data["q_gt"].astype(np.float32)               # (N,4)
    gt_valid = data["gt_valid"].astype(np.uint8) if "gt_valid" in data.files else np.ones((kpts2d_pred.shape[0],), dtype=np.uint8)
    K = data["K"].astype(np.float32)                     # (3,3)
    kpts_3d = data["kpts_3d"].astype(np.float32)         # (K,3)

    N = int(kpts2d_pred.shape[0])
    have_gt_majority = bool(np.mean(gt_valid) > 0.5)  # only used for "objective_columns" legacy
    print(f"[SWEEP] Loaded cache: N={N} from {npz_path}")
    print(f"[SWEEP] GT available (majority): {have_gt_majority} | gt_valid_mean={float(np.mean(gt_valid)):.3f}")

    if cfg.pnp_method == "EPNP":
        pnp_flag = cv2.SOLVEPNP_EPNP
    else:
        pnp_flag = cv2.SOLVEPNP_ITERATIVE

    min_inliers_list = cfg.min_inliers_list or [6, 7, 8, 9]
    rmse_list = cfg.rmse_inliers_thr_list or [8.0, 12.0, 15.0, 20.0]
    area_list = cfg.min_kpt_area_list or [1000.0, 3000.0, 5000.0, 8000.0]
    tratio_list = cfg.t_ratio_max_list or [3.0, 5.0, 10.0, 20.0]

    # If you believe t_ratio is not meaningful on HIL, you can set it to [inf] externally via CLI.
    grid = list(itertools.product(min_inliers_list, rmse_list, area_list, tratio_list))
    print(f"[SWEEP] Grid size: {len(grid)} configs")

    rows: List[Dict[str, Any]] = []
    t0 = time.perf_counter()

    for (min_inl, rmse_thr, area_thr, tratio_max) in tqdm(grid, desc="[SWEEP] configs"):
        accepted = 0

        Et_list: List[float] = []
        et_list: List[float] = []
        eq_list: List[float] = []
        slab_list: List[float] = []

        ninl_list: List[float] = []
        rmse_inliers_list: List[float] = []

        c_pnp_fail = 0
        c_too_few_inl = 0
        c_reproj_gate = 0
        c_area_gate = 0
        c_tratio_gate = 0
        c_gt_missing = 0

        for i in range(N):
            k2d = kpts2d_pred[i]

            R, t_est, inliers = solve_pnp_ransac(
                kpts_3d=kpts_3d,
                kpts_2d=k2d,
                K=K,
                reprojection_error_px=cfg.reprojection_error_px,
                confidence=cfg.ransac_confidence,
                iterations=cfg.ransac_iterations,
                pnp_flag=pnp_flag,
            )
            if R is None or t_est is None or inliers is None:
                c_pnp_fail += 1
                continue

            idx = inliers.reshape(-1)
            ninl = int(idx.size)
            if ninl < int(min_inl):
                c_too_few_inl += 1
                continue

            proj = reprojection_points(kpts_3d, K, R, t_est)
            rmse_inl = reprojection_rmse_px(proj[idx], k2d[idx])
            if rmse_inl > float(rmse_thr):
                c_reproj_gate += 1
                continue

            xy = k2d[idx]
            w = float(xy[:, 0].max() - xy[:, 0].min())
            h = float(xy[:, 1].max() - xy[:, 1].min())
            area = float(w * h)
            if area < float(area_thr):
                c_area_gate += 1
                continue

            # Optional oracle-ish gate (ONLY if GT exists for sample i, else skip)
            if np.isfinite(tratio_max):
                if gt_valid[i] == 0:
                    c_gt_missing += 1
                else:
                    t_est_norm = float(np.linalg.norm(t_est))
                    t_gt_norm = float(np.linalg.norm(t_gt[i]) + 1e-12)
                    tratio = float(t_est_norm / t_gt_norm)
                    if tratio > float(tratio_max):
                        c_tratio_gate += 1
                        continue

            accepted += 1
            ninl_list.append(float(ninl))
            rmse_inliers_list.append(float(rmse_inl))

            # pose metrics only if GT valid for that sample
            if gt_valid[i] == 1:
                q_est_xyzw = SciRot.from_matrix(R).as_quat().astype(np.float32)
                q_est_wxyz = np.array([q_est_xyzw[3], q_est_xyzw[0], q_est_xyzw[1], q_est_xyzw[2]], dtype=np.float32)

                Et = float(np.linalg.norm(t_gt[i] - t_est))
                et = float(Et / (np.linalg.norm(t_gt[i]) + 1e-12))
                eq = float(orientation_error_rad(q_gt[i], q_est_wxyz))
                slab = float(et + eq)

                Et_list.append(Et)
                et_list.append(et)
                eq_list.append(eq)
                slab_list.append(slab)

        accept_rate = float(accepted / N) if N > 0 else 0.0

        s_ninl = summarize(np.array(ninl_list))
        s_rmse = summarize(np.array(rmse_inliers_list))
        s_Et = summarize(np.array(Et_list))
        s_et = summarize(np.array(et_list))
        s_eq = summarize(np.array(eq_list))
        s_slab = summarize(np.array(slab_list))

        rows.append({
            "min_inliers": int(min_inl),
            "rmse_inliers_thr_px": float(rmse_thr),
            "min_kpt_area_px2": float(area_thr),
            "t_ratio_max": float(tratio_max) if np.isfinite(tratio_max) else float("inf"),

            "N_total": int(N),
            "N_accepted": int(accepted),
            "accept_rate": accept_rate,
            "discard_pct": float(100.0 * (1.0 - accept_rate)),

            "ninl_mean": s_ninl["mean"], "ninl_median": s_ninl["median"], "ninl_p90": s_ninl["p90"],
            "rmse_inliers_mean": s_rmse["mean"], "rmse_inliers_median": s_rmse["median"], "rmse_inliers_p90": s_rmse["p90"],

            "Et_mean": s_Et["mean"], "Et_median": s_Et["median"], "Et_p90": s_Et["p90"],
            "et_mean": s_et["mean"], "et_median": s_et["median"], "et_p90": s_et["p90"],
            "eq_mean": s_eq["mean"], "eq_median": s_eq["median"], "eq_p90": s_eq["p90"],
            "slab_mean": s_slab["mean"], "slab_median": s_slab["median"], "slab_p90": s_slab["p90"],

            "c_pnp_fail": int(c_pnp_fail),
            "c_too_few_inl": int(c_too_few_inl),
            "c_reproj_gate": int(c_reproj_gate),
            "c_area_gate": int(c_area_gate),
            "c_tratio_gate": int(c_tratio_gate),
            "c_gt_missing": int(c_gt_missing),
        })

    dt = time.perf_counter() - t0
    print(f"[SWEEP] Done in {dt:.2f}s")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "results.csv", index=False)

    # -----------------------------
    # Selection set (guardrail + finite filtering)
    # -----------------------------
    df_sel = df.copy()
    df_sel = df_sel[np.isfinite(df_sel["accept_rate"].astype(float))]
    if cfg.min_accept_rate and cfg.min_accept_rate > 0:
        df_guard = df_sel[df_sel["accept_rate"].astype(float) >= float(cfg.min_accept_rate)]
        if len(df_guard) > 0:
            df_sel = df_guard

    if len(df_sel) == 0:
        df_sel = df.copy()

    # -----------------------------
    # BEST selection (NOW respects select_mode)
    # -----------------------------
    obj_med_col, obj_p90_col = _objective_columns(have_gt=have_gt_majority)

    if cfg.select_mode == "ideal_point":
        best, criterion = choose_ideal_point_3d_mean(df_sel)
        rank_df = rank_by_ideal_point_3d_mean(df_sel)
    elif cfg.select_mode == "weighted":
        best = choose_weighted(
            df_sel,
            accept_col="accept_rate",
            obj_med_col=obj_med_col,
            obj_p90_col=obj_p90_col,
            w_accept=float(cfg.w_accept),
            w_obj=float(cfg.w_objective),
            w_robust=float(cfg.w_robust),
        )
        criterion = f"weighted (w_accept={cfg.w_accept}, w_obj={cfg.w_objective}, w_robust={cfg.w_robust}) on normalized 2D objective"
        # ranking consistent with weighted
        rank_df = df_sel.copy()
        # compute weighted score for rank_df
        rank_df = rank_df[np.isfinite(rank_df["accept_rate"].astype(float))].copy()
        rank_df = rank_df[np.isfinite(rank_df[obj_med_col].astype(float))].copy()
        rank_df = rank_df[np.isfinite(rank_df[obj_p90_col].astype(float))].copy()
        if len(rank_df) > 0:
            acc = rank_df["accept_rate"].to_numpy(dtype=float)
            obj = rank_df[obj_med_col].to_numpy(dtype=float)
            obj90 = rank_df[obj_p90_col].to_numpy(dtype=float)

            def norm01(v: np.ndarray) -> np.ndarray:
                vmin, vmax = float(np.min(v)), float(np.max(v))
                return (v - vmin) / (vmax - vmin + 1e-12)

            acc_n = norm01(acc)
            obj_n = norm01(obj)
            obj90_n = norm01(obj90)
            score = cfg.w_accept * (1.0 - acc_n) + cfg.w_objective * obj_n + cfg.w_robust * obj90_n
            rank_df["mo_score"] = score
            rank_df.sort_values(["mo_score", obj_med_col, "accept_rate"], ascending=[True, True, False], inplace=True)
        else:
            rank_df["mo_score"] = np.nan
    else:  # pareto_knee
        best = choose_pareto_knee(df_sel, accept_col="accept_rate", obj_med_col=obj_med_col)
        criterion = "pareto_knee (closest-to-utopia on Pareto front) on 2D objective"
        rank_df = df_sel.copy()
        rank_df = rank_df[np.isfinite(rank_df["accept_rate"].astype(float))].copy()
        rank_df = rank_df[np.isfinite(rank_df[obj_med_col].astype(float))].copy()
        if len(rank_df) > 0:
            x = rank_df["accept_rate"].to_numpy(dtype=float)
            y = rank_df[obj_med_col].to_numpy(dtype=float)
            pf = pareto_front_max_min(x, y)
            rank_df["is_pareto"] = pf.astype(int)

            x_min, x_max = float(np.min(x)), float(np.max(x))
            y_min, y_max = float(np.min(y)), float(np.max(y))
            xn = (x - x_min) / (x_max - x_min + 1e-12)
            yn = (y - y_min) / (y_max - y_min + 1e-12)
            dist = np.sqrt((1.0 - xn) ** 2 + (0.0 - yn) ** 2)
            rank_df["mo_score"] = dist
            rank_df.sort_values(["is_pareto", "mo_score", obj_med_col, "accept_rate"], ascending=[False, True, True, False], inplace=True)
        else:
            rank_df["mo_score"] = np.nan

    # save top-k
    rank_df.head(int(cfg.top_k)).to_csv(out_dir / "best.csv", index=False)

    # -----------------------------
    # Plots (Option B legacy scatter) + BEST highlighted
    # -----------------------------
    color_cols = ["min_inliers", "rmse_inliers_thr_px", "t_ratio_max", "min_kpt_area_px2"]
    y_col_plot = obj_med_col if cfg.y_stat == "median" else (obj_med_col.replace("_median", "_mean") if obj_med_col.endswith("_median") else obj_med_col)
    if y_col_plot not in df.columns:
        y_col_plot = obj_med_col

    save_optionB_plots(
        df=df,
        out_dir=out_dir,
        x_axis=cfg.x_axis,
        y_col=y_col_plot,
        best_row=best,
        best_label=None,
        color_cols=color_cols,
    )

    # Save best point json (include the 3D means used by ideal-point)
    best_keys = [
        "min_inliers", "rmse_inliers_thr_px", "min_kpt_area_px2", "t_ratio_max",
        "accept_rate", "discard_pct",
        "rmse_inliers_mean", "slab_mean",
        obj_med_col, obj_p90_col,
        "ninl_mean", "ninl_median",
        "rmse_inliers_median"
    ]
    payload = {
        "selection": {
            "mode": cfg.select_mode,
            "criterion": criterion,
            "have_gt_majority": have_gt_majority,
            "min_accept_rate_guardrail": cfg.min_accept_rate,
        },
        "best": {k: (None if k not in best else (float(best[k]) if isinstance(best[k], (float, np.floating)) else int(best[k])))
                 for k in best_keys if k in best.index},
    }
    save_json(out_dir / "best_point.json", payload)

    save_json(out_dir / "sweep_config.json", {
        "npz": str(npz_path),
        "out_dir": str(out_dir),
        "have_gt_majority": have_gt_majority,
        "pnp": {
            "method": cfg.pnp_method,
            "reprojection_error_px": cfg.reprojection_error_px,
            "confidence": cfg.ransac_confidence,
            "iterations": cfg.ransac_iterations,
        },
        "grid": {
            "min_inliers_list": min_inliers_list,
            "rmse_inliers_thr_list": rmse_list,
            "min_kpt_area_list": area_list,
            "t_ratio_max_list": tratio_list,
        },
        "selection": {
            "select_mode": cfg.select_mode,
            "min_accept_rate": cfg.min_accept_rate,
            "top_k": cfg.top_k,
            "weighted": {"w_accept": cfg.w_accept, "w_objective": cfg.w_objective, "w_robust": cfg.w_robust},
            "legacy_objective_columns": {"median": obj_med_col, "p90": obj_p90_col},
        },
        "plots": {
            "x_axis": cfg.x_axis,
            "y_stat": cfg.y_stat,
            "color_plots": color_cols,
            "best_highlighted": True
        }
    })

    print("[SWEEP] Saved:")
    print(" -", (out_dir / "results.csv").resolve())
    print(" -", (out_dir / "best.csv").resolve())
    print(" -", (out_dir / "best_point.json").resolve())
    print(" -", (out_dir / "sweep_config.json").resolve())
    print(" - 4 scatter plots in:", out_dir.resolve())


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=str, default="both", choices=["cache", "sweep", "both"])

    # CACHE
    ap.add_argument("--ckpt", type=str, default="")
    ap.add_argument("--domain", type=str, default="lightbox", choices=["synthetic", "lightbox", "sunlamp"])
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--img_size", type=int, nargs=2, default=[512, 512])
    ap.add_argument("--heatmap_size", type=int, nargs=2, default=[128, 128])
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--beta_softarg", type=float, default=50.0)
    ap.add_argument("--max_samples", type=int, default=-1)
    ap.add_argument("--out_npz", type=str, default="")

    # SWEEP
    ap.add_argument("--npz", type=str, default="")
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--pnp_method", type=str, default="EPNP", choices=["EPNP", "ITERATIVE"])
    ap.add_argument("--reprojection_error_px", type=float, default=8.0)
    ap.add_argument("--confidence", type=float, default=0.999)
    ap.add_argument("--iterations", type=int, default=100)

    ap.add_argument("--min_inliers_list", type=str, default="")
    ap.add_argument("--rmse_inliers_thr_list", type=str, default="")
    ap.add_argument("--min_kpt_area_list", type=str, default="")
    ap.add_argument("--t_ratio_max_list", type=str, default="")

    # Selection
    ap.add_argument("--min_accept_rate", type=float, default=0.10)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--select_mode", type=str, default="ideal_point",
                    choices=["ideal_point", "pareto_knee", "weighted"])

    ap.add_argument("--w_accept", type=float, default=0.60)
    ap.add_argument("--w_objective", type=float, default=0.40)
    ap.add_argument("--w_robust", type=float, default=0.10)

    # Plot options (OptionB scatter)
    ap.add_argument("--x_axis", type=str, default="discard_pct", choices=["accept_rate", "discard_pct"])
    ap.add_argument("--y_stat", type=str, default="median", choices=["median", "mean"])

    args = ap.parse_args()

    out_dir = resolve_path(args.out_dir)
    ensure_dir(out_dir)

    if not args.out_npz:
        args.out_npz = str(out_dir / f"cache_{args.domain}_{args.split}.npz")

    cache_cfg = CacheCfg(
        ckpt=args.ckpt,
        domain=args.domain,
        split=args.split,
        out_npz=args.out_npz,
        img_size=(int(args.img_size[0]), int(args.img_size[1])),
        heatmap_size=(int(args.heatmap_size[0]), int(args.heatmap_size[1])),
        sigma=float(args.sigma),
        beta_softarg=float(args.beta_softarg),
        max_samples=int(args.max_samples),
    )

    sweep_cfg = SweepCfg(
        npz=(args.npz if args.npz else args.out_npz),
        out_dir=str(out_dir),
        pnp_method=args.pnp_method,
        reprojection_error_px=float(args.reprojection_error_px),
        ransac_confidence=float(args.confidence),
        ransac_iterations=int(args.iterations),
        min_inliers_list=(parse_list_int(args.min_inliers_list) if args.min_inliers_list else None),
        rmse_inliers_thr_list=(parse_list_float(args.rmse_inliers_thr_list) if args.rmse_inliers_thr_list else None),
        min_kpt_area_list=(parse_list_float(args.min_kpt_area_list) if args.min_kpt_area_list else None),
        t_ratio_max_list=(parse_list_float(args.t_ratio_max_list) if args.t_ratio_max_list else None),
        min_accept_rate=float(args.min_accept_rate),
        top_k=int(args.top_k),
        select_mode=str(args.select_mode),
        w_accept=float(args.w_accept),
        w_objective=float(args.w_objective),
        w_robust=float(args.w_robust),
        x_axis=str(args.x_axis),
        y_stat=str(args.y_stat),
    )

    if args.stage in ("cache", "both"):
        if not cache_cfg.ckpt:
            raise ValueError("--ckpt is required for stage=cache/both")
        run_cache(cache_cfg)

    if args.stage in ("sweep", "both"):
        run_sweep(sweep_cfg)


if __name__ == "__main__":
    main()
