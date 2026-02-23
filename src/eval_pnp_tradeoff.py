# src/eval_pnp_tradeoff.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import time
import json

import cv2
import numpy as np
import torch
from tqdm import tqdm
from scipy.spatial.transform import Rotation as SciRot

from src.softargmax import soft_argmax_2d, hm_to_image_coords
from src.config import (
    CHECKPOINTS_DIR,
    OUTPUTS_DIR,
    KEYPOINTS_3D_NPY,
    load_camera_matrix,
    scale_intrinsics,
)
from src.dataset_speedplus import SpeedPlusKeypointDataset
from src.model_hrnet_pose import HRNetKeypointModel


# ---------------------------
# Metrics (SPEED+/ESA)
# ---------------------------
def orientation_error_rad(q_gt_wxyz: np.ndarray, q_est_wxyz: np.ndarray) -> float:
    qg = np.asarray(q_gt_wxyz, dtype=np.float64)
    qe = np.asarray(q_est_wxyz, dtype=np.float64)
    qg = qg / (np.linalg.norm(qg) + 1e-12)
    qe = qe / (np.linalg.norm(qe) + 1e-12)
    dot = float(np.abs(np.dot(qg, qe)))
    dot = max(-1.0, min(1.0, dot))
    return float(2.0 * np.arccos(dot))


def reprojection_rmse_px(kpts_3d, kpts_2d_gt, K, R_CT, t_C, dist=None) -> float:
    rvec, _ = cv2.Rodrigues(R_CT.astype(np.float64))
    tvec = t_C.reshape(3, 1).astype(np.float64)
    proj, _ = cv2.projectPoints(
        kpts_3d.astype(np.float64), rvec, tvec, K.astype(np.float64), dist
    )
    proj = proj.reshape(-1, 2).astype(np.float32)
    diff = proj - kpts_2d_gt.astype(np.float32)
    mse = float(np.mean(np.sum(diff * diff, axis=1)))
    return float(np.sqrt(mse))


def esa_score_components(t_gt, t_est, q_gt_wxyz, q_est_wxyz):
    denom = float(np.linalg.norm(t_gt) + 1e-12)
    err_pos = float(np.linalg.norm(t_gt - t_est) / denom)
    err_ori = float(orientation_error_rad(q_gt_wxyz, q_est_wxyz))

    thr_pos = 0.002173
    thr_ori = np.deg2rad(0.169)

    score_pos = 0.0 if err_pos < thr_pos else err_pos
    score_ori = 0.0 if err_ori < thr_ori else err_ori

    score = 0.5 * (score_pos + score_ori)
    return err_pos, err_ori, score_pos, score_ori, score


def summarize(arr):
    arr = np.asarray(arr, dtype=np.float64)
    return dict(
        N=int(arr.size),
        mean=float(np.mean(arr)) if arr.size else np.nan,
        median=float(np.median(arr)) if arr.size else np.nan,
        p90=float(np.percentile(arr, 90)) if arr.size else np.nan,
    )


def summarize_times_ms(times_ms):
    times_ms = np.asarray(times_ms, dtype=np.float64)
    if times_ms.size == 0:
        return None
    mean_ms = float(times_ms.mean())
    med_ms = float(np.median(times_ms))
    p90_ms = float(np.percentile(times_ms, 90))
    fps_mean = 1000.0 / mean_ms if mean_ms > 0 else float("inf")
    fps_med = 1000.0 / med_ms if med_ms > 0 else float("inf")
    return mean_ms, med_ms, p90_ms, fps_mean, fps_med


# ---------------------------
# Solver wrappers
# ---------------------------
P3P_FLAGS = {cv2.SOLVEPNP_P3P, cv2.SOLVEPNP_AP3P}


def solve_pnp_one(method_flag: int,
                  pts3d: np.ndarray,
                  pts2d: np.ndarray,
                  K: np.ndarray,
                  dist=None):
    """
    Returns: ok, R_CT, t_C, rmse
    Handles P3P/AP3P via solvePnPGeneric.
    """
    pts3d = np.asarray(pts3d, dtype=np.float64)
    pts2d = np.asarray(pts2d, dtype=np.float64)

    if pts2d.shape[0] < 4:
        return False, None, None, np.nan, "too_few_points"

    # P3P/AP3P require exactly 4 points
    if method_flag in P3P_FLAGS:
        if pts2d.shape[0] != 4:
            return False, None, None, np.nan, "p3p_needs_4_points"

        try:
            ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                pts3d, pts2d, K, dist, flags=int(method_flag)
            )
        except cv2.error as e:
            return False, None, None, np.nan, f"opencv_error:{str(e).splitlines()[-1]}"

        if (not ok) or (rvecs is None) or (tvecs is None) or (len(rvecs) == 0):
            return False, None, None, np.nan, "no_solution"

        best_rmse = np.inf
        best_R, best_t = None, None
        for rvec, tvec in zip(rvecs, tvecs):
            R_CT, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
            t_C = np.asarray(tvec, dtype=np.float64).reshape(3)
            rmse = reprojection_rmse_px(pts3d, pts2d, K, R_CT, t_C, dist)
            if rmse < best_rmse:
                best_rmse = rmse
                best_R, best_t = R_CT, t_C

        return True, best_R, best_t, float(best_rmse), "ok"

    # General case
    try:
        ok, rvec, tvec = cv2.solvePnP(
            pts3d, pts2d, K, dist, flags=int(method_flag))
    except cv2.error as e:
        return False, None, None, np.nan, f"opencv_error:{str(e).splitlines()[-1]}"

    if not ok:
        return False, None, None, np.nan, "no_solution"

    R_CT, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    t_C = np.asarray(tvec, dtype=np.float64).reshape(3)
    rmse = reprojection_rmse_px(pts3d, pts2d, K, R_CT, t_C, dist)
    return True, R_CT, t_C, float(rmse), "ok"


def refine_iterative(R_init, t_init, pts3d, pts2d, K, dist=None):
    rvec_init, _ = cv2.Rodrigues(R_init.astype(np.float64))
    tvec_init = t_init.reshape(3, 1).astype(np.float64)

    try:
        ok, rvec, tvec = cv2.solvePnP(
            pts3d.astype(np.float64),
            pts2d.astype(np.float64),
            K.astype(np.float64),
            dist,
            rvec=rvec_init,
            tvec=tvec_init,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
    except cv2.error:
        return False, None, None

    if not ok:
        return False, None, None

    R_CT, _ = cv2.Rodrigues(rvec)
    return True, R_CT, tvec.reshape(3)


# ---------------------------
# Keypoint selection
# ---------------------------
def select_topk(kpts2d: np.ndarray, conf: np.ndarray, k: int):
    """
    Return top-k points by conf. If k<=0 or k>=K -> return all.
    """
    K = kpts2d.shape[0]
    if (k is None) or (k <= 0) or (k >= K):
        idx = np.arange(K)
    else:
        idx = np.argsort(-conf)[:k]
    return idx


# ---------------------------
# Methods to compare
# ---------------------------
def build_methods():
    # Note: DLS/UPNP often fallback to EPnP internally.
    methods = {
        "ITERATIVE": cv2.SOLVEPNP_ITERATIVE,
        "EPNP": cv2.SOLVEPNP_EPNP,
        "SQPNP": cv2.SOLVEPNP_SQPNP,
        "P3P": cv2.SOLVEPNP_P3P,
        "AP3P": cv2.SOLVEPNP_AP3P,
        "IPPE": cv2.SOLVEPNP_IPPE,  # valid mainly for coplanar sets
        # "DLS": cv2.SOLVEPNP_DLS,
        # "UPNP": cv2.SOLVEPNP_UPNP,
    }
    return methods


# ---------------------------
# Main
# ---------------------------
def main():
    # ------------------------ config
    split = "validation"
    domain = "synthetic"
    max_samples = 2000          # set to len(ds) for full eval
    topk = 11                   # set e.g. 8, 6, 4
    do_refine = True            # refine with ITERATIVE after initial solution
    refine_min_pts = 6          # don’t refine if too few points

    # ------------------------ device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available())
    print("Using device:", device)

    # ------------------------ dataset
    ds = SpeedPlusKeypointDataset(
        split=split,
        domain=domain,
        img_size=(512, 512),
        heatmap_size=(128, 128),
        sigma=2.0,
        return_heatmaps=True,
    )
    n_eval = min(len(ds), max_samples)

    # ------------------------ 3D kpts
    kpts_3d_all = np.load(KEYPOINTS_3D_NPY).astype(np.float32)
    num_kpts = int(kpts_3d_all.shape[0])
    print("Num keypoints:", num_kpts)

    # ------------------------ intrinsics
    K0, W0, H0 = load_camera_matrix()
    K = scale_intrinsics(K0, (W0, H0), (512, 512)).astype(np.float32)
    dist = None

    # ------------------------ model
    model = HRNetKeypointModel(num_keypoints=num_kpts).to(device)
    ckpt_path = Path(CHECKPOINTS_DIR) / "hrnet_kpts_best.pth"
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state"] if isinstance(
        ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    # ------------------------ methods
    methods = build_methods()

    # per-method accumulators
    per = {}
    for m in methods:
        per[m] = dict(
            E_t=[],
            e_t=[],
            e_q=[],
            rmse_px=[],
            esa=[],
            esa_pos=[],
            esa_ori=[],
            attempted_ms=[],
            success_ms=[],
            ok=0,
            fail=0,
            skipped=0,
        )

    amp_device = "cuda" if device.type == "cuda" else "cpu"

    # ------------------------ loop
    for i in tqdm(range(n_eval), desc=f"PnP tradeoff ({split}/{domain})"):
        sample = ds[i]

        img = sample["image"].unsqueeze(0).to(device)
        q_gt_wxyz = sample["q"] if isinstance(
            sample["q"], np.ndarray) else sample["q"].detach().cpu().numpy()
        t_gt = sample["t"] if isinstance(
            sample["t"], np.ndarray) else sample["t"].detach().cpu().numpy()
        kpts_2d_gt = sample["kpts_2d"] if isinstance(
            sample["kpts_2d"], np.ndarray) else sample["kpts_2d"].detach().cpu().numpy()

        # predict heatmaps
        with torch.no_grad():
            with torch.amp.autocast(amp_device, enabled=(device.type == "cuda")):
                hmaps_t = model(img)  # (1,K,Hm,Wm)

        # soft-argmax keypoints + conf
        coords_hm, conf = soft_argmax_2d(hmaps_t, beta=50.0)
        coords_img = hm_to_image_coords(
            coords_hm,
            heatmap_size=(hmaps_t.shape[-2], hmaps_t.shape[-1]),
            img_size=(512, 512),
        )
        kpts_2d_pred_all = coords_img[0].detach(
        ).cpu().numpy().astype(np.float32)  # (K,2)
        conf_all = conf[0].detach().cpu().numpy().astype(
            np.float32)               # (K,)

        # choose subset
        idx = select_topk(kpts_2d_pred_all, conf_all, k=topk)
        kpts_2d_pred = kpts_2d_pred_all[idx]
        kpts_3d = kpts_3d_all[idx]

        # for P3P/AP3P we must use exactly 4 points
        # so we will dynamically take the top-4 subset when needed
        for mname, flag in methods.items():
            t0 = time.perf_counter()

            pts2d_use = kpts_2d_pred
            pts3d_use = kpts_3d

            # Special handling: P3P/AP3P need exactly 4 points
            if flag in P3P_FLAGS:
                if pts2d_use.shape[0] < 4:
                    per[mname]["skipped"] += 1
                    continue
                # top 4 by confidence among selected idx
                order = np.argsort(-conf_all[idx])
                top4 = order[:4]
                pts2d_use = pts2d_use[top4]
                pts3d_use = pts3d_use[top4]

            ok, R_CT, t_C, rmse, status = solve_pnp_one(
                flag, pts3d_use, pts2d_use, K, dist)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            per[mname]["attempted_ms"].append(dt_ms)

            if not ok:
                per[mname]["fail"] += 1
                continue

            # optional refinement (LM) for all methods (including P3P/AP3P) if enough points
            if do_refine and pts2d_use.shape[0] >= refine_min_pts:
                ok_ref, Rr, tr = refine_iterative(
                    R_CT, t_C, pts3d_use, pts2d_use, K, dist)
                if ok_ref:
                    R_CT, t_C = Rr, tr
                    rmse = reprojection_rmse_px(
                        pts3d_use, pts2d_use, K, R_CT, t_C, dist)

            per[mname]["success_ms"].append(dt_ms)
            per[mname]["ok"] += 1

            # NOTE: you already verified GT convention => q is R_CT (Camera <- Tango)
            q_est_xyzw = SciRot.from_matrix(
                R_CT).as_quat().astype(np.float32)  # xyzw
            q_est_wxyz = np.array(
                [q_est_xyzw[3], q_est_xyzw[0], q_est_xyzw[1], q_est_xyzw[2]], dtype=np.float32)

            Et = float(np.linalg.norm(t_gt - t_C))
            et = float(Et / (np.linalg.norm(t_gt) + 1e-12))
            eq = float(orientation_error_rad(q_gt_wxyz, q_est_wxyz))

            # compute RMSE w.r.t. GT 2D (all points, not subset)
            # (fair: compares pose to the full GT keypoints set)
            rmse_full = reprojection_rmse_px(
                kpts_3d_all, kpts_2d_gt, K, R_CT, t_C, dist)

            err_pos, err_ori, sc_pos, sc_ori, sc = esa_score_components(
                t_gt, t_C, q_gt_wxyz, q_est_wxyz)

            per[mname]["E_t"].append(Et)
            per[mname]["e_t"].append(et)
            per[mname]["e_q"].append(eq)
            per[mname]["rmse_px"].append(rmse_full)
            per[mname]["esa"].append(sc)
            per[mname]["esa_pos"].append(sc_pos)
            per[mname]["esa_ori"].append(sc_ori)

    # ------------------------ summary print + save
    out_dir = Path(OUTPUTS_DIR) / "pnp_tradeoff"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    print("\n=== PnP SOLVER TRADEOFF SUMMARY ===")
    print(
        f"Eval: {split}/{domain}  n_eval={n_eval}  topk={topk}  refine={do_refine}")

    for mname in methods.keys():
        ok = per[mname]["ok"]
        fail = per[mname]["fail"]
        attempted = len(per[mname]["attempted_ms"])
        fail_rate = (fail / attempted) if attempted > 0 else np.nan

        s_esa = summarize(per[mname]["esa"])
        s_eq = summarize(per[mname]["e_q"])
        s_et = summarize(per[mname]["e_t"])
        s_rmse = summarize(per[mname]["rmse_px"])
        tA = summarize_times_ms(per[mname]["attempted_ms"])
        tS = summarize_times_ms(per[mname]["success_ms"])

        row = dict(
            method=mname,
            ok=ok,
            fail=fail,
            attempted=attempted,
            fail_rate_attempted=float(
                fail_rate) if np.isfinite(fail_rate) else None,

            median_esa=s_esa["median"],
            p90_esa=s_esa["p90"],

            median_e_q_rad=s_eq["median"],
            p90_e_q_rad=s_eq["p90"],

            median_e_t=s_et["median"],
            p90_e_t=s_et["p90"],

            median_rmse_px=s_rmse["median"],
            p90_rmse_px=s_rmse["p90"],
        )

        if tS is not None:
            row.update(
                median_ms_success=tS[1],
                mean_ms_success=tS[0],
                p90_ms_success=tS[2],
                fps_median_success=tS[4],
                fps_mean_success=tS[3],
            )
        else:
            row.update(
                median_ms_success=None,
                mean_ms_success=None,
                p90_ms_success=None,
                fps_median_success=None,
                fps_mean_success=None,
            )

        rows.append(row)

        # console friendly
        tmed = row["median_ms_success"]
        fpsm = row["fps_median_success"]
        print(
            f"{mname:10s} | ok={ok:5d} fail={fail:5d} fail_rate={fail_rate:6.3f} | "
            f"ESA_med={row['median_esa']:.4f} | e_q_med={row['median_e_q_rad']:.4f} rad | "
            f"time_med={tmed:.3f} ms ({fpsm:.1f} FPS)" if tmed is not None else
            f"{mname:10s} | ok={ok:5d} fail={fail:5d} fail_rate={fail_rate:6.3f} | "
            f"ESA_med={row['median_esa']:.4f} | e_q_med={row['median_e_q_rad']:.4f} rad | time_med=NA"
        )

    # save JSON
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2))

    # save CSV
    csv_path = out_dir / "summary.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("\nSaved:")
    print(" -", out_dir / "summary.json")
    print(" -", out_dir / "summary.csv")


if __name__ == "__main__":
    main()
