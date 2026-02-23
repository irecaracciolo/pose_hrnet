#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare_eval_subsets.py (v2 robust)

Compare 3 runs stored as per_sample.json (e.g., HM-only vs coord-fixed vs coord-warmup).

Key features:
- Robust JSON parsing:
    * if top-level has a nested dict under common keys (per_sample/samples/results/...)
      we use that dict as the sample map.
    * if top-level is a list, we build a dict using an ID field (key/id/img_name/name/stem).
- Coverage:
    * "pose-valid" requires ALL pose_metrics to be finite for that sample.
- FAIR metrics:
    * for each metric, evaluate only samples valid in ALL runs.
- Coverage-aware penalized metrics on FULL intersection:
    * missing/invalid per-sample metric gets a run+metric specific penalty
      penalty = q-quantile(valid) * penalty_mult
    * if no valid values exist, use a large finite fallback penalty (no inf).

Outputs:
- Console report
- plots/coverage.png
- plots/penalized_boxplots.png
- plots/penalized_medians.png
- table_main.tex (LaTeX-ready)
- summary.json

Assumes "lower is better" for all metrics (including ESA if your eval defines it as an error).
"""

from __future__ import annotations
import matplotlib.pyplot as plt

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")


# ----------------------------
# JSON parsing (ROBUST)
# ----------------------------
NESTED_CANDIDATES = [
    "per_sample",
    "perSample",
    "samples",
    "results",
    "preds",
    "data",
    "items",
]


ID_CANDIDATES = [
    "key",
    "id",
    "img_key",
    "img_id",
    "img_name",
    "image",
    "name",
    "stem",
    "sample_id",
    "sample",
    "uid",
]


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_per_sample(obj: Any, source_name: str = "") -> Dict[str, Any]:
    """
    Convert whatever JSON structure into:
        dict[str(sample_key)] -> dict(metrics...)
    """
    # Case 1) already dict that looks like sample map
    if isinstance(obj, dict):
        # If it contains one of the nested candidates as dict, use that
        for k in NESTED_CANDIDATES:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]

        # If it contains one of the nested candidates as list, convert list->dict
        for k in NESTED_CANDIDATES:
            if k in obj and isinstance(obj[k], list):
                return list_to_sample_map(obj[k], source_name=source_name)

        # Heuristic: if many keys look like samples (not meta/config)
        # If it has at least, say, 50 keys and values are dicts, assume it's sample map.
        if len(obj) >= 50 and all(isinstance(v, dict) for v in obj.values()):
            return obj

        # Otherwise, last attempt: if it has exactly one big dict inside, pick it
        dict_children = [v for v in obj.values() if isinstance(v, dict)]
        if len(dict_children) == 1 and len(dict_children[0]) >= 50:
            return dict_children[0]

        # If nothing worked, return as "single key" map -> will show warning
        return obj

    # Case 2) list of entries
    if isinstance(obj, list):
        return list_to_sample_map(obj, source_name=source_name)

    # Fallback
    return {"__single__": obj}


def list_to_sample_map(items: List[Any], source_name: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for idx, it in enumerate(items):
        if not isinstance(it, dict):
            out[f"idx_{idx:06d}"] = {"value": it}
            continue

        # find id key
        sid: Optional[str] = None
        for k in ID_CANDIDATES:
            if k in it:
                v = it.get(k)
                if v is not None:
                    sid = str(v)
                    break

        if sid is None:
            # last resort: if it has "meta" and "metrics", maybe meta contains name
            if "meta" in it and isinstance(it["meta"], dict):
                for k in ID_CANDIDATES:
                    if k in it["meta"]:
                        sid = str(it["meta"][k])
                        break

        if sid is None:
            sid = f"idx_{idx:06d}"

        out[sid] = it

    return out


# ----------------------------
# Metric extraction / validity
# ----------------------------
def to_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return v


ALIASES = {
    "rmse_px": ["rmse_px", "rmse", "rmse_pix", "rmse_pixels"],
    "eq_rad": ["eq_rad", "eq", "e_q_rad", "rot_err_rad"],
    "et": ["et", "e_t", "et_norm", "trans_err_norm"],
    "Et_m": ["Et_m", "Et", "Et_meter", "Et_meters", "trans_err_m"],
    "esa": ["esa", "ESA", "e_sa"],
}


def extract_metric(entry: Dict[str, Any], metric: str) -> float | None:
    if not isinstance(entry, dict):
        return None

    # direct
    if metric in entry:
        return to_float(entry.get(metric))

    # nested common patterns: sometimes metrics are under "metrics" or "vals"
    for nest_key in ["metrics", "vals", "values", "errors"]:
        if nest_key in entry and isinstance(entry[nest_key], dict):
            if metric in entry[nest_key]:
                return to_float(entry[nest_key].get(metric))

    # aliases
    if metric in ALIASES:
        for k in ALIASES[metric]:
            if k in entry:
                return to_float(entry.get(k))
            for nest_key in ["metrics", "vals", "values", "errors"]:
                if nest_key in entry and isinstance(entry[nest_key], dict) and k in entry[nest_key]:
                    return to_float(entry[nest_key].get(k))

    return None


def is_valid(v: float | None) -> bool:
    return (v is not None) and np.isfinite(v)


def valid_for_pose(entry: Dict[str, Any], pose_metrics: List[str]) -> bool:
    return all(is_valid(extract_metric(entry, m)) for m in pose_metrics)


# ----------------------------
# Stats
# ----------------------------
def summarize(vals: np.ndarray) -> Dict[str, Any]:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"N": 0, "mean": None, "median": None, "p90": None}
    return {
        "N": int(vals.size),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p90": float(np.quantile(vals, 0.90)),
    }


def fmt(x: Any, nd: int = 6) -> str:
    if x is None:
        return "-"
    try:
        v = float(x)
        if not np.isfinite(v):
            return "-"
        return f"{v:.{nd}f}"
    except Exception:
        return "-"


# ----------------------------
# Penalization (finite always)
# ----------------------------
def compute_penalty(valid_vals: np.ndarray, q: float, mult: float) -> float:
    valid_vals = valid_vals[np.isfinite(valid_vals)]
    if valid_vals.size == 0:
        # large finite fallback (avoids inf in deltas)
        return 1e6
    base = float(np.quantile(valid_vals, q))
    # still guard against 0 -> keep >0
    pen = max(base * mult, 1e-12)
    # also guard against insane penalties
    return float(min(pen, 1e9))


def penalized_values(keys: List[str], run: Dict[str, Any], metric: str, penalty: float) -> np.ndarray:
    out = np.empty(len(keys), dtype=np.float64)
    for i, k in enumerate(keys):
        ent = run.get(k, {})
        v = extract_metric(ent, metric) if isinstance(ent, dict) else None
        out[i] = float(v) if is_valid(v) else float(penalty)
    return out


# ----------------------------
# Helpers
# ----------------------------
def collect_keys(*runs: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    sets = [set(r.keys()) for r in runs]
    keys_union = sorted(set().union(*sets))
    keys_inter = sorted(set.intersection(*sets))
    return keys_union, keys_inter


def get_valid_keys_for_metric(run: Dict[str, Any], keys: List[str], metric: str) -> List[str]:
    out = []
    for k in keys:
        ent = run.get(k, {})
        v = extract_metric(ent, metric) if isinstance(ent, dict) else None
        if is_valid(v):
            out.append(k)
    return out


def get_values_for_metric(run: Dict[str, Any], keys: List[str], metric: str) -> np.ndarray:
    vals: List[float] = []
    for k in keys:
        ent = run.get(k, {})
        v = extract_metric(ent, metric) if isinstance(ent, dict) else None
        if is_valid(v):
            vals.append(float(v))
    return np.asarray(vals, dtype=np.float64)


def count_wins(deltas: np.ndarray) -> Tuple[int, int, int]:
    d = deltas[np.isfinite(deltas)]
    winsB = int(np.sum(d < 0.0))  # lower is better -> B wins if delta < 0
    winsA = int(np.sum(d > 0.0))
    ties = int(np.sum(d == 0.0))
    return winsB, winsA, ties


# ----------------------------
# Plots
# ----------------------------
def save_coverage_plot(out_png: Path, labels: List[str], valid_counts: List[int], total: int):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cov = [c / max(total, 1) for c in valid_counts]
    plt.figure(figsize=(7.0, 4.2))
    plt.bar(labels, [100.0 * v for v in cov])
    plt.ylabel("Valid coverage [%]")
    plt.title("Coverage on common key intersection")
    plt.ylim(0, 100)
    for i, (c, v) in enumerate(zip(valid_counts, cov)):
        plt.text(i, 100.0 * v + 1.2, f"{100.0*v:.2f}%\n({c}/{total})",
                 ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def save_penalized_boxplots(out_png: Path, labels: List[str],
                            penalized_by_run_metric: Dict[Tuple[str, str], np.ndarray],
                            metric_names: List[str]):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    n = len(metric_names)
    plt.figure(figsize=(9.0, 2.6 * n))
    for i, m in enumerate(metric_names, start=1):
        plt.subplot(n, 1, i)
        vals = []
        for lab in labels:
            v = penalized_by_run_metric[(lab, m)]
            v = v[np.isfinite(v)]
            vals.append(v)
        plt.boxplot(vals, tick_labels=labels, showfliers=False)
        plt.ylabel(m)
        plt.grid(True, axis="y", alpha=0.3)
        if i == 1:
            plt.title("Penalized metrics on FULL intersection (lower is better)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def save_penalized_medians_bar(out_png: Path, labels: List[str],
                               penalized_medians: Dict[Tuple[str, str], float],
                               metric_names: List[str]):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(metric_names))
    w = 0.26
    plt.figure(figsize=(10.5, 4.6))
    for j, lab in enumerate(labels):
        ys = [penalized_medians[(lab, m)] for m in metric_names]
        plt.bar(x + (j - 1) * w, ys, width=w, label=lab)
    plt.xticks(x, metric_names)
    plt.ylabel("Penalized median")
    plt.title("Penalized medians on FULL intersection")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


# ----------------------------
# LaTeX
# ----------------------------
def latex_table_main(out_tex: Path,
                     labels: List[str],
                     cov_counts: Dict[str, int],
                     total: int,
                     fair_stats: Dict[str, Dict[str, Dict[str, Any]]],
                     pen_stats: Dict[str, Dict[str, Dict[str, Any]]],
                     metric_names: List[str],
                     caption: str,
                     label: str):
    out_tex.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{6pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    colspec = "l c " + " ".join(["c c"] * len(metric_names))
    lines.append(rf"\begin{{tabular}}{{{colspec}}}")
    lines.append(r"\toprule")
    lines.append(
        r"\multirow{2}{*}{Run} & \multirow{2}{*}{Coverage [\%]} & "
        + " & ".join([rf"\multicolumn{{2}}{{c}}{{{m}}}" for m in metric_names])
        + r" \\"
    )
    lines.append(
        r" & & "
        + " & ".join([r"\textit{FAIR} med & \textit{Pen.} med" for _ in metric_names])
        + r" \\"
    )
    lines.append(r"\midrule")

    for lab in labels:
        cov = cov_counts[lab] / max(total, 1)
        row = [lab, f"{100.0*cov:.2f}"]
        for m in metric_names:
            fmed = fair_stats[lab][m]["median"] if fair_stats[lab][m]["N"] else None
            pmed = pen_stats[lab][m]["median"] if pen_stats[lab][m]["N"] else None
            row.append(fmt(fmed, 4))
            row.append(fmt(pmed, 4))
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table}")

    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--c", required=True)
    ap.add_argument("--name_a", default="A")
    ap.add_argument("--name_b", default="B")
    ap.add_argument("--name_c", default="C")
    ap.add_argument("--metrics", default="Et_m,et,eq_rad,rmse_px,esa")
    ap.add_argument("--pose_metrics", default="Et_m,et,eq_rad,esa")
    ap.add_argument("--penalty_q", type=float, default=0.90)
    ap.add_argument("--penalty_mult", type=float, default=3.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--latex_label", default="tab:hrnet_loss_ablation")
    ap.add_argument("--latex_caption",
                    default="Ablation study comparing three training configurations.")
    args = ap.parse_args()

    path_a = Path(args.a)
    path_b = Path(args.b)
    path_c = Path(args.c)

    la, lb, lc = args.name_a, args.name_b, args.name_c
    metric_names = [s.strip() for s in args.metrics.split(",") if s.strip()]
    pose_metrics = [s.strip()
                    for s in args.pose_metrics.split(",") if s.strip()]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=== INPUTS ===")
    print(f"{la}: {path_a}")
    print(f"{lb}: {path_b}")
    print(f"{lc}: {path_c}")
    print()

    raw_a = load_json(path_a)
    raw_b = load_json(path_b)
    raw_c = load_json(path_c)

    ra = normalize_per_sample(raw_a, source_name=la)
    rb = normalize_per_sample(raw_b, source_name=lb)
    rc = normalize_per_sample(raw_c, source_name=lc)

    # sanity print
    print("=== PARSING SANITY ===")
    print(f"{la}: top-level type={type(raw_a).__name__}, normalized samples={len(ra)}")
    print(f"{lb}: top-level type={type(raw_b).__name__}, normalized samples={len(rb)}")
    print(f"{lc}: top-level type={type(raw_c).__name__}, normalized samples={len(rc)}")
    print()

    keys_union, keys_inter = collect_keys(ra, rb, rc)

    print("=== SET SIZES ===")
    print(f"keys_union: {len(keys_union)}")
    print(f"keys_intersection: {len(keys_inter)}")
    print()

    if len(keys_inter) < 10:
        print("[WARN] Intersection is very small. This usually means JSON parsing did not find the sample map.")
        print("       Inspect one per_sample.json: maybe metrics are nested under a different key.")
        print()

    # Coverage
    valid_a_pose = [k for k in keys_inter if valid_for_pose(
        ra.get(k, {}), pose_metrics)]
    valid_b_pose = [k for k in keys_inter if valid_for_pose(
        rb.get(k, {}), pose_metrics)]
    valid_c_pose = [k for k in keys_inter if valid_for_pose(
        rc.get(k, {}), pose_metrics)]

    print("=== COVERAGE (valid for pose metrics) ===")
    print(f"{la} valid: {len(valid_a_pose)} / {len(keys_inter)} = {100.0*len(valid_a_pose)/max(len(keys_inter), 1):.2f}%")
    print(f"{lb} valid: {len(valid_b_pose)} / {len(keys_inter)} = {100.0*len(valid_b_pose)/max(len(keys_inter), 1):.2f}%")
    print(f"{lc} valid: {len(valid_c_pose)} / {len(keys_inter)} = {100.0*len(valid_c_pose)/max(len(keys_inter), 1):.2f}%")
    print()

    # FAIR per metric
    fair_common_by_metric: Dict[str, List[str]] = {}
    for m in metric_names:
        va = set(get_valid_keys_for_metric(ra, keys_inter, m))
        vb = set(get_valid_keys_for_metric(rb, keys_inter, m))
        vc = set(get_valid_keys_for_metric(rc, keys_inter, m))
        fair_common_by_metric[m] = sorted(list(va & vb & vc))

    # A single headline fair set for pose_metrics
    pose_fair = set(keys_inter)
    for m in pose_metrics:
        pose_fair &= set(fair_common_by_metric.get(m, []))
    pose_fair = sorted(list(pose_fair))

    print("=== FAIR SUBSET (common valid) ===")
    print(
        f"common_valid_pose: {len(pose_fair)} / {len(keys_inter)} = {100.0*len(pose_fair)/max(len(keys_inter), 1):.2f}% of intersection")
    print()

    # Own valid summaries
    print("=== METRICS SUMMARY (OWN VALID SET) ===")
    for m in metric_names:
        va = get_values_for_metric(
            ra, get_valid_keys_for_metric(ra, keys_inter, m), m)
        vb = get_values_for_metric(
            rb, get_valid_keys_for_metric(rb, keys_inter, m), m)
        vc = get_values_for_metric(
            rc, get_valid_keys_for_metric(rc, keys_inter, m), m)
        sa, sb, sc = summarize(va), summarize(vb), summarize(vc)
        print(
            f"{m:<8} | {la}: N={sa['N']}, med={fmt(sa['median'], 12)} | {lb}: N={sb['N']}, med={fmt(sb['median'], 12)} | {lc}: N={sc['N']}, med={fmt(sc['median'], 12)}")
    print()

    # FAIR summaries
    fair_stats: Dict[str, Dict[str, Dict[str, Any]]] = {la: {}, lb: {}, lc: {}}
    print("=== METRICS SUMMARY (COMMON VALID) [FAIR] ===")
    for m in metric_names:
        kf = fair_common_by_metric[m]
        va = np.asarray([extract_metric(ra[k], m)
                        for k in kf], dtype=np.float64)
        vb = np.asarray([extract_metric(rb[k], m)
                        for k in kf], dtype=np.float64)
        vc = np.asarray([extract_metric(rc[k], m)
                        for k in kf], dtype=np.float64)
        va, vb, vc = va[np.isfinite(va)], vb[np.isfinite(
            vb)], vc[np.isfinite(vc)]
        sa, sb, sc = summarize(va), summarize(vb), summarize(vc)
        fair_stats[la][m], fair_stats[lb][m], fair_stats[lc][m] = sa, sb, sc
        print(
            f"{m:<8} | "
            f"{la}: N={sa['N']}, mean={fmt(sa['mean'], 6)}, med={fmt(sa['median'], 6)}, p90={fmt(sa['p90'], 6)} | "
            f"{lb}: mean={fmt(sb['mean'], 6)}, med={fmt(sb['median'], 6)}, p90={fmt(sb['p90'], 6)} | "
            f"{lc}: mean={fmt(sc['mean'], 6)}, med={fmt(sc['median'], 6)}, p90={fmt(sc['p90'], 6)}"
        )
    print()

    # Paired FAIR comparisons
    def paired_fair(nameX: str, runX: Dict[str, Any], nameY: str, runY: Dict[str, Any]):
        print(f"=== PAIRED COMPARISON ON FAIR ( {nameY} - {nameX} ) ===")
        for m in metric_names:
            kf = fair_common_by_metric[m]
            d = []
            for k in kf:
                vx = extract_metric(runX[k], m)
                vy = extract_metric(runY[k], m)
                if is_valid(vx) and is_valid(vy):
                    d.append(float(vy) - float(vx))
            d = np.asarray(d, dtype=np.float64)
            sd = summarize(d)
            winsY, winsX, ties = count_wins(d)
            print(
                f"{m:<8} delta({nameY}-{nameX}): N={sd['N']} mean={fmt(sd['mean'], 6)} "
                f"med={fmt(sd['median'], 6)} p90={fmt(sd['p90'], 6)} | "
                f"wins {nameY}={winsY}, wins {nameX}={winsX}, ties={ties}"
            )
        print()

    paired_fair(la, ra, lb, rb)
    paired_fair(la, ra, lc, rc)
    paired_fair(lb, rb, lc, rc)

    # Penalized full intersection
    print("=== COVERAGE-AWARE (PENALIZED) ON FULL INTERSECTION ===")
    print(
        f"Penalty: q={args.penalty_q:.2f} of valid * mult={args.penalty_mult:.2f} (computed per metric, per run)")
    print()

    penalties: Dict[str, Dict[str, float]] = {la: {}, lb: {}, lc: {}}
    penalized_by_run_metric: Dict[Tuple[str, str], np.ndarray] = {}
    pen_stats: Dict[str, Dict[str, Dict[str, Any]]] = {la: {}, lb: {}, lc: {}}
    penalized_medians: Dict[Tuple[str, str], float] = {}

    for m in metric_names:
        va_valid = get_values_for_metric(
            ra, get_valid_keys_for_metric(ra, keys_inter, m), m)
        vb_valid = get_values_for_metric(
            rb, get_valid_keys_for_metric(rb, keys_inter, m), m)
        vc_valid = get_values_for_metric(
            rc, get_valid_keys_for_metric(rc, keys_inter, m), m)

        pa = compute_penalty(va_valid, args.penalty_q, args.penalty_mult)
        pb = compute_penalty(vb_valid, args.penalty_q, args.penalty_mult)
        pc = compute_penalty(vc_valid, args.penalty_q, args.penalty_mult)

        penalties[la][m], penalties[lb][m], penalties[lc][m] = pa, pb, pc

        a_pen = penalized_values(keys_inter, ra, m, pa)
        b_pen = penalized_values(keys_inter, rb, m, pb)
        c_pen = penalized_values(keys_inter, rc, m, pc)

        penalized_by_run_metric[(la, m)] = a_pen
        penalized_by_run_metric[(lb, m)] = b_pen
        penalized_by_run_metric[(lc, m)] = c_pen

        sa, sb, sc = summarize(a_pen), summarize(b_pen), summarize(c_pen)
        pen_stats[la][m], pen_stats[lb][m], pen_stats[lc][m] = sa, sb, sc

        penalized_medians[(la, m)] = float(
            sa["median"]) if sa["median"] is not None else 1e9
        penalized_medians[(lb, m)] = float(
            sb["median"]) if sb["median"] is not None else 1e9
        penalized_medians[(lc, m)] = float(
            sc["median"]) if sc["median"] is not None else 1e9

    print("METRIC (penalized) summaries on FULL intersection:")
    for m in metric_names:
        sa, sb, sc = pen_stats[la][m], pen_stats[lb][m], pen_stats[lc][m]
        print(
            f"{m:<8} | "
            f"{la}: N={len(keys_inter)} mean={fmt(sa['mean'], 6)} med={fmt(sa['median'], 6)} p90={fmt(sa['p90'], 6)} | "
            f"{lb}: mean={fmt(sb['mean'], 6)} med={fmt(sb['median'], 6)} p90={fmt(sb['p90'], 6)} | "
            f"{lc}: mean={fmt(sc['mean'], 6)} med={fmt(sc['median'], 6)} p90={fmt(sc['p90'], 6)} | "
            f"penalties: {la}={fmt(penalties[la][m], 6)}, {lb}={fmt(penalties[lb][m], 6)}, {lc}={fmt(penalties[lc][m], 6)}"
        )
    print()

    def paired_pen(nameX: str, nameY: str):
        print(
            f"=== PAIRED COMPARISON ON PENALIZED FULL INTERSECTION ({nameY} - {nameX}) ===")
        for m in metric_names:
            x = penalized_by_run_metric[(nameX, m)]
            y = penalized_by_run_metric[(nameY, m)]
            d = (y - x)
            sd = summarize(d)
            winsY, winsX, ties = count_wins(d)
            print(
                f"{m:<8} delta({nameY}-{nameX}): N={sd['N']} mean={fmt(sd['mean'], 6)} "
                f"med={fmt(sd['median'], 6)} p90={fmt(sd['p90'], 6)} | "
                f"wins {nameY}={winsY}, wins {nameX}={winsX}, ties={ties}"
            )
        print()

    paired_pen(la, lb)
    paired_pen(la, lc)
    paired_pen(lb, lc)

    # Save plots
    cov_counts = {la: len(valid_a_pose), lb: len(
        valid_b_pose), lc: len(valid_c_pose)}
    save_coverage_plot(plots_dir / "coverage.png", [la, lb, lc],
                       [cov_counts[la], cov_counts[lb], cov_counts[lc]], total=len(keys_inter))
    save_penalized_boxplots(plots_dir / "penalized_boxplots.png", [la, lb, lc],
                            penalized_by_run_metric, metric_names)
    save_penalized_medians_bar(plots_dir / "penalized_medians.png", [la, lb, lc],
                               penalized_medians, metric_names)

    # LaTeX table
    latex_table_main(
        out_tex=out_dir / "table_main.tex",
        labels=[la, lb, lc],
        cov_counts=cov_counts,
        total=len(keys_inter),
        fair_stats=fair_stats,
        pen_stats=pen_stats,
        metric_names=metric_names,
        caption=args.latex_caption,
        label=args.latex_label,
    )

    # Summary JSON
    summary_obj = {
        "inputs": {la: str(path_a), lb: str(path_b), lc: str(path_c)},
        "parsing": {
            la: {"raw_type": type(raw_a).__name__, "normalized_len": len(ra)},
            lb: {"raw_type": type(raw_b).__name__, "normalized_len": len(rb)},
            lc: {"raw_type": type(raw_c).__name__, "normalized_len": len(rc)},
        },
        "keys_union": len(keys_union),
        "keys_intersection": len(keys_inter),
        "coverage_pose": {
            la: {"valid": cov_counts[la], "total": len(keys_inter)},
            lb: {"valid": cov_counts[lb], "total": len(keys_inter)},
            lc: {"valid": cov_counts[lc], "total": len(keys_inter)},
        },
        "metrics": metric_names,
        "pose_metrics": pose_metrics,
        "fair_stats": fair_stats,
        "penalized_stats": pen_stats,
        "penalties": penalties,
        "args": {"penalty_q": args.penalty_q, "penalty_mult": args.penalty_mult},
        "outputs": {
            "coverage_png": str((plots_dir / "coverage.png").resolve()),
            "penalized_boxplots_png": str((plots_dir / "penalized_boxplots.png").resolve()),
            "penalized_medians_png": str((plots_dir / "penalized_medians.png").resolve()),
            "latex_table": str((out_dir / "table_main.tex").resolve()),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary_obj, indent=2), encoding="utf-8")

    print("[DONE] Plots written to:", plots_dir.resolve())
    print("[DONE] LaTeX table:", (out_dir / "table_main.tex").resolve())
    print("[DONE] Summary JSON:", (out_dir / "summary.json").resolve())


if __name__ == "__main__":
    main()
