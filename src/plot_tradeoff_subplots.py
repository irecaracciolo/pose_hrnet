#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl

UNITS = {
    "accept_rate": "[%]",
    "rmse_inliers_mean": "[px]",
    "slab_mean": "[-]",
    "rmse_inliers_thr_px": "[px]",
    "min_inliers": "[-]",
    "min_kpt_area_px2": "[px²]",
    "t_ratio_max": "[-]",
}


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def to_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def same_val(a, b, tol=1e-9):
    if np.isinf(a) and np.isinf(b):
        return True
    return abs(a - b) <= tol


def load_best_from_json(best_json: Path, df: pd.DataFrame):
    """
    Read best_point.json and find the matching row in results.csv
    using gate parameters.
    """
    with open(best_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    best = data["best"]

    mask = np.ones(len(df), dtype=bool)
    for k in ["min_inliers", "rmse_inliers_thr_px", "min_kpt_area_px2", "t_ratio_max"]:
        if k not in best or k not in df.columns:
            raise KeyError(
                f"Key '{k}' missing in best_point.json or results.csv")

        v = float(best[k])
        # compare robustly
        mask &= df[k].apply(lambda x: same_val(float(x), v))

    idx = df[mask].index
    if len(idx) == 0:
        raise RuntimeError(
            "BEST point from best_point.json not found in results.csv")

    if len(idx) > 1:
        print("[WARN] Multiple rows match BEST parameters, using the first one")

    return int(idx[0])


# ------------------------------------------------------------
# Grid builder
# ------------------------------------------------------------
def make_grid(df_slice, xcol, ycol, zcol):
    xs = np.sort(df_slice[xcol].unique())
    ys = np.sort(df_slice[ycol].unique())
    Z = np.full((len(ys), len(xs)), np.nan)

    # NOTE: this assumes xcol,ycol take exact values found in xs,ys
    # If you ever have float jitter, consider rounding.
    for _, r in df_slice.iterrows():
        xi = np.where(xs == r[xcol])[0][0]
        yi = np.where(ys == r[ycol])[0][0]
        Z[yi, xi] = r[zcol]

    return xs, ys, Z


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
def plot_metric(df, metric, best_idx, args, out_png):
    areas = np.sort(df[args.area_col].unique())
    tratios = np.sort(df[args.tratio_col].unique())

    # Work on a copy for plotting
    df_plot = df.copy()

    # Convert accept_rate to percentage BEFORE computing vmin/vmax and grids
    if metric == "accept_rate":
        df_plot[metric] = df_plot[metric] * 100.0

    z_all = df_plot[metric].to_numpy(float)
    z_all = z_all[np.isfinite(z_all)]
    if len(z_all) == 0:
        print(f"[SKIP] {metric}: no finite values")
        return

    vmin = float(np.nanmin(z_all))
    vmax = float(np.nanmax(z_all))

    if args.clip_p99:
        vmax = float(np.nanpercentile(z_all, 99))

    # Build GLOBAL levels so every subplot uses the same boundaries
    # 18 filled bands -> 19 boundaries
    levels = np.linspace(vmin, vmax, 256)

    cmap = "viridis_r" if metric == "accept_rate" else "viridis"
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(4 * len(tratios) + 1, 3.2 * len(areas)))
    gs = GridSpec(
        len(areas),
        len(tratios) + 1,
        width_ratios=[1] * len(tratios) + [0.06],
        wspace=0.3,
        hspace=0.35,
    )

    cax = fig.add_subplot(gs[:, -1])

    best_row = df.loc[best_idx]

    last_cf = None  # not strictly needed anymore, but kept for completeness

    for i, a in enumerate(areas):
        for j, t in enumerate(tratios):
            ax = fig.add_subplot(gs[i, j])

            dsl = df_plot[(df_plot[args.area_col] == a) &
                          (df_plot[args.tratio_col] == t)]

            ax.set_title(
                f"{args.area_col}={a:g} | {args.tratio_col}={t:g}", fontsize=9)
            ax.set_xlabel(f"{args.xcol}{UNITS.get(args.xcol, '')}")
            ax.set_ylabel(f"{args.ycol}{UNITS.get(args.ycol, '')}")

            if len(dsl) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                continue

            xs, ys, Z = make_grid(dsl, args.xcol, args.ycol, metric)

            # Clip to match the global scale (especially if clip_p99 is active)
            Z = np.clip(Z, vmin, vmax)

            # Use global levels + global norm for consistent colors everywhere
            # extend="both" shows triangles if something is clipped (useful diagnostics)
            cf = ax.contourf(
                xs,
                ys,
                Z,
                levels=levels,
                cmap=cmap,
                norm=norm,
                extend="both",
            )
            last_cf = cf

            if same_val(a, best_row[args.area_col]) and same_val(t, best_row[args.tratio_col]):
                ax.scatter(
                    best_row[args.xcol],
                    best_row[args.ycol],
                    marker="X",
                    s=140,
                    edgecolors="k",
                    linewidths=1.5,
                    zorder=5,
                )
                ax.annotate(
                    "BEST",
                    (best_row[args.xcol], best_row[args.ycol]),
                    textcoords="offset points",
                    xytext=(8, 8),
                    fontsize=9,
                    zorder=6,
                )
    cf = ax.contourf(xs, ys, Z, levels=levels,
                     cmap=cmap, norm=norm, extend="both")

    # Build a GLOBAL colorbar (independent of last subplot)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(f"{metric}{UNITS.get(metric, '')}")

    fig.suptitle(
        f"Trade-off colormaps (mean) | metric={metric} {UNITS.get(metric, '')}",
        fontsize=14,
    )

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_png)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_csv", required=True)
    ap.add_argument("--best_json", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--xcol", default="rmse_inliers_thr_px")
    ap.add_argument("--ycol", default="min_inliers")
    ap.add_argument("--area_col", default="min_kpt_area_px2")
    ap.add_argument("--tratio_col", default="t_ratio_max")

    ap.add_argument("--min_accept_rate", type=float, default=0.0)
    ap.add_argument("--clip_p99", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.results_csv)
    df = to_numeric(
        df,
        [
            args.xcol,
            args.ycol,
            args.area_col,
            args.tratio_col,
            "accept_rate",
            "slab_mean",
            "rmse_inliers_mean",
        ],
    )

    if args.min_accept_rate > 0:
        df = df[df["accept_rate"] >= args.min_accept_rate]

    if len(df) == 0:
        raise RuntimeError(
            "No rows left after filtering (min_accept_rate too strict?)")

    best_idx = load_best_from_json(Path(args.best_json), df)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric in ["slab_mean", "rmse_inliers_mean", "accept_rate"]:
        if metric in df.columns and np.isfinite(df[metric]).any():
            plot_metric(df, metric, best_idx, args,
                        out_dir / f"tradeoff_{metric}.png")

    b = df.loc[best_idx]
    (out_dir / "best_point.txt").write_text(
        "\n".join([f"{k} = {b[k]}" for k in b.index]),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

# python -m src.plot_tradeoff_subplots \
#   --results_csv /.../results.csv \
#   --best_json   /.../best_point.json \
#   --out_dir     /.../tradeoff_pngs \
#   --clip_p99
