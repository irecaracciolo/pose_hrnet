#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
2D-only thesis colormaps:
X = min_inliers
Y = rmse_inliers_thr_px
Color = chosen metric (slab_median, discard_pct, eq_median, ...)

- Optional interpolation (linear/cubic) to avoid blocky tiles.
- Best point highlighted (min objective with accept_rate constraint).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.interpolate import griddata
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def resolve_path(p: str) -> Path:
    return Path(p).expanduser().resolve()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def compute_best_point(df: pd.DataFrame, objective: str, min_accept_rate: float) -> Optional[dict]:
    if objective not in df.columns or "accept_rate" not in df.columns:
        return None

    d = df.copy()
    d = d[np.isfinite(d[objective].astype(float))]
    d = d[np.isfinite(d["accept_rate"].astype(float))]
    d = d[d["accept_rate"].astype(float) >= float(min_accept_rate)]
    if len(d) == 0:
        return None

    sort_cols = [objective]
    asc = [True]
    if "slab_p90" in d.columns and objective != "slab_p90":
        sort_cols.append("slab_p90")
        asc.append(True)
    sort_cols.append("accept_rate")
    asc.append(False)

    d = d.sort_values(sort_cols, ascending=asc).reset_index(drop=True)
    b = d.iloc[0].to_dict()

    keep = [
        "min_inliers", "rmse_inliers_thr_px",
        "N_total", "N_accepted", "accept_rate", "discard_pct",
        "slab_median", "slab_p90", "et_median", "eq_median"
    ]
    return {
        "objective": objective,
        "min_accept_rate": float(min_accept_rate),
        "best": {k: b.get(k, None) for k in keep if k in b},
    }


def plot_2d_colormap(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    z_col: str,
    out_path: Path,
    title: str,
    cmap: str = "viridis",
    interp: str = "cubic",     # none|linear|cubic
    interp_res: int = 250,
    highlight_xy: Optional[Tuple[float, float]] = None,
) -> None:
    ensure_dir(out_path.parent)

    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    z = df[z_col].to_numpy(dtype=float)

    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[ok], y[ok], z[ok]

    fig = plt.figure(figsize=(7.6, 5.6))
    ax = plt.gca()

    if interp != "none" and _HAVE_SCIPY and len(z) >= 6:
        xi = np.linspace(float(np.min(x)), float(np.max(x)), interp_res)
        yi = np.linspace(float(np.min(y)), float(np.max(y)), interp_res)
        XI, YI = np.meshgrid(xi, yi)
        ZI = griddata(np.stack([x, y], axis=1), z, (XI, YI), method=interp)
        ZI = np.ma.masked_invalid(ZI)
        m = ax.contourf(XI, YI, ZI, levels=28, cmap=cmap)
    else:
        # discrete fallback (still OK if grid is dense)
        pivot = df.pivot_table(index=y_col, columns=x_col,
                               values=z_col, aggfunc="mean")
        xs = pivot.columns.to_numpy(dtype=float)
        ys = pivot.index.to_numpy(dtype=float)
        Z = pivot.to_numpy(dtype=float)
        Zm = np.ma.masked_invalid(Z)
        m = ax.imshow(
            Zm, origin="lower", aspect="auto",
            extent=[float(xs.min()), float(xs.max()),
                    float(ys.min()), float(ys.max())],
            cmap=cmap
        )

    cb = plt.colorbar(m, ax=ax)
    cb.set_label(z_col)

    if highlight_xy is not None:
        hx, hy = highlight_xy
        ax.scatter([hx], [hy], marker="X", s=160,
                   c="black", linewidths=1.6, zorder=10)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.grid(True, linewidth=0.35, alpha=0.35)

    plt.savefig(out_path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--results", default="results.csv")

    ap.add_argument("--x", default="min_inliers")
    ap.add_argument("--y", default="rmse_inliers_thr_px")

    ap.add_argument("--objective", default="slab_median")
    ap.add_argument("--min_accept_rate_best", type=float, default=0.10)

    ap.add_argument("--interp", default="cubic",
                    choices=["none", "linear", "cubic"])
    ap.add_argument("--interp_res", type=int, default=250)
    ap.add_argument("--cmap", default="viridis")

    args = ap.parse_args()

    in_dir = resolve_path(args.in_dir)
    out_dir = resolve_path(args.out_dir) if args.out_dir else (
        in_dir / "plots_2d")
    ensure_dir(out_dir)

    df = pd.read_csv(in_dir / args.results)

    # ensure discard_pct
    if "discard_pct" not in df.columns and "accept_rate" in df.columns:
        df["discard_pct"] = 100.0 * (1.0 - df["accept_rate"].astype(float))

    best = compute_best_point(df, args.objective, args.min_accept_rate_best)
    highlight_xy = None
    if best is not None:
        (out_dir / "best_point.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
        b = best["best"]
        if args.x in b and args.y in b:
            highlight_xy = (float(b[args.x]), float(b[args.y]))

    # Produce a small suite of maps (edit as you like)
    maps = [
        ("slab_median", "SLAB median"),
        ("slab_p90", "SLAB p90"),
        ("discard_pct", "Discarded samples [%]"),
        ("eq_median", "Rotation error median [deg]"),
        ("et_median", "Translation error median [m]"),
    ]

    for z, ttl in maps:
        if z in df.columns:
            plot_2d_colormap(
                df,
                x_col=args.x, y_col=args.y, z_col=z,
                out_path=out_dir / f"map_{z}.png",
                title=ttl,
                cmap=args.cmap,
                interp=args.interp,
                interp_res=int(args.interp_res),
                highlight_xy=highlight_xy,
            )

    print("[DONE] 2D colormaps in:", out_dir)


if __name__ == "__main__":
    main()
