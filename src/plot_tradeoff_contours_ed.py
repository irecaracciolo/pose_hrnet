#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def pick_objective_column(df: pd.DataFrame, prefer: str = "mean") -> str:
    """
    Prefer mean columns:
      - If slab_mean exists and has finite values -> use it
      - Else use rmse_inliers_mean
    """
    if prefer != "mean":
        raise ValueError("This script is built for mean only.")

    if "slab_mean" in df.columns and np.isfinite(df["slab_mean"].to_numpy(dtype=float)).any():
        return "slab_mean"
    if "rmse_inliers_mean" in df.columns and np.isfinite(df["rmse_inliers_mean"].to_numpy(dtype=float)).any():
        return "rmse_inliers_mean"
    raise ValueError(
        "Could not find a valid objective mean column (slab_mean or rmse_inliers_mean).")


def make_grid(df_slice: pd.DataFrame, xcol: str, ycol: str, zcol: str):
    """
    Build a regular grid for contour/surface:
      x = sorted unique rmse_thr
      y = sorted unique min_inliers
      z[y_i, x_j] = value
    """
    xs = np.sort(df_slice[xcol].unique())
    ys = np.sort(df_slice[ycol].unique())

    z = np.full((len(ys), len(xs)), np.nan, dtype=float)

    # fill grid
    for _, r in df_slice.iterrows():
        xi = np.where(xs == r[xcol])[0][0]
        yi = np.where(ys == r[ycol])[0][0]
        z[yi, xi] = float(r[zcol])

    return xs, ys, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--min_accept_rate", type=float, default=0.0)
    ap.add_argument("--xcol", type=str, default="rmse_inliers_thr_px")
    ap.add_argument("--ycol", type=str, default="min_inliers")
    ap.add_argument("--area_col", type=str, default="min_kpt_area_px2")
    ap.add_argument("--tratio_col", type=str, default="t_ratio_max")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.results_csv)

    # guardrail
    if args.min_accept_rate > 0:
        df = df[df["accept_rate"] >= args.min_accept_rate].copy()

    obj_col = pick_objective_column(df, prefer="mean")

    xcol, ycol = args.xcol, args.ycol
    area_col, tratio_col = args.area_col, args.tratio_col

    # Ensure required columns
    for c in [xcol, ycol, area_col, tratio_col, "accept_rate", obj_col]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    # Available slices
    areas = np.sort(df[area_col].unique())
    tratios = np.sort(df[tratio_col].unique())

    # Build frames for (area, tratio)
    frames_obj = []
    frames_acc = []
    frame_names = []

    for a in areas:
        for t in tratios:
            dsl = df[(df[area_col] == a) & (df[tratio_col] == t)].copy()
            if len(dsl) == 0:
                continue

            xs, ys, z_obj = make_grid(dsl, xcol, ycol, obj_col)
            _,  _, z_acc = make_grid(dsl, xcol, ycol, "accept_rate")

            name = f"area={a:g}, tratio={t:g}"
            frame_names.append(name)

            frames_obj.append(go.Frame(
                name=name,
                data=[go.Contour(
                    x=xs,
                    y=ys,
                    z=z_obj,
                    contours=dict(showlabels=True),
                    colorbar=dict(title=obj_col),
                )]
            ))

            frames_acc.append(go.Frame(
                name=name,
                data=[go.Contour(
                    x=xs,
                    y=ys,
                    z=z_acc,
                    contours=dict(showlabels=True),
                    colorbar=dict(title="accept_rate"),
                )]
            ))

    if len(frames_obj) == 0:
        raise ValueError(
            "No slice produced any frame. Check your columns / filters.")

    # --- Contour: objective_mean ---
    fig_obj = go.Figure(
        data=frames_obj[0].data,
        frames=frames_obj
    )

    fig_obj.update_layout(
        title=f"Contour map: {obj_col} (mean) over ({ycol} vs {xcol}) | slice: area,tratio",
        xaxis_title=xcol,
        yaxis_title=ycol,
        updatemenus=[{
            "type": "buttons",
            "showactive": True,
            "x": 0.02,
            "y": 1.10,
            "buttons": [
                {"label": "Play", "method": "animate",
                 "args": [None, {"frame": {"duration": 600, "redraw": True}, "fromcurrent": True}]},
                {"label": "Pause", "method": "animate",
                 "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]},
            ],
        }],
        sliders=[{
            "active": 0,
            "currentvalue": {"prefix": "Slice: "},
            "steps": [
                {"label": n, "method": "animate", "args": [
                    [n], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]}
                for n in frame_names
            ],
        }],
        margin=dict(l=60, r=40, t=90, b=60),
    )

    fig_obj.write_html(out_dir / "contour_objective_mean.html")

    # --- Contour: accept_rate ---
    fig_acc = go.Figure(
        data=frames_acc[0].data,
        frames=frames_acc
    )

    fig_acc.update_layout(
        title=f"Contour map: accept_rate over ({ycol} vs {xcol}) | slice: area,tratio",
        xaxis_title=xcol,
        yaxis_title=ycol,
        updatemenus=fig_obj.layout.updatemenus,
        sliders=fig_obj.layout.sliders,
        margin=dict(l=60, r=40, t=90, b=60),
    )

    fig_acc.write_html(out_dir / "contour_accept_rate.html")

    # --- 3D surface: objective_mean ---
    # (Same frames, but Surface)
    frames_surf = []
    for a in areas:
        for t in tratios:
            dsl = df[(df[area_col] == a) & (df[tratio_col] == t)].copy()
            if len(dsl) == 0:
                continue
            xs, ys, z_obj = make_grid(dsl, xcol, ycol, obj_col)
            name = f"area={a:g}, tratio={t:g}"
            frames_surf.append(go.Frame(
                name=name,
                data=[go.Surface(x=xs, y=ys, z=z_obj)]
            ))

    fig_surf = go.Figure(data=frames_surf[0].data, frames=frames_surf)
    fig_surf.update_layout(
        title=f"3D surface: {obj_col} (mean) | X={xcol}, Y={ycol}, Z={obj_col}",
        scene=dict(
            xaxis_title=xcol,
            yaxis_title=ycol,
            zaxis_title=obj_col,
        ),
        updatemenus=fig_obj.layout.updatemenus,
        sliders=fig_obj.layout.sliders,
        margin=dict(l=20, r=20, t=70, b=20),
    )
    fig_surf.write_html(out_dir / "surface_objective_mean.html")

    print("Saved HTML plots in:", out_dir.resolve())
    print(" - contour_objective_mean.html")
    print(" - contour_accept_rate.html")
    print(" - surface_objective_mean.html")


if __name__ == "__main__":
    main()
