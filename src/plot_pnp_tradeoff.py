# src/plot_pnp_tradeoff.py
from __future__ import annotations
import matplotlib.pyplot as plt

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def main():
    # Adatta se hai OUTPUTS_DIR diverso
    in_json = Path("outputs") / "pnp_tradeoff" / "summary.json"
    out_dir = Path("outputs") / "pnp_tradeoff"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_json.exists():
        raise FileNotFoundError(f"Not found: {in_json.resolve()}")

    rows = json.loads(in_json.read_text())
    df = pd.DataFrame(rows)

    # --- colonne "chiave" (se manca qualcosa, non crasha) ---
    wanted = [
        "method",
        "ok", "fail", "attempted", "fail_rate_attempted",
        "median_esa", "p90_esa",
        "median_e_q_rad", "p90_e_q_rad",
        "median_e_t", "p90_e_t",
        "median_rmse_px", "p90_rmse_px",
        "median_ms_success", "p90_ms_success",
        "fps_median_success",
    ]
    cols = [c for c in wanted if c in df.columns]
    df = df[cols].copy()

    # Cast numerici dove serve
    for c in df.columns:
        if c != "method":
            df[c] = df[c].apply(safe_float)

    # Ordina (prima accuratezza, poi tempo) – puoi cambiare criterio
    if "median_esa" in df.columns:
        df = df.sort_values(
            ["median_esa", "median_ms_success"], ascending=[True, True])

    # -------------------------
    # 1) CSV + Markdown table
    # -------------------------
    csv_path = out_dir / "summary_table.csv"
    md_path = out_dir / "summary_table.md"
    df.to_csv(csv_path, index=False)

    # markdown: format più leggibile
    df_md = df.copy()
    # formato colonne principali
    fmt_cols = {
        "fail_rate_attempted": "{:.3f}",
        "median_esa": "{:.4f}",
        "p90_esa": "{:.4f}",
        "median_e_q_rad": "{:.4f}",
        "p90_e_q_rad": "{:.4f}",
        "median_ms_success": "{:.3f}",
        "p90_ms_success": "{:.3f}",
        "fps_median_success": "{:.1f}",
        "median_e_t": "{:.4f}",
        "p90_e_t": "{:.4f}",
        "median_rmse_px": "{:.2f}",
        "p90_rmse_px": "{:.2f}",
    }
    for c, f in fmt_cols.items():
        if c in df_md.columns:
            df_md[c] = df_md[c].apply(
                lambda x: f.format(x) if np.isfinite(x) else "NA")
    md_path.write_text(df_md.to_markdown(index=False))

    # -------------------------
    # 2) Bar charts
    # -------------------------
    methods = df["method"].tolist()

    def barplot(metric, ylabel, filename, ascending=True):
        if metric not in df.columns:
            print(f"Skip plot {metric}: missing column")
            return
        d = df[["method", metric]].copy()
        d = d.sort_values(metric, ascending=ascending)
        plt.figure(figsize=(10, 4))
        plt.bar(d["method"], d[metric])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel(ylabel)
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=200)
        plt.close()

    barplot("median_esa", "median ESA score (lower is better)",
            "bar_esa.png", ascending=True)
    barplot("median_e_q_rad",
            "median rotation error e_q [rad] (lower is better)", "bar_eq.png", ascending=True)
    barplot("median_ms_success",
            "median runtime [ms] (lower is better)", "bar_time.png", ascending=True)

    # -------------------------
    # 3) Pareto: accuracy vs time vs robustness
    # -------------------------
    if all(c in df.columns for c in ["median_esa", "median_ms_success", "fail_rate_attempted"]):
        x = df["median_ms_success"].to_numpy()
        y = df["median_esa"].to_numpy()
        fr = df["fail_rate_attempted"].to_numpy()

        # size proportional to fail rate
        fr_clean = np.nan_to_num(fr, nan=np.nanmax(
            fr[np.isfinite(fr)]) if np.any(np.isfinite(fr)) else 0.0)
        sizes = 80 + 800 * (fr_clean - np.nanmin(fr_clean)) / \
            (np.nanmax(fr_clean) - np.nanmin(fr_clean) + 1e-12)

        plt.figure(figsize=(6.5, 5))
        sc = plt.scatter(x, y, s=sizes, alpha=0.75)
        for i, m in enumerate(df["method"]):
            plt.annotate(
                m, (x[i], y[i]), textcoords="offset points", xytext=(5, 5), fontsize=8)

        plt.xlabel("median time [ms] (lower better)")
        plt.ylabel("median ESA score (lower better)")
        plt.title("Pareto: ESA vs runtime (bubble size = fail rate)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "pareto.png", dpi=220)
        plt.close()

    print("Written:")
    print(" -", csv_path)
    print(" -", md_path)
    for p in ["bar_esa.png", "bar_eq.png", "bar_time.png", "pareto.png"]:
        fp = out_dir / p
        if fp.exists():
            print(" -", fp)


if __name__ == "__main__":
    main()
