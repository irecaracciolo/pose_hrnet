import pandas as pd
import matplotlib.pyplot as plt

old_path = "/nfs/home/caracciolo/pose_hrnet/runs/2026-01-23_09-23-41_hrnet_hmOnly_weightedMSE_pw20/loss_history.csv"
new_path = "/nfs/home/caracciolo/pose_hrnet/runs/2026-01-23_09-23-41_hrnet_hmOnly_weightedMSE_pw20/loss_history_resume_clean.csv"

old = pd.read_csv(old_path)
new = pd.read_csv(new_path)

# 1) scarta la riga rotta (epoch 55) nel vecchio: è quella con config NaN
old = old[~old["hm_loss_type"].isna()].copy()

# 2) rinomina colonne del nuovo -> stesso naming del vecchio
new = new.rename(columns={
    "train_loss_total": "train_total",
    "train_loss_hm": "train_hm",
    "train_loss_coord": "train_coord",
    "val_loss_total": "val_total",
    "val_loss_hm": "val_hm",
    "val_loss_coord": "val_coord",
})

# 3) aggiungi nel nuovo le colonne di config che non esistono (riempi con last-known del vecchio)
cfg_cols = [
    "use_coord_loss", "lambda_coord", "beta_softarg", "hm_loss_type",
    "pos_weight", "thr", "debug_hm_every", "debug_hm_n", "debug_hm_k", "debug_hm_seed"
]
last_cfg = old.sort_values("epoch").iloc[-1][cfg_cols].to_dict()
for c in cfg_cols:
    new[c] = last_cfg[c]

# 4) tieni le stesse colonne, concatena e ordina
cols = ["epoch", "train_total", "train_hm", "train_coord", "val_total", "val_hm", "val_coord",
        "epoch_time_s"] + cfg_cols
merged = pd.concat([old[cols], new[cols]],
                   ignore_index=True).sort_values("epoch")

# 5) salva CSV unico
merged.to_csv("/nfs/home/caracciolo/pose_hrnet/runs/2026-01-23_09-23-41_hrnet_hmOnly_weightedMSE_pw20/loss_history_merged.csv", index=False)

# 6) plot (con linea di "resume")
resume_epoch = new["epoch"].min()

plt.figure()
plt.plot(merged["epoch"], merged["train_total"], label="train_total")
plt.plot(merged["epoch"], merged["val_total"], label="val_total")
plt.axvline(resume_epoch, linestyle="--", linewidth=1, label="resume")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("/nfs/home/caracciolo/pose_hrnet/runs/2026-01-23_09-23-41_hrnet_hmOnly_weightedMSE_pw20/plots/loss_curves_merged.png", dpi=200)
plt.close()

print("Saved: loss_history_merged.csv and loss_curves_merged.png")
