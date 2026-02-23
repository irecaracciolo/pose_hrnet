# src/test_dataset.py
import torch
import numpy as np

from src.dataset_speedplus import SpeedPlusKeypointDataset


def main():
    ds = SpeedPlusKeypointDataset(
        split="train",
        domain="synthetic",
        img_size=(512, 512),
        heatmap_size=(128, 128),
        return_heatmaps=True,
    )

    print("Dataset size:", len(ds))

    sample = ds[0]

    kpts_2d = sample["kpts_2d"]          # (K,2)
    heatmaps = sample["heatmaps"]        # (K,Hm,Wm)

    print("Image name:", sample["img_name"])
    print("q (wxyz):", sample["q"])
    print("t_vbs:", sample["t_vbs2tango"])
    print("kpts_2d shape:", kpts_2d.shape)
    print("heatmaps shape:", heatmaps.shape)

    img_w, img_h = 512, 512
    hm_h, hm_w = heatmaps.shape[1], heatmaps.shape[2]

    sx = hm_w / img_w
    sy = hm_h / img_h

    print("\nHeatmap peak check (first 3 keypoints):")

    for i in range(3):
        u, v = kpts_2d[i].numpy()
        u_hm_exp = u * sx
        v_hm_exp = v * sy

        hm = heatmaps[i].numpy()
        y_peak, x_peak = np.unravel_index(np.argmax(hm), hm.shape)

        du = x_peak - u_hm_exp
        dv = y_peak - v_hm_exp

        print(
            f"KP {i}: "
            f"expected HM (x,y)=({u_hm_exp:.2f},{v_hm_exp:.2f}) | "
            f"peak HM (x,y)=({x_peak},{y_peak}) | "
            f"delta=({du:.2f},{dv:.2f})"
        )


if __name__ == "__main__":
    main()
