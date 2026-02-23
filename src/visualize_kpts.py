# src/visualize_kpts.py
import os
import cv2
import numpy as np

from src.dataset_speedplus import SpeedPlusKeypointDataset


def main():
    ds = SpeedPlusKeypointDataset(
        img_size=(512, 512),
        heatmap_size=(128, 128),
        sigma=2.0
    )

    print("Dataset size:", len(ds))

    idx = 0  # cambia indice se vuoi vedere altre immagini
    sample = ds[idx]

    img_name = sample["img_name"]
    print("Visualizing sample:", img_name)

    # image: tensor (3,H,W) in [0,1]
    img = sample["image"].numpy().transpose(1, 2, 0) * 255.0
    img = img.astype(np.uint8).copy()

    # keypoints 2D (riscalati a 512x512)
    kpts_2d = sample["kpts_2d"].numpy()  # shape (11, 2)

    # disegna keypoints
    for i, (x, y) in enumerate(kpts_2d):
        cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)
        cv2.putText(img, str(i), (int(x) + 5, int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

    # salva su file (BGR per OpenCV)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    out_name = f"debug_kpts_{os.path.splitext(img_name)[0]}.jpg"
    out_path = os.path.join(os.path.dirname(__file__), "..", out_name)
    cv2.imwrite(out_path, img_bgr)
    print("Saved visualization to:", out_path)


if __name__ == "__main__":
    main()
