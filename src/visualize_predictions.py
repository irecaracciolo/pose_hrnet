# src/visualize_predictions.py
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from src.config import OUTPUTS_DIR, CHECKPOINTS_DIR
from src.dataset_speedplus import SpeedPlusKeypointDataset
from src.model_hrnet_pose import HRNetKeypointModel
from src.config import KEYPOINTS_3D_NPY


def heatmaps_to_kpts_argmax(hmaps: np.ndarray, img_size=(512, 512)) -> np.ndarray:
    """
    hmaps: (K, Hm, Wm)
    returns: (K, 2) in pixel coords of img_size
    """
    K, Hm, Wm = hmaps.shape
    img_w, img_h = img_size
    kpts = np.zeros((K, 2), dtype=np.float32)

    sx = img_w / float(Wm)
    sy = img_h / float(Hm)

    for i in range(K):
        idx = int(hmaps[i].argmax())
        y, x = np.unravel_index(idx, (Hm, Wm))
        kpts[i, 0] = x * sx
        kpts[i, 1] = y * sy
    return kpts


def draw_kpts(img_rgb: np.ndarray, kpts: np.ndarray, color_bgr, label_prefix=None):
    for i, (x, y) in enumerate(kpts):
        cv2.circle(img_rgb, (int(x), int(y)), 3,
                   color_bgr, -1, lineType=cv2.LINE_AA)
        if label_prefix is not None:
            cv2.putText(
                img_rgb, f"{label_prefix}{i}",
                (int(x) + 4, int(y) - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_bgr, 1, cv2.LINE_AA
            )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # dataset: puoi cambiare domain/split
    ds = SpeedPlusKeypointDataset(
        split="validation",
        domain="synthetic",
        img_size=(512, 512),
        heatmap_size=(128, 128),
        sigma=2.0,
        return_heatmaps=True,
    )

    kpts_3d = np.load(KEYPOINTS_3D_NPY)
    num_kpts = int(kpts_3d.shape[0])

    # model + checkpoint
    model = HRNetKeypointModel(num_keypoints=num_kpts).to(device)
    ckpt_path = Path(CHECKPOINTS_DIR) / "hrnet_kpts_best.pth"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    # supporta sia checkpoint dict che state_dict puro
    state = ckpt["model_state"] if isinstance(
        ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    out_dir = Path(OUTPUTS_DIR) / "debug_preds"
    out_dir.mkdir(parents=True, exist_ok=True)

    # scegli N campioni random
    N = 20
    idxs = random.sample(range(len(ds)), k=min(N, len(ds)))

    for idx in idxs:
        sample = ds[idx]
        img_t = sample["image"].unsqueeze(0).to(device)  # (1,3,512,512)

        with torch.no_grad():
            hmaps = model(img_t)[0].detach().cpu().numpy()  # (K,Hm,Wm)

        kpts_pred = heatmaps_to_kpts_argmax(hmaps, img_size=(512, 512))
        kpts_gt = sample["kpts_2d"].cpu().numpy()

        # immagine per OpenCV
        img = (sample["image"].cpu().numpy().transpose(
            1, 2, 0) * 255.0).astype(np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # GT verde, PRED rosso (OpenCV è BGR)
        draw_kpts(img_bgr, kpts_gt, (0, 255, 0), label_prefix="g")
        draw_kpts(img_bgr, kpts_pred, (0, 0, 255), label_prefix="p")

        # errore pixel medio (solo diagnostico)
        err_px = np.linalg.norm(kpts_pred - kpts_gt, axis=1).mean()

        name = sample["img_name"]
        out_path = out_dir / f"{Path(name).stem}_err{err_px:.2f}.jpg"
        cv2.imwrite(str(out_path), img_bgr)

    print(f"Saved {len(idxs)} overlays to: {out_dir}")


if __name__ == "__main__":
    main()
