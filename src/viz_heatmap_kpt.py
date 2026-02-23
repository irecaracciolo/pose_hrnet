# src/viz_heatmap_kpt.py
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import timm

from src.dataset_speedplus import SpeedPlusKeypointDataset


# -------------------------
# Model factory (auto)
# -------------------------
class HRNetKeypointModelAuto(nn.Module):
    """
    HRNet features_only gives 5 feature maps:
      0: 64  @ 256x256
      1: 128 @ 128x128
      2: 256 @ 64x64
      3: 512 @ 32x32
      4: 1024@ 16x16

    We choose out_index based on desired feature channels.
    """

    def __init__(self, num_keypoints: int, out_index: int):
        super().__init__()
        self.out_index = int(out_index)
        self.backbone = timm.create_model(
            "hrnet_w32",
            pretrained=True,
            features_only=True,
            out_indices=(self.out_index,),
        )
        out_ch = self.backbone.feature_info[self.out_index]["num_chs"]
        self.head = nn.Conv2d(out_ch, num_keypoints,
                              kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        feats = self.backbone(x)[0]
        return self.head(feats)


def infer_out_index_from_checkpoint(state: dict) -> int:
    """
    Detect head input channels from ckpt:
      head.weight shape = (K, C, 1, 1) -> C in {64,128,256,512,1024}
    Map C -> out_index in HRNet list above.
    """
    w = state.get("head.weight", None)
    if w is None:
        raise KeyError(
            "Checkpoint does not contain 'head.weight' (unexpected format).")
    c_in = int(w.shape[1])
    c_to_index = {64: 0, 128: 1, 256: 2, 512: 3, 1024: 4}
    if c_in not in c_to_index:
        raise ValueError(
            f"Unrecognized head input channels: {c_in}. Expected one of {sorted(c_to_index.keys())}")
    return c_to_index[c_in]


def load_state(ckpt_path: Path, device: torch.device) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device)
    # support dict checkpoint or raw state_dict
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    # strip module.
    state = {k.replace("module.", ""): v for k, v in state.items()}
    return state


# -------------------------
# Visualization helpers
# -------------------------
def hm_to_uint8(hm: np.ndarray, fixed_01: bool) -> np.ndarray:
    hm = hm.astype(np.float32)
    if fixed_01:
        hm = np.clip(hm, 0.0, 1.0)
        hm = hm / 1.0
    else:
        hm = hm - hm.min()
        hm = hm / (hm.max() + 1e-8)
    return (hm * 255.0).clip(0, 255).astype(np.uint8)


def overlay(img_bgr: np.ndarray, hm: np.ndarray, alpha=0.4, fixed_01=False) -> np.ndarray:
    H, W = img_bgr.shape[:2]
    hm_f = cv2.resize(hm.astype(np.float32), (W, H),
                      interpolation=cv2.INTER_LINEAR)
    hm_u8 = hm_to_uint8(hm_f, fixed_01=fixed_01)
    hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 1 - alpha, hm_color, alpha, 0.0)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True,
                    help="path to checkpoint .pth")
    ap.add_argument("--split", type=str, default="validation",
                    choices=["train", "validation"])
    ap.add_argument("--domain", type=str, default="synthetic",
                    choices=["synthetic", "lightbox", "sunlamp"])
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="outputs/viz_heatmaps_ckpt")
    ap.add_argument("--k_list", type=str, default="0,3,8",
                    help="comma-separated kp ids")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # dataset (GT heatmaps available only for synthetic with return_heatmaps=True)
    ds = SpeedPlusKeypointDataset(
        split=args.split,
        domain=args.domain,
        img_size=(512, 512),
        heatmap_size=(128, 128),
        sigma=2.0,
        return_heatmaps=(args.domain == "synthetic"),
        normalize_rgb=False,
    )
    print("dataset size:", len(ds))

    ckpt_path = Path(args.ckpt)
    state = load_state(ckpt_path, device)
    out_index = infer_out_index_from_checkpoint(state)
    print(
        f"[INFO] checkpoint head expects C_in={int(state['head.weight'].shape[1])} -> using out_index={out_index}")

    # num keypoints from ckpt head weight
    num_kpts = int(state["head.weight"].shape[0])

    model = HRNetKeypointModelAuto(
        num_keypoints=num_kpts, out_index=out_index).to(device).eval()
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("[INFO] loaded. missing:", len(
        missing), "unexpected:", len(unexpected))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    k_list = [int(x) for x in args.k_list.split(",") if x.strip() != ""]
    k_list = [k for k in k_list if 0 <= k < num_kpts]
    if not k_list:
        raise ValueError("k_list is empty after filtering")

    idxs = random.sample(range(len(ds)), k=min(args.n, len(ds)))

    for idx in idxs:
        s = ds[idx]
        img = (s["image"].numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        x = s["image"].unsqueeze(0).to(device)
        logits = model(x)[0].detach().cpu().numpy()  # (K,Hm_pred,Wm_pred)
        # NOTE: logits scale is unbounded; for visualization use sigmoid
        pred = 1.0 / (1.0 + np.exp(-logits))

        stem = Path(s["img_name"]).stem

        # max over kpts for global view
        pred_max = pred.max(axis=0)
        cv2.imwrite(str(out_dir / f"{stem}_PRED_overlay.jpg"),
                    overlay(img_bgr, pred_max, fixed_01=True))

        if args.domain == "synthetic":
            gt = s["heatmaps"].numpy()
            gt_max = gt.max(axis=0)
            cv2.imwrite(str(out_dir / f"{stem}_GT_overlay.jpg"),
                        overlay(img_bgr, gt_max, fixed_01=False))

        # save some single-k heatmaps
        for k in k_list:
            cv2.imwrite(str(out_dir / f"{stem}_PRED_hm{k:02d}.jpg"),
                        cv2.applyColorMap(hm_to_uint8(pred[k], fixed_01=True), cv2.COLORMAP_JET))
            if args.domain == "synthetic":
                cv2.imwrite(str(out_dir / f"{stem}_GT_hm{k:02d}.jpg"),
                            cv2.applyColorMap(hm_to_uint8(gt[k], fixed_01=False), cv2.COLORMAP_JET))

    print("Saved to:", out_dir.resolve())


if __name__ == "__main__":
    main()
