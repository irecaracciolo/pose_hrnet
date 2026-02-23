# src/debug_bbox_crop.py
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from src.dataset_speedplus import SpeedPlusKeypointDataset


def kpts_bbox_xyxy(
    kpts: np.ndarray,
    img_wh: Tuple[int, int],
    margin: float = 0.10,
    make_square: bool = True,
    min_size_px: int = 32,
):
    """
    kpts: (K,2) in pixel coords in the resized image space (img_wh)
    returns bbox as (x1,y1,x2,y2) inclusive-exclusive in int, clipped.
    Uses only "valid" keypoints inside image.
    """
    W, H = img_wh
    k = np.asarray(kpts, dtype=np.float32).reshape(-1, 2)

    # valid points: inside image
    m = (k[:, 0] >= 0) & (k[:, 0] < W) & (k[:, 1] >= 0) & (k[:, 1] < H)
    if not np.any(m):
        return None

    xs = k[m, 0]
    ys = k[m, 1]
    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())

    # add margin
    bw = max(x2 - x1, 1e-6)
    bh = max(y2 - y1, 1e-6)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    bw *= (1.0 + margin)
    bh *= (1.0 + margin)

    if make_square:
        s = max(bw, bh)
        bw = bh = s

    # enforce min size
    bw = max(bw, float(min_size_px))
    bh = max(bh, float(min_size_px))

    x1 = cx - 0.5 * bw
    x2 = cx + 0.5 * bw
    y1 = cy - 0.5 * bh
    y2 = cy + 0.5 * bh

    # clip
    x1 = int(np.floor(max(0.0, x1)))
    y1 = int(np.floor(max(0.0, y1)))
    x2 = int(np.ceil(min(float(W), x2)))
    y2 = int(np.ceil(min(float(H), y2)))

    if (x2 - x1) < 2 or (y2 - y1) < 2:
        return None
    return (x1, y1, x2, y2)


def crop_and_resize(
    img_rgb_u8: np.ndarray,
    kpts: np.ndarray,
    bbox_xyxy: Tuple[int, int, int, int],
    out_wh: Tuple[int, int],
):
    """
    img_rgb_u8: (H,W,3) uint8
    kpts: (K,2) in original image coords
    bbox: (x1,y1,x2,y2)
    out_wh: (Wout,Hout)
    returns: img_out_u8, kpts_out
    """
    x1, y1, x2, y2 = bbox_xyxy
    crop = img_rgb_u8[y1:y2, x1:x2, :]

    ch, cw = crop.shape[:2]
    Wout, Hout = out_wh

    # scale factors crop->out
    sx = Wout / float(max(cw, 1))
    sy = Hout / float(max(ch, 1))

    k = kpts.copy().astype(np.float32)
    k[:, 0] = (k[:, 0] - x1) * sx
    k[:, 1] = (k[:, 1] - y1) * sy

    crop_res = cv2.resize(crop, (Wout, Hout), interpolation=cv2.INTER_LINEAR)
    return crop_res, k


def draw_kpts(img_bgr: np.ndarray, kpts: np.ndarray, color=(0, 255, 0), prefix="k"):
    for i, (x, y) in enumerate(kpts):
        if x < 0 or y < 0:
            continue
        cv2.circle(img_bgr, (int(x), int(y)), 3,
                   color, -1, lineType=cv2.LINE_AA)
        cv2.putText(
            img_bgr, f"{prefix}{i}",
            (int(x) + 4, int(y) - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA
        )


def draw_bbox(img_bgr: np.ndarray, bbox_xyxy, color=(255, 0, 0)):
    x1, y1, x2, y2 = bbox_xyxy
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="train",
                    choices=["train", "val", "validation"])
    ap.add_argument("--domain", type=str, default="synthetic",
                    choices=["synthetic"])
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--square", type=int, default=1)
    ap.add_argument("--min_size", type=int, default=64)
    ap.add_argument("--out_dir", type=str, default="outputs/debug_bbox_crop")
    ap.add_argument("--img_size", type=int, nargs=2,
                    default=[512, 512], metavar=("W", "H"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = SpeedPlusKeypointDataset(
        split=args.split,
        domain=args.domain,
        img_size=(args.img_size[0], args.img_size[1]),
        heatmap_size=(128, 128),
        sigma=2.0,
        return_heatmaps=False,
    )

    rng = random.Random(args.seed)
    idxs = rng.sample(range(len(ds)), k=min(args.n, len(ds)))

    W, H = args.img_size[0], args.img_size[1]

    ok = 0
    for idx in idxs:
        s = ds[idx]
        img = (s["image"].numpy().transpose(1, 2, 0)
               * 255.0).clip(0, 255).astype(np.uint8)
        kpts = s["kpts_2d"].numpy().astype(np.float32)

        bbox = kpts_bbox_xyxy(
            kpts,
            img_wh=(W, H),
            margin=float(args.margin),
            make_square=bool(args.square),
            min_size_px=int(args.min_size),
        )
        if bbox is None:
            continue

        # BEFORE
        bgr0 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        draw_bbox(bgr0, bbox, color=(255, 0, 0))
        draw_kpts(bgr0, kpts, color=(0, 255, 0), prefix="g")

        # AFTER
        crop_rgb, kpts_crop = crop_and_resize(img, kpts, bbox, out_wh=(W, H))
        bgr1 = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        draw_kpts(bgr1, kpts_crop, color=(0, 255, 0), prefix="g")

        stem = Path(s["img_name"]).stem
        cv2.imwrite(str(out_dir / f"{stem}_0_before.jpg"), bgr0)
        cv2.imwrite(str(out_dir / f"{stem}_1_after.jpg"), bgr1)
        ok += 1

    print(f"[DONE] saved {ok} samples to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
