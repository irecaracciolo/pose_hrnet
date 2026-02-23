# src/augment_debug.py
from __future__ import annotations

import os
import random
from pathlib import Path

import cv2
import numpy as np

# pip install albumentations
import albumentations as A

from src.dataset_speedplus import SpeedPlusKeypointDataset

import inspect


def safe_aug(name: str, ctor, **kwargs):
    """
    Crea un'augmentation ignorando kwargs non supportati dalla tua versione.
    Se la classe non esiste o fallisce la validazione, ritorna None.
    """
    try:
        sig = inspect.signature(ctor)
        filt = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return ctor(**filt)
    except Exception as e:
        print(f"[SKIP] {name}: {e}")
        return None


# -------------------------
# Drawing utils
# -------------------------
def draw_kpts(img_bgr: np.ndarray, kpts_xy: np.ndarray, color=(0, 255, 0), prefix: str = ""):
    out = img_bgr.copy()
    for i, (x, y) in enumerate(kpts_xy):
        if x < 0 or y < 0:
            continue
        cv2.circle(out, (int(round(x)), int(round(y))),
                   3, color, -1, lineType=cv2.LINE_AA)
        cv2.putText(
            out,
            f"{prefix}{i}",
            (int(round(x)) + 4, int(round(y)) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def to_bgr_uint8(img_chw_01: np.ndarray) -> np.ndarray:
    # CHW float [0,1] -> HWC uint8 RGB -> BGR
    img_rgb = (np.clip(img_chw_01.transpose(1, 2, 0), 0, 1)
               * 255.0).astype(np.uint8)
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


# -------------------------
# Augmentation builders
# -------------------------
def aug_hsv():
    # alteration of hue, saturation & value
    return A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=20, p=1.0)


def aug_distortion():
    # image distortion
    return A.OneOf(
        [
            A.OpticalDistortion(distort_limit=2, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.8, p=1.0),
        ],
        p=1.0,
    )


def aug_black_patches():
    # black patches / occlusions
    return A.CoarseDropout(
        num_holes_range=(4, 10),
        hole_height_range=(16, 96),
        hole_width_range=(16, 96),
        fill=0,
        p=1.0,
    )


def aug_noise():
    # addition of noise / gaussian noise
    return A.OneOf(
        [
            A.GaussNoise(std_range=(0.02, 0.12), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ],
        p=1.0,
    )


def aug_sharpen():
    return A.Sharpen(alpha=(0.3, 0.5), lightness=(0.9, 1.6), p=1.0)


def aug_spatter():
    # a seconda della build può non esserci o cambiare firma
    aug = safe_aug(
        "Spatter",
        A.Spatter,
        mean=(0.4, 0.8),
        std=(0.3, 0.5),
        gauss_sigma=(1.0, 2.0),
        cutout_threshold=(0.3, 0.7),
        p=1.0,
    )
    return aug  # può essere None


def aug_equalize():
    # histogram equalization
    return A.Equalize(mode="cv", by_channels=True, p=1.0)


def aug_gamma():
    # gamma correction
    return A.RandomGamma(gamma_limit=(80, 160), p=1.0)


def aug_invert():
    # pixel inversion
    return A.InvertImg(p=1.0)


def aug_bit_reduction():
    # bit reduction (posterize)
    return A.Posterize(num_bits=(4, 6), p=1.0)


def aug_solar_flare():
    aug = safe_aug(
        "RandomSunFlare",
        A.RandomSunFlare,
        flare_roi=(0.0, 0.0, 1.0, 0.6),
        angle_lower=0.0,
        angle_upper=1.0,
        num_flare_circles_lower=7,
        num_flare_circles_upper=8,
        src_radius=100,
        p=1.0,
    )
    return aug  # può essere None


def aug_jitter_deformation():
    # jitter & deformation (geometric)
    return A.OneOf(
        [
            A.Affine(
                translate_percent=(-0.05, 0.05),
                scale=(0.9, 1.1),
                rotate=(-10, 10),
                shear=(-5, 5),
                p=1.0,
            ),
            A.ElasticTransform(alpha=20, sigma=6, p=1.0),
        ],
        p=1.0,
    )


def aug_brightness_contrast():
    return A.RandomBrightnessContrast(brightness_limit=0.8, contrast_limit=0.8, p=1.0)


def aug_exposure():
    # “exposure augmentation” ~ variare esposizione / highlight
    return A.RandomToneCurve(scale=0.8, p=1.0)


def aug_random_texture():
    # “random texture augmentation” (leggera): rumore + motion blur + compressione
    return A.Compose(
        [
            A.ImageCompression(quality_range=(30, 80), p=1.0),
            A.MotionBlur(blur_limit=7, p=0.7),
            A.GaussNoise(std_range=(0.02, 0.10), p=0.7),
        ],
        p=1.0,
    )


def aug_random_background():
    # Attenzione: con un’immagine già “composited” non hai una mask del satellite.
    # Qui è un FAKE “background random” (blend leggero con texture), utile solo come stress-test.
    return A.Compose(
        [
            A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.2,
                        alpha_coef=0.08, p=0.7),
            A.RandomShadow(p=0.7),
        ],
        p=1.0,
    )


def aug_blur():
    # Defocus a volte cambia / non c'è: fall back su GaussianBlur
    d = safe_aug("Defocus", A.Defocus, radius=(
        2, 5), alias_blur=(0.1, 0.5), p=1.0)
    g = A.GaussianBlur(blur_limit=(3, 9), p=1.0)
    if d is None:
        return g
    return A.OneOf([g, d], p=1.0)


# “deep augment” / “style augmentation” -> placeholder (non deep):
def aug_style_light():
    # Non è style transfer vero; è una proxy “style-like” economica.
    return A.Compose(
        [
            A.ColorJitter(brightness=0.2, contrast=0.25,
                          saturation=0.25, hue=0.05, p=1.0),
            A.ToGray(p=0.2),
        ],
        p=1.0,
    )


# -------------------------
# Main
# -------------------------
def main():
    out_dir = Path("outputs/aug_debug")
    out_dir.mkdir(parents=True, exist_ok=True)

    # scegli split/domain come vuoi
    ds = SpeedPlusKeypointDataset(
        split="val",
        domain="synthetic",
        img_size=(512, 512),
        heatmap_size=(128, 128),
        return_heatmaps=False,
    )

    idx = 0  # cambia per provare altre immagini
    sample = ds[idx]

    img_bgr = to_bgr_uint8(sample["image"].numpy())
    kpts = sample["kpts_2d"].numpy().astype(
        np.float32)  # (K,2) in pixel (512x512)

    base = draw_kpts(img_bgr, kpts, color=(0, 255, 0), prefix="g")
    cv2.imwrite(str(out_dir / f"idx{idx:06d}_ORIG.jpg"), base)

    # lista di augmentation “one-by-one”
    augs = [
        ("hsv", aug_hsv()),
        ("distortion", aug_distortion()),
        ("black_patches", aug_black_patches()),
        ("noise", aug_noise()),
        ("sharpen", aug_sharpen()),
        ("spatter", aug_spatter()),
        ("equalize", aug_equalize()),
        ("gamma", aug_gamma()),
        ("invert", aug_invert()),
        ("bit_reduction", aug_bit_reduction()),
        ("solar_flare", aug_solar_flare()),
        ("jitter_deformation", aug_jitter_deformation()),
        ("brightness_contrast", aug_brightness_contrast()),
        ("exposure", aug_exposure()),
        ("random_texture", aug_random_texture()),
        ("random_background_fake", aug_random_background()),
        ("blur", aug_blur()),
        ("style_light", aug_style_light()),
    ]

    augs = [(n, a) for (n, a) in augs if a is not None]

    # Albumentations: per trasformare anche i keypoints
    kp_params = A.KeypointParams(format="xy", remove_invisible=False)

    for name, aug in augs:
        tfm = A.Compose([aug], keypoint_params=kp_params)
        res = tfm(image=img_bgr, keypoints=[tuple(xy) for xy in kpts])

        img_aug = res["image"]
        kpts_aug = np.array(res["keypoints"], dtype=np.float32)

        vis = draw_kpts(img_aug, kpts_aug, color=(0, 0, 255), prefix="a")
        cv2.imwrite(str(out_dir / f"idx{idx:06d}_{name}.jpg"), vis)

    # esempio: pipeline random (composta)
    random_pipeline = A.Compose(
        [
            A.OneOf([aug_hsv(), aug_brightness_contrast(),
                    aug_gamma(), aug_equalize()], p=0.9),
            A.OneOf([aug_noise(), aug_blur(), aug_sharpen()], p=0.8),
            A.OneOf([aug_black_patches(), aug_distortion(),
                    aug_jitter_deformation()], p=0.7),
        ],
        keypoint_params=kp_params,
    )

    for r in range(10):
        res = random_pipeline(image=img_bgr, keypoints=[
                              tuple(xy) for xy in kpts])
        vis = draw_kpts(res["image"], np.array(
            res["keypoints"], np.float32), color=(255, 0, 0), prefix="r")
        cv2.imwrite(str(out_dir / f"idx{idx:06d}_RANDOM_{r:02d}.jpg"), vis)

    print(f"[DONE] Saved outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    # evita crash se Spatter non è disponibile nella tua versione
    try:
        main()
    except AttributeError as e:
        print("[ERROR] Albumentations operator missing:", e)
        print("Tip: aggiorna albumentations oppure rimuovi l'augmentation che manca (es. Spatter).")
