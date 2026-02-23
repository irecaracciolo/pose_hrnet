# src/config.py
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

# -----------------------------------------------------------------------------
# Project / dataset paths
# -----------------------------------------------------------------------------
# Questo file è in: pose_hrnet/src/config.py
# PROJECT_ROOT = pose_hrnet/
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# Dataset root: ../dataset/speedplusv2 (accanto a pose_hrnet/)
DATA_DIR: Path = (PROJECT_ROOT / ".." / "dataset" / "speedplusv2").resolve()

# Files comuni
CAMERA_JSON: Path = DATA_DIR / "camera.json"

# Synthetic (train/validation)
SYNTHETIC_DIR: Path = DATA_DIR / "synthetic"
POSES_TRAIN_PATH: Path = SYNTHETIC_DIR / "train.json"
POSES_VAL_PATH: Path = SYNTHETIC_DIR / "validation.json"

# Nota: la tua versione precedente puntava a synthetic/images.
# In SPEED+ tipicamente le immagini stanno in synthetic/images oppure in
# synthetic/train/images e synthetic/validation/images.
# Qui mettiamo una default "compatibile" e lasciamo anche le alternative.
IMAGES_DIR: Path = SYNTHETIC_DIR / "images"
IMAGES_TRAIN_DIR: Path = SYNTHETIC_DIR / "train" / "images"
IMAGES_VAL_DIR: Path = SYNTHETIC_DIR / "validation" / "images"

# Test sets (labels non disponibili)
LIGHTBOX_DIR: Path = DATA_DIR / "lightbox"
SUNLAMP_DIR: Path = DATA_DIR / "sunlamp"
LIGHTBOX_TEST_PATH: Path = LIGHTBOX_DIR / "test.json"
SUNLAMP_TEST_PATH: Path = SUNLAMP_DIR / "test.json"
LIGHTBOX_IMAGES_DIR: Path = LIGHTBOX_DIR / "images"
SUNLAMP_IMAGES_DIR: Path = SUNLAMP_DIR / "images"

# Keypoints 3D del target (TANGO)
DATA_LOCAL_DIR: Path = (PROJECT_ROOT / "data").resolve()
KEYPOINTS_3D_NPY: Path = DATA_LOCAL_DIR / "keypoints_3d.npy"

# Output folders
OUTPUTS_DIR: Path = (PROJECT_ROOT / "outputs").resolve()
CHECKPOINTS_DIR: Path = (PROJECT_ROOT / "checkpoints").resolve()
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Training / inference defaults
# -----------------------------------------------------------------------------
INPUT_SIZE = (512, 512)       # (W, H) per la CNN
HEATMAP_SIZE = (128, 128)     # (W, H) heatmap output
DEFAULT_SIGMA = 2.0

# -----------------------------------------------------------------------------
# Camera utilities
# -----------------------------------------------------------------------------


def load_camera_matrix(camera_json: Path | None = None):
    """
    Carica intrinseche K e dimensioni immagine originali del dataset (Nu,Nv).
    Ritorna:
        K: (3,3) float32
        W: int (Nu)
        H: int (Nv)

    Il tuo camera.json SPEED+ tipicamente include:
      - Nu, Nv
      - cameraMatrix (3x3)
    """
    cj = Path(camera_json) if camera_json is not None else CAMERA_JSON
    with open(cj, "r", encoding="utf-8") as f:
        cam = json.load(f)

    # Dimensioni immagine originali (SPEED+ usa Nu, Nv)
    W = int(cam.get("Nu", cam.get("width", 0)))
    H = int(cam.get("Nv", cam.get("height", 0)))
    if W <= 0 or H <= 0:
        raise ValueError(
            f"Impossibile leggere Nu/Nv (o width/height) da {cj}. "
            f"Trovati: Nu={cam.get('Nu')}, Nv={cam.get('Nv')}"
        )

    if "cameraMatrix" not in cam:
        raise KeyError(
            f"cameraMatrix non trovato in {cj}. Chiavi disponibili: {list(cam.keys())}"
        )

    K = np.array(cam["cameraMatrix"], dtype=np.float32)
    if K.shape != (3, 3):
        raise ValueError(
            f"cameraMatrix deve essere 3x3, trovato {K.shape} in {cj}")

    return K, W, H


def scale_intrinsics(K: np.ndarray, src_wh: tuple[int, int], dst_wh: tuple[int, int]) -> np.ndarray:
    """
    Scala la matrice intrinseca K dal sistema di pixel src_wh=(W,H) a dst_wh=(W,H).
    Utile quando ridimensioni le immagini (es. 1920x1200 -> 512x512) e vuoi
    usare PnP con keypoints nel frame ridimensionato.

    K' = [ fx*sx   0    cx*sx
            0    fy*sy cy*sy
            0     0      1  ]
    """
    src_w, src_h = src_wh
    dst_w, dst_h = dst_wh
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)

    K2 = K.copy().astype(np.float32)
    K2[0, 0] *= sx  # fx
    K2[1, 1] *= sy  # fy
    K2[0, 2] *= sx  # cx
    K2[1, 2] *= sy  # cy
    return K2


def pick_images_dir(split: str = "train") -> Path:
    """
    Ritorna la cartella immagini più probabile per lo split richiesto.
    Serve perché alcune versioni del dataset hanno:
      - synthetic/images
    altre:
      - synthetic/train/images, synthetic/validation/images
    """
    split = split.lower()
    if split in ("train", "training"):
        if IMAGES_TRAIN_DIR.exists():
            return IMAGES_TRAIN_DIR
        if IMAGES_DIR.exists():
            return IMAGES_DIR
        return IMAGES_TRAIN_DIR  # fallback (anche se non esiste)
    elif split in ("val", "valid", "validation"):
        if IMAGES_VAL_DIR.exists():
            return IMAGES_VAL_DIR
        if IMAGES_DIR.exists():
            return IMAGES_DIR
        return IMAGES_VAL_DIR
    else:
        # per test set scegli altrove (lightbox/sunlamp)
        return IMAGES_DIR


def assert_dataset_layout():
    """
    Piccolo check per evitare ore perse per path sbagliati.
    """
    problems = []
    if not DATA_DIR.exists():
        problems.append(f"DATA_DIR non esiste: {DATA_DIR}")
    if not CAMERA_JSON.exists():
        problems.append(f"camera.json non trovato: {CAMERA_JSON}")
    if not POSES_TRAIN_PATH.exists():
        problems.append(f"train.json non trovato: {POSES_TRAIN_PATH}")
    if not POSES_VAL_PATH.exists():
        problems.append(f"validation.json non trovato: {POSES_VAL_PATH}")
    if not LIGHTBOX_IMAGES_DIR.exists():
        problems.append(
            f"LIGHTBOX images dir non trovato: {LIGHTBOX_IMAGES_DIR}")
    if not SUNLAMP_IMAGES_DIR.exists():
        problems.append(
            f"SUNLAMP images dir non trovato: {SUNLAMP_IMAGES_DIR}")

    # immagini: basta che esista almeno una delle alternative
    if not (IMAGES_DIR.exists() or IMAGES_TRAIN_DIR.exists() or IMAGES_VAL_DIR.exists()):
        problems.append(
            "Nessuna cartella immagini trovata tra: "
            f"{IMAGES_DIR}, {IMAGES_TRAIN_DIR}, {IMAGES_VAL_DIR}"
        )
    if problems:
        msg = "Problemi layout dataset:\n  - " + "\n  - ".join(problems)
        raise FileNotFoundError(msg)
