# src/dataset_speedplus.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import (
    POSES_TRAIN_PATH,
    POSES_VAL_PATH,
    KEYPOINTS_3D_NPY,
    load_camera_matrix,
    scale_intrinsics,
    pick_images_dir,
    LIGHTBOX_IMAGES_DIR,
    SUNLAMP_IMAGES_DIR,
    LIGHTBOX_TEST_PATH,
    SUNLAMP_TEST_PATH,
)

# -----------------------------------------------------------------------------
# Quaternion utilities
# -----------------------------------------------------------------------------


def quat_wxyz_to_rotmat(q_wxyz: np.ndarray) -> np.ndarray:
    """
    q_wxyz = [w, x, y, z]
    Returns R (3,3)
    """
    q_wxyz = np.asarray(q_wxyz, dtype=np.float32).reshape(4,)
    w, x, y, z = q_wxyz
    Rm = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
             2 * (x * z + y * w)],
            [2 * (x * y + z * w),     1 - 2 *
             (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w),     2 * (y * z + x * w),
             1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    return Rm


def project_points(
    pts_3d_obj: np.ndarray,
    R_cam_obj: np.ndarray,
    t_cam_obj: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """
    pts_3d_obj: (K,3) points in object frame (TANGO)
    R_cam_obj:  (3,3) Camera <- Object
    t_cam_obj:  (3,)  translation in camera frame
    K: (3,3)

    Returns pts_2d: (K,2) pixel coordinates
    """
    pts_3d_obj = np.asarray(pts_3d_obj, dtype=np.float32)
    # convenzione R = q (verificata)
    R_cam_obj = np.asarray(R_cam_obj, dtype=np.float32)
    t_cam_obj = np.asarray(t_cam_obj, dtype=np.float32).reshape(3,)

    Xc = (R_cam_obj @ pts_3d_obj.T).T + t_cam_obj[None, :]  # (K,3)

    x = Xc[:, 0]
    y = Xc[:, 1]
    z = Xc[:, 2].clip(min=1e-6)

    u = K[0, 0] * (x / z) + K[0, 2]
    v = K[1, 1] * (y / z) + K[1, 2]
    return np.stack([u, v], axis=1).astype(np.float32)


# -----------------------------------------------------------------------------
# Heatmap utilities
# -----------------------------------------------------------------------------
def draw_gaussian(heatmap: np.ndarray, center_xy: Tuple[float, float], sigma: float) -> None:
    """
    heatmap: (H,W) float32
    center_xy: (x,y) in heatmap coords
    """
    H, W = heatmap.shape
    x, y = center_xy

    if x < 0 or y < 0 or x >= W or y >= H:
        return

    radius = int(3 * sigma)
    x0 = int(round(x))
    y0 = int(round(y))

    x_min = max(0, x0 - radius)
    x_max = min(W - 1, x0 + radius)
    y_min = max(0, y0 - radius)
    y_max = min(H - 1, y0 + radius)

    if x_max < x_min or y_max < y_min:
        return

    xs = np.arange(x_min, x_max + 1, dtype=np.float32)
    ys = np.arange(y_min, y_max + 1, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")

    g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) /
               (2 * sigma * sigma)).astype(np.float32)
    patch = heatmap[y_min:y_max + 1, x_min:x_max + 1]
    np.maximum(patch, g, out=patch)
    heatmap[y_min:y_max + 1, x_min:x_max + 1] = patch


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------
def _as_items_list(data: Any, json_path: Path) -> List[Dict[str, Any]]:
    """Accept list or dict-with-list and return the list of items."""
    if isinstance(data, dict):
        if "images" in data and isinstance(data["images"], list):
            return data["images"]
        if "annotations" in data and isinstance(data["annotations"], list):
            return data["annotations"]
        lists = [v for v in data.values() if isinstance(v, list)]
        if not lists:
            raise ValueError(f"Unrecognized JSON format in {json_path}")
        return lists[0]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unrecognized JSON format in {json_path}")


def _extract_fname(it: Dict[str, Any]) -> Optional[str]:
    fname = it.get("filename") or it.get(
        "file_name") or it.get("img_name") or it.get("image")
    if fname is None:
        return None
    # Normalize: if json stores "images/img0001.jpg" keep only basename
    return Path(str(fname)).name


def _extract_q_t(it: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    # quaternion candidates (wxyz)
    q = (
        it.get("q_vbs2tango")
        or it.get("q")
        or it.get("quaternion")
        or it.get("q_vbs2tango_true")
    )
    # translation candidates (xyz)
    t = (
        it.get("t_vbs2tango")
        or it.get("t")
        or it.get("r_Vo2To_vbs_true")
        or it.get("translation")
    )

    if isinstance(q, dict):
        q = [q.get("w", 1.0), q.get("x", 0.0),
             q.get("y", 0.0), q.get("z", 0.0)]
    if isinstance(t, dict):
        t = [t.get("x", 0.0), t.get("y", 0.0), t.get("z", 0.0)]

    q_arr: Optional[np.ndarray] = None
    t_arr: Optional[np.ndarray] = None

    if q is not None:
        q_arr = np.array(q, dtype=np.float32).reshape(4,)
        # standardize sign
        if q_arr[0] < 0:
            q_arr = -q_arr

    if t is not None:
        t_arr = np.array(t, dtype=np.float32).reshape(3,)

    return q_arr, t_arr


def _list_images(img_dir: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    if not img_dir.exists():
        return []
    return sorted([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


def _resolve_test_image_dir(preferred: Path) -> Path:
    """
    Some SPEED+ layouts have test images in:
      - lightbox/images/*.jpg  (preferred)
    others:
      - lightbox/*.jpg         (parent)
    This picks the first directory that actually contains images.
    """
    preferred = Path(preferred)
    imgs = _list_images(preferred)
    if len(imgs) > 0:
        return preferred

    parent = preferred.parent
    imgs_parent = _list_images(parent)
    if len(imgs_parent) > 0:
        return parent

    # last resort: return preferred (error will be raised later with a clear msg)
    return preferred


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class SpeedPlusKeypointDataset(Dataset):
    """
    SPEED+ dataset wrapper for keypoint heatmap training and pose eval.

    - synthetic train/validation: reads pose JSON and projects 3D keypoints -> 2D GT.
    - lightbox/sunlamp test:
        * if test.json exists: reads pose labels too (so you can compute metrics)
        * otherwise: images-only fallback.

    Safety:
      - disallow using HIL domains with train/val splits.
    """

    def __init__(
        self,
        split: str = "train",
        domain: str = "synthetic",  # "synthetic", "lightbox", "sunlamp"
        img_size: Tuple[int, int] = (512, 512),      # (W,H)
        heatmap_size: Tuple[int, int] = (128, 128),  # (W,H)
        sigma: float = 2.0,
        return_heatmaps: bool = True,
        normalize_rgb: bool = False,
    ) -> None:
        super().__init__()

        self.split = split.lower()
        self.domain = domain.lower()

        # Safety: never allow training/validation on HIL domains
        if self.domain in ("lightbox", "sunlamp") and self.split in ("train", "training", "val", "valid", "validation"):
            raise ValueError(
                f"Refusing domain='{self.domain}' with split='{self.split}'. "
                "HIL domains are evaluation-only (use split='test')."
            )

        self.img_w, self.img_h = int(img_size[0]), int(img_size[1])
        self.hm_w, self.hm_h = int(heatmap_size[0]), int(heatmap_size[1])
        self.sigma = float(sigma)
        self.return_heatmaps = bool(return_heatmaps)
        self.normalize_rgb = bool(normalize_rgb)

        # Camera intrinsics (scaled to resized input)
        self.K0, self.W0, self.H0 = load_camera_matrix()
        self.K = scale_intrinsics(
            self.K0, (self.W0, self.H0), (self.img_w, self.img_h))

        # 3D keypoints
        self.kpts_3d = np.load(KEYPOINTS_3D_NPY).astype(np.float32)
        self.num_kpts = int(self.kpts_3d.shape[0])

        # samples
        self.samples: List[Dict[str, Any]] = self._load_samples()

        # optional normalization
        self.mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.std = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    def _load_samples(self) -> List[Dict[str, Any]]:
        # -------------------------
        # synthetic (train/val)
        # -------------------------
        if self.domain == "synthetic":
            if self.split in ("train", "training"):
                poses_path = Path(POSES_TRAIN_PATH)
            elif self.split in ("val", "valid", "validation"):
                poses_path = Path(POSES_VAL_PATH)
            else:
                raise ValueError(f"Invalid split for synthetic: {self.split}")

            if not poses_path.exists():
                raise FileNotFoundError(f"Pose JSON not found: {poses_path}")

            with open(poses_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            items = _as_items_list(data, poses_path)
            img_dir = pick_images_dir(self.split)

            samples: List[Dict[str, Any]] = []
            for it in items:
                fname = _extract_fname(it)
                if fname is None:
                    continue

                q, t = _extract_q_t(it)

                samples.append(
                    {
                        "img_path": (Path(img_dir) / fname).as_posix(),
                        "img_name": str(fname),
                        "q": q,
                        "t": t,
                    }
                )
            return samples

        # -------------------------
        # HIL test (lightbox/sunlamp)
        # -------------------------
        if self.domain in ("lightbox", "sunlamp"):
            if self.domain == "lightbox":
                poses_path = Path(LIGHTBOX_TEST_PATH)
                img_dir_pref = Path(LIGHTBOX_IMAGES_DIR)
            else:
                poses_path = Path(SUNLAMP_TEST_PATH)
                img_dir_pref = Path(SUNLAMP_IMAGES_DIR)

            img_dir = _resolve_test_image_dir(img_dir_pref)

            # Preferred: labeled test (pose available) -> compute metrics
            if poses_path.exists():
                with open(poses_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                items = _as_items_list(data, poses_path)

                samples: List[Dict[str, Any]] = []
                for it in items:
                    fname = _extract_fname(it)
                    if fname is None:
                        continue
                    q, t = _extract_q_t(it)

                    samples.append(
                        {
                            "img_path": (img_dir / fname).as_posix(),
                            "img_name": str(fname),
                            "q": q,
                            "t": t,
                        }
                    )

                # sanity: if json parsed but paths are wrong, fail early
                if len(samples) == 0:
                    raise ValueError(
                        f"No samples parsed from {poses_path}. Check JSON keys/format.")
                return samples

            # Fallback: images-only (no GT)
            img_paths = _list_images(img_dir)
            if len(img_paths) == 0:
                raise FileNotFoundError(
                    f"No images found for domain='{self.domain}'. Tried: {img_dir_pref} and {img_dir_pref.parent}"
                )
            return [{"img_path": p.as_posix(), "img_name": p.name, "q": None, "t": None} for p in img_paths]

        raise ValueError(f"Invalid domain: {self.domain}")

    # -------------------------
    # Dataset API
    # -------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def _read_image(self, path: str) -> np.ndarray:
        """Read image and return RGB uint8 resized to (img_w,img_h)."""
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Image not found/unreadable: {path}")

        # grayscale -> RGB
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img = img[..., :3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (self.img_w, self.img_h),
                         interpolation=cv2.INTER_LINEAR)
        return img

    def _image_to_tensor(self, img_rgb_uint8: np.ndarray) -> torch.Tensor:
        x = img_rgb_uint8.astype(np.float32) / 255.0
        if self.normalize_rgb:
            x = (x - self.mean) / self.std
        return torch.from_numpy(x).permute(2, 0, 1).contiguous()

    def _make_heatmaps(self, kpts_2d: np.ndarray) -> torch.Tensor:
        heatmaps = np.zeros(
            (self.num_kpts, self.hm_h, self.hm_w), dtype=np.float32)

        sx = self.hm_w / float(self.img_w)
        sy = self.hm_h / float(self.img_h)

        for i in range(self.num_kpts):
            x, y = float(kpts_2d[i, 0]), float(kpts_2d[i, 1])
            if x < 0 or y < 0 or x >= self.img_w or y >= self.img_h:
                continue
            draw_gaussian(heatmaps[i], (x * sx, y * sy), self.sigma)

        return torch.from_numpy(heatmaps)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]

        img_rgb = self._read_image(s["img_path"])
        img_t = self._image_to_tensor(img_rgb)

        q = s["q"]
        t = s["t"]

        # Compute GT 2D keypoints if pose is available
        if q is not None and t is not None:
            R_cam_obj = quat_wxyz_to_rotmat(q)
            kpts_2d = project_points(self.kpts_3d, R_cam_obj, t, self.K)
        else:
            kpts_2d = np.full((self.num_kpts, 2), -1.0, dtype=np.float32)

        out: Dict[str, Any] = {
            "image": img_t,
            "img_name": s["img_name"],
            "kpts_2d": torch.from_numpy(kpts_2d),
            "q": q if q is not None else np.array([np.nan] * 4, dtype=np.float32),
            "t": t if t is not None else np.array([np.nan] * 3, dtype=np.float32),
            # aliases for eval_pnp compatibility
            "q_vbs2tango": q if q is not None else np.array([np.nan] * 4, dtype=np.float32),
            "t_vbs2tango": t if t is not None else np.array([np.nan] * 3, dtype=np.float32),
        }

        if self.return_heatmaps:
            out["heatmaps"] = self._make_heatmaps(kpts_2d)

        return out
