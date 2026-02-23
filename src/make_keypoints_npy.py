# src/make_keypoints_npy.py
import os
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)

INPUT_TXT = os.path.join(PROJECT_ROOT, "data", "keypoints_3d.txt")
OUTPUT_NPY = os.path.join(PROJECT_ROOT, "data", "keypoints_3d.npy")


def main():
    kpts = []
    with open(INPUT_TXT, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            x, y, z = map(float, parts[:3])
            kpts.append([x, y, z])
    kpts = np.array(kpts, dtype=np.float32)
    print("Loaded keypoints:", kpts.shape)
    np.save(OUTPUT_NPY, kpts)
    print("Saved to", OUTPUT_NPY)


if __name__ == "__main__":
    main()
