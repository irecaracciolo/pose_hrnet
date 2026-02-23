# pose_hrnet

Monocular 6-DoF spacecraft pose estimation pipeline based on HRNet keypoint detection and geometric PnP solvers.

---

## Overview

This repository implements a modular pipeline for spacecraft relative pose estimation from monocular images. 

The framework combines:

- High-Resolution Network (HRNet) for 2D keypoint detection
- Perspective-n-Point (PnP) solvers for 3D pose recovery
- Optional nonlinear refinement (Levenberg–Marquardt)
- Trade-off and robustness analysis tools

The project is developed within a Master’s Thesis in Aerospace Engineering, focusing on vision-based navigation for proximity operations.

---

## Pipeline Description

The pose estimation pipeline follows a two-stage approach:

1. **Keypoint Detection**
   - Heatmap-based regression
   - Subpixel localization
   - Data augmentation strategies

2. **Pose Recovery**
   - EPnP / SQPnP solvers
   - RANSAC-based inlier filtering
   - Optional refinement stage
   - Evaluation under domain shift conditions

---

## Inputs

- Monocular spacecraft images
- 3D keypoint model (`keypoints_3d.txt` or `.npy`)
- Camera intrinsic parameters (`camera.json`)
- Trained HRNet model weights

---

## Outputs

- Estimated rotation and translation
- Reprojection error metrics
- Inlier statistics
- Evaluation logs

---

## Repository Structure

pose_hrnet/
- src/               Core scripts
- data/              Dataset main files 
- file.job           Job files to run on GPU
- requirements.txt
- README.md

---

## Installation

Install dependencies:

pip install -r requirements.txt

---

## Configuration

The file `config.py` defines: central configuration file for dataset paths, default training settings, and camera utilities.

It defines:
- Project root and SPEED+ dataset location (`DATA_DIR`)
- Paths to synthetic train/validation annotations (`train.json`, `validation.json`)
- Paths to HIL domains (lightbox/sunlamp) and their test annotations (optional)
- Location of the 3D keypoint model (`keypoints_3d.npy`)
- Output directories (`outputs/`, `checkpoints/`)

It also provides helper functions to:
- Load camera intrinsics from `camera.json` (`load_camera_matrix`)
- Scale intrinsics when resizing images (`scale_intrinsics`)
- Automatically select the correct synthetic image directory layout (`pick_images_dir`)
- Validate the dataset folder structure (`assert_dataset_layout`)

**Note:** you typically only need to update `DATA_DIR` to match your local SPEED+ dataset path.

---

## Dataset Loader

The file `dataset_speedplus.py` implements a custom PyTorch `Dataset` class for the SPEED+ dataset.

It supports both synthetic training/validation splits and HIL test domains (lightbox and sunlamp).

Main features:

- Loads camera intrinsics and scales them to the selected input resolution
- Loads 3D spacecraft keypoints
- Parses pose annotations (quaternion + translation)
- Projects 3D keypoints into 2D image coordinates
- Generates Gaussian heatmaps for keypoint training targets
- Supports evaluation-only domains without ground-truth poses

For each sample, the dataset returns:
- The resized RGB image (as a PyTorch tensor)
- 2D ground-truth keypoints (if pose is available)
- Optional heatmaps for training
- Ground-truth rotation (quaternion) and translation vectors

## Model definition

The file `model_hrnet_pose.py` implements the keypoint detection network used in the pipeline.

The model is composed of:
1. **HRNet backbone** from `timm` (feature extraction at multiple resolutions)
2. **Convolutional prediction head** that maps backbone features to **K heatmap logits**, one per keypoint

Key design choice:
- `out_index` selects the HRNet feature resolution. With a `512×512` input, `out_index=1` produces `128×128` feature maps, matching the default ground-truth heatmap resolution.

Main parameters:
- `backbone_name`: HRNet variant (default `hrnet_w64`)
- `pretrained`: enable ImageNet pretraining
- `out_index`: backbone output level used for heatmaps
- `head_channels`, `head_blocks`: capacity of the heatmap head
- `dropout_p`, `use_bn`: regularization and stability options

I/O:
- **Input:** `(B, 3, 512, 512)` float tensor
- **Output:** `(B, K, 128, 128)` heatmap logits

---

## Training

Run:
- `train_hrnet_pose.py`  
  Basic HRNet training script. 

The script trains `HRNetKeypointModel` on the SPEED+ synthetic split using heatmap supervision and (optionally) an additional coordinate-level loss.

**Key features**
- Loads SPEED+ samples through `SpeedPlusKeypointDataset` (synthetic train/val)
- Predicts K heatmap logits and applies `sigmoid` for probability heatmaps
- Loss = heatmap loss + λ(epoch) · coordinate loss
  - Heatmap loss: `mse` or `weighted_mse` (foreground up-weighting)
  - Coordinate loss: SmoothL1 between soft-argmax keypoint coordinates and GT coordinates
  - Coordinate loss is activated via a warmup + linear ramp schedule (`lambda_schedule`)
  - Optional peak-gating: coordinate loss applied only if predicted heatmap peak exceeds a threshold

**How to run**
file `spc.job`

- `train_hrnet_pose_aug.py`  
  Training with data augmentation.

  **How to run**
  file `spca.job`

- `train_hrnet_pose_aug_crop.py`  
  Training with cropping strategy.

  **How to run**
  file `train_aug_crop.job`

---

## Evaluation

Run:

- `eval_pnp.py`  
  Basic PnP pose estimation from predicted keypoints.

  **How to run**
  file `epnp.job`

- `eval_pnp_crop.py`  
  Pose estimation with bounding-box cropping.

  **How to run**
  file `epnp_crop.job`

- `eval_pnp_crop_dynamic.py`  
  Dynamic evaluation filtering: gradually decreasing #inliers for evaluation on entire dataset.

  **How to run**
  file `epnp_crop_dyn.job`

- `eval_pnp_crop_dynamic_lm_refinement.py`  
  Pose estimation with nonlinear refinement.

  **How to run**
  file `epnp_crop_dyn_lm.job`

---

### Trade-Off Analysis 

- `eval_pnp_tradeoff.py`  
  Parameter sweep and trade-off evaluation.

- `tradeoff_analysis.py`  
  Trade off study with 4 parameters: min_inliers, rmse_inliers_thr_px, min_kpts_area_px2, t_ratio_max 

  **How to run**
  file `toff.job`

  **Colormap Visualisation**
  file `plot_tradeoff_subplots.py`
  
---

## Author

Irene Caracciolo
MSc Aerospace Engineering