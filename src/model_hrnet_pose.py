# src/model_hrnet_pose.py
from __future__ import annotations

import torch
import torch.nn as nn
import timm


class HRNetKeypointModel(nn.Module):
    """
    HRNet backbone (timm, features_only) + convolutional head to predict K heatmap logits.

    Output:
      heatmaps_logits: (B, K, Hm, Wm)

    Notes for SPEED+/your setup:
      - With input 512x512:
          out_index=0 -> 256x256, C=64
          out_index=1 -> 128x128, C=128   <-- matches GT heatmap_size=(128,128)
          out_index=2 -> 64x64,   C=256
          out_index=3 -> 32x32,   C=512
          out_index=4 -> 16x16,   C=1024
    """

    def __init__(
        self,
        num_keypoints: int,
        backbone_name: str = "hrnet_w64",
        pretrained: bool = True,
        out_index: int = 1,          # <-- IMPORTANT default
        head_channels: int = 256,
        head_blocks: int = 2,
        dropout_p: float = 0.1,
        use_bn: bool = True,
    ):
        super().__init__()

        if out_index not in (0, 1, 2, 3, 4):
            raise ValueError(f"out_index must be in [0..4], got {out_index}")

        self.out_index = int(out_index)

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(self.out_index,),
        )

        out_ch = self.backbone.feature_info[self.out_index]["num_chs"]

        layers: list[nn.Module] = []
        in_ch = out_ch

        # (Conv3x3 + BN + ReLU) x head_blocks
        for _ in range(max(1, int(head_blocks))):
            layers.append(
                nn.Conv2d(in_ch, head_channels, kernel_size=3,
                          stride=1, padding=1, bias=not use_bn)
            )
            if use_bn:
                layers.append(nn.BatchNorm2d(head_channels))
            layers.append(nn.ReLU(inplace=True))
            in_ch = head_channels

        if dropout_p and float(dropout_p) > 0:
            layers.append(nn.Dropout2d(p=float(dropout_p)))

        self.head = nn.Sequential(*layers)

        # Final 1x1 conv to K heatmap logits
        self.final = nn.Conv2d(in_ch, num_keypoints,
                               kernel_size=1, stride=1, padding=0, bias=True)

        # Init: keep logits small at start
        nn.init.normal_(self.final.weight, mean=0.0, std=0.001)
        nn.init.constant_(self.final.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)[0]       # (B, C, Hm, Wm)
        xh = self.head(feats)             # (B, head_channels, Hm, Wm)
        heatmaps = self.final(xh)         # (B, K, Hm, Wm)
        return heatmaps
