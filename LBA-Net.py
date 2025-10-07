#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LBA-Net: Lightweight Boundary-Aware Network
for Breast Ultrasound Image Segmentation

Backbone: MobileNetV3-Small
Module: Lightweight Boundary-Aware (ECA + Spatial Attention)
Author: Adapted
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ========== 1. Efficient Channel Attention (ECA) ==========
class ECA(nn.Module):
    def __init__(self, c, k=3):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y = self.avg(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * self.sig(y)


# ========== 2. Spatial Attention (Spatial Context Enhancement) ==========
class SpatialAtt(nn.Module):
    def __init__(self):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(1, 1, 3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.dw(torch.mean(x, dim=1, keepdim=True))


# ========== 3. Lightweight Boundary-Aware Block ==========
class LBA_Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.eca = ECA(c)
        self.spa = SpatialAtt()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        return self.alpha * self.eca(x) + self.beta * self.spa(x)


# ========== 4. Decoder Block (Upsample + Skip + Conv + LBA) ==========
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.lba = LBA_Block(out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.lba(x)
        return x


# ========== 5. LBA-Net 主体结构 ==========
class LBA_Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder: MobileNetV3-Small backbone (features only)
        self.enc = timm.create_model('mobilenetv3_small_100', pretrained=True, features_only=True)
        ch = self.enc.feature_info.channels()  # [16, 24, 40, 96, 576]

        # ASPP 模块 (多尺度空洞卷积)
        self.aspp = nn.Sequential(
            nn.Conv2d(ch[-1], 96, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 1, bias=False),
            nn.BatchNorm2d(96), nn.ReLU(inplace=True)
        )

        # Decoder 部分
        self.dec4 = DecoderBlock(96, ch[3], 96)
        self.dec3 = DecoderBlock(96, ch[2], 64)
        self.dec2 = DecoderBlock(64, ch[1], 48)
        self.dec1 = DecoderBlock(48, ch[0], 24)

        # 输出头
        self.head = nn.Conv2d(24, 1, 1)

    def forward(self, x):
        feats = self.enc(x)      # F0, F1, F2, F3, F4
        x = self.aspp(feats[-1]) # ASPP on deepest feature
        x = self.dec4(x, feats[3])
        x = self.dec3(x, feats[2])
        x = self.dec2(x, feats[1])
        x = self.dec1(x, feats[0])
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.head(x)


# ========== 6. Quick Test ==========
if __name__ == "__main__":
    model = LBA_Net()
    x = torch.randn(1, 3, 512, 512)
    y = model(x)
    print("Output shape:", y.shape)

