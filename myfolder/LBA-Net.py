#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LBA-Net: Lightweight Boundary-Aware Network
for Breast Ultrasound Image Segmentation

Backbone: MobileNetV3-Small
Module: Lightweight Boundary-Aware (ECA + Spatial Attention)
Author: Deng
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# ---------- 注意力模块 ----------
class ECA(nn.Module):
    """Efficient Channel Attention"""
    def __init__(self, c, k=3):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # x: (B,c,h,w)
        y = self.avg(x)           # (B,c,1,1)
        y = self.conv(y.squeeze(-1).transpose(-1,-2)).transpose(-1,-2).unsqueeze(-1)
        return x * self.sig(y)    # 通道加权


class SpatialAtt(nn.Module):
    """Depth-wise 1×1→3×3 空间注意力"""
    def __init__(self):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(1, 1, 3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 先平均降通道→1 维，再空间加权
        att = torch.mean(x, dim=1, keepdim=True)  # (B,1,H,W)
        return x * self.dw(att)


class LBA_Block(nn.Module):
    """Learnable Balanced Attention"""
    def __init__(self, c):
        super().__init__()
        self.eca = ECA(c)
        self.spa = SpatialAtt()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta  = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        return self.alpha * self.eca(x) + self.beta * self.spa(x)


# ---------- ASPP ----------
class ASPP(nn.Module):
    def __init__(self, in_c, out_c=96):
        super().__init__()
        d = out_c // 4
        self.d1 = nn.Sequential(
            nn.Conv2d(in_c, d, 3, padding=6,  dilation=6,  bias=False),
            nn.BatchNorm2d(d), nn.ReLU(inplace=True))
        self.d2 = nn.Sequential(
            nn.Conv2d(in_c, d, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(d), nn.ReLU(inplace=True))
        self.d3 = nn.Sequential(
            nn.Conv2d(in_c, d, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(d), nn.ReLU(inplace=True))
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_c, d, 1, bias=False),
            nn.BatchNorm2d(d), nn.ReLU(inplace=True))
        self.fuse = nn.Sequential(
            nn.Conv2d(out_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))

    def forward(self, x):
        h, w = x.shape[2:]
        g = self.gap(x)
        g = F.interpolate(g, (h, w), mode='bilinear', align_corners=False)
        y = torch.cat([self.d1(x), self.d2(x), self.d3(x), g], dim=1)
        return self.fuse(y)


# ---------- 解码器 ----------
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.lba = LBA_Block(out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return self.lba(x)


# ---------- LBA-Net ----------
class LBA_Net(nn.Module):
    def __init__(self, img_size=512):
        super().__init__()
        # 编码器
        self.enc = timm.create_model(
            'mobilenetv3_small_100', pretrained=True, features_only=True)
        ch = self.enc.feature_info.channels()  # [16, 24, 48, 96, 576]

        # 中间 ASPP
        self.aspp = ASPP(ch[-1], 96)

        # 解码器
        self.dec4 = DecoderBlock(96, ch[3], 96)
        self.dec3 = DecoderBlock(96, ch[2], 64)
        self.dec2 = DecoderBlock(64, ch[1], 48)
        self.dec1 = DecoderBlock(48, ch[0], 24)

        # 双任务头
        self.seg_head = nn.Conv2d(24, 1, 1)
        self.bdy_head = nn.Conv2d(24, 1, 1)

        self.img_size = img_size

    def forward(self, x):
        feats = self.enc(x)
        x = self.aspp(feats[-1])          # 16×
        x = self.dec4(x, feats[3])        # 32×
        x = self.dec3(x, feats[2])        # 64×
        x = self.dec2(x, feats[1])        # 128×
        x = self.dec1(x, feats[0])        # 256×

        seg = F.interpolate(
            self.seg_head(x), scale_factor=2, mode='bilinear', align_corners=False)  # 512×
        bdy = F.interpolate(
            self.bdy_head(x), scale_factor=2, mode='bilinear', align_corners=False)
        return seg, bdy


# ---------- 快速自检 ----------
if __name__ == "__main__":
    net = LBA_Net().cuda()
    dummy = torch.randn(2, 3, 512, 512).cuda()
    with torch.no_grad():
        s, b = net(dummy)
    print("seg:", s.shape, "bdy:", b.shape)  # seg: (2,1,512,512)

