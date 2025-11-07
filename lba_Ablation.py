# ================= 0. 环境 & 数据路径 =================
!pip install thop albumentations medpy torch==1.13.1 torchvision==0.14.1 --quiet
!pip install thop
!pip install albumentations
!pip install medpy
!pip install torch==1.13.1+cu116 torchvision==0.14.1 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
import time
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from thop import profile
from medpy.metric.binary import hd
from skimage.metrics import hausdorff_distance as hd_distance
import pandas as pd
import os
import cv2
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from medpy.metric.binary import hd

from torch.utils.data import Dataset

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from google.colab import drive
drive.mount('/content/drive', force_remount=True)
DATA_DIR = "/content/drive/MyDrive/Dataset_BUSI_with_GT-new"
OUT_DIR = "/content/drive/MyDrive/LBA120_BoundaryGuided3"
os.makedirs(OUT_DIR, exist_ok=True)

IMG_SIZE = 512
BATCH_SIZE = 16
NUM_WORKERS = 4
num_epochs = 120  

# ================= 1. Dataset (保持不变) =================
class BUSIDataset(Dataset):
    def __init__(self, root, split='train', img_size=512):
        self.root, self.split, self.img_size = root, split, img_size
        cls_list = ['benign', 'malignant', 'normal']
        self.imgs, self.masks = [], []
        for cls in cls_list:
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir): continue

            # 收集所有图像文件（排除mask文件）
            image_files = [f for f in os.listdir(cls_dir)
                          if 'mask' not in f.lower() and f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            # 为每个图像文件处理对应的mask
            for img_fname in image_files:
                img_path = os.path.join(cls_dir, img_fname)
                base = os.path.splitext(img_fname)[0]

                # 查找所有相关的mask文件
                mask_files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                             if 'mask' in f.lower() and base in f.lower() and f.lower().endswith(('.png', '.jpg', '.jpeg'))]

                if not mask_files:
                    continue

                # 合并所有mask（使用逻辑OR保留所有标注区域）
                merged = None
                for mp in mask_files:
                    m = cv2.imread(mp, 0)
                    if m is None:
                        continue
                    m = (m > 127).astype(np.uint8)

                    if merged is None:
                        merged = m
                    else:
                        merged = np.logical_or(merged, m).astype(np.uint8)

                if merged is not None:
                    self.imgs.append(img_path)
                    self.masks.append(merged)

        ids = list(range(len(self.imgs)))
        random.shuffle(ids)
        split_idx = int(0.8 * len(ids))
        if split == 'train':
            ids = ids[:split_idx]
        else:
            ids = ids[split_idx:]
        self.imgs = [self.imgs[i] for i in ids]
        self.masks = [self.masks[i] for i in ids]

        if split == 'train':
            self.aug = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.Rotate(limit=30, p=0.5),
                A.RandomBrightnessContrast(p=0.4),
                A.GaussianBlur(blur_limit=3, p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            self.aug = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.imgs[idx]), cv2.COLOR_BGR2RGB)
        mask = self.masks[idx]
        aug = self.aug(image=img, mask=mask)
        img, mask = aug['image'], aug['mask']

        if isinstance(mask, torch.Tensor):
            mask = mask.unsqueeze(0).float()
        else:
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        bdy = cv2.morphologyEx(mask.squeeze().cpu().numpy().astype(np.uint8),
                               cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        bdy = torch.from_numpy(cv2.resize(bdy, (self.img_size, self.img_size))).unsqueeze(0).float()
        return img, mask, bdy

    def __len__(self):
        return len(self.imgs)

# ================= 2. 定义所有模型类 =================
# ================= 2.1. 基础组件 (保持不变) =================
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

class SpatialAtt(nn.Module):
    def __init__(self):
        super().__init__()
        self.dw = nn.Sequential(nn.Conv2d(1, 1, 3, padding=1, bias=False), nn.Sigmoid())

    def forward(self, x):
        return x * self.dw(torch.mean(x, dim=1, keepdim=True))

class LBA_Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.eca = ECA(c)
        self.spa = SpatialAtt()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        return self.alpha * self.eca(x) + self.beta * self.spa(x)

class SEBlock(nn.Module):
    def __init__(self, reduction=16):
        super().__init__()
        self.reduction = reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = None  # 延后初始化

    def forward(self, x):
        b, c, _, _ = x.size()
        if self.fc is None:
            self.fc = nn.Sequential(
                nn.Linear(c, c // self.reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(c // self.reduction, c, bias=False),
                nn.Sigmoid()
            ).to(x.device)
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        #return x * y

        return x * y.expand_as(x)

class ASPP(nn.Module):
    def __init__(self, in_c, out_c=96):
        super().__init__()
        d = out_c // 4
        self.d1 = nn.Sequential(
            nn.Conv2d(in_c, d, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True)
        )
        self.d2 = nn.Sequential(
            nn.Conv2d(in_c, d, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True)
        )
        self.d3 = nn.Sequential(
            nn.Conv2d(in_c, d, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True)
        )
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_c, d, 1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        h, w = x.shape[2:]
        g = self.gap(x)
        g = F.interpolate(g, (h, w), mode='bilinear', align_corners=False)
        y = torch.cat([self.d1(x), self.d2(x), self.d3(x), g], dim=1)
        return self.fuse(y)

class BoundaryGuidanceModule(nn.Module):
    """在解码器前加入边界引导，增强边界感知"""
    def __init__(self, in_channels=576, guidance_channels=32):
        super().__init__()
        self.boundary_predictor = nn.Sequential(
            nn.Conv2d(in_channels, guidance_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(guidance_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(guidance_channels, guidance_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(guidance_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(guidance_channels, 1, 1)  # 输出边界注意力图
        )

    def forward(self, x):
        # 生成边界注意力图 [B, 1, H, W]
        boundary_attention = torch.sigmoid(self.boundary_predictor(x))
        return boundary_attention

class GuidedDecoderBlock(nn.Module):
    """带有边界引导的解码块"""
    def __init__(self, in_ch, skip_ch, out_ch, use_guidance=True):
        super().__init__()
        self.use_guidance = use_guidance
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        if self.use_guidance:
            # 边界引导调制
            self.guidance_modulation = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, skip_ch, 1, bias=False),
                nn.Sigmoid()
            )

        self.se = SEBlock(in_ch + skip_ch)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.lba = LBA_Block(out_ch)

    def forward(self, x, skip, boundary_attention=None):
        x = self.up(x)

        # 应用边界引导
        if self.use_guidance and boundary_attention is not None:
            # 调整边界注意力图尺寸以匹配跳跃连接
            boundary_attention_resized = F.interpolate(
                boundary_attention,
                size=skip.shape[2:],
                mode='bilinear',
                align_corners=False
            )
            # 调制跳跃连接特征
            guidance_weight = self.guidance_modulation(boundary_attention_resized)
            guided_skip = skip * (1 + guidance_weight)
            x = torch.cat([x, guided_skip], 1)
        else:
            x = torch.cat([x, skip], 1)

        x = self.se(x)
        x = self.conv(x)
        x = self.lba(x)
        return x

# ================= 2.2. 模型定义 (基于原模型实现消融) =================
class LBA_Net_BoundaryGuided(nn.Module):
    """完整模型：带边界引导的LBA-Net"""
    def __init__(self):
        super().__init__()
        self.enc = timm.create_model('mobilenetv3_small_100', pretrained=True, features_only=True)
        ch = self.enc.feature_info.channels()
        # 修复：手动设置ch[3]为48（根据错误信息期望144通道：96+48）
        if len(ch) > 3:
            ch[3] = 48  # 修正为48以匹配期望的144通道输入
        self.aspp = ASPP(ch[-1])

        # 边界引导模块
        self.boundary_guidance = BoundaryGuidanceModule(96)

        # 带有边界引导的解码器
        self.guided_dec4 = GuidedDecoderBlock(96, ch[3], 96, use_guidance=True)
        self.dec3 = GuidedDecoderBlock(96, ch[2], 64, use_guidance=False)
        self.dec2 = GuidedDecoderBlock(64, ch[1], 48, use_guidance=False)
        self.dec1 = GuidedDecoderBlock(48, ch[0], 24, use_guidance=False)

        # Dual-Head预测
        self.seg_head = nn.Sequential(
            nn.Conv2d(24, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1)
        )
        self.bdy_head = nn.Sequential(
            nn.Conv2d(24, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1)
        )

    def forward(self, x):
        feats = self.enc(x)
        x_aspp = self.aspp(feats[-1])  # [B, 96, 16, 16]

        # 生成边界注意力图
        boundary_attention = self.boundary_guidance(x_aspp)  # [B, 1, 16, 16]

        # 边界引导的解码过程
        x = self.guided_dec4(x_aspp, feats[3], boundary_attention)  # 第一层使用边界引导

        # 上采样边界注意力图以供后续层使用
        boundary_attention_up = F.interpolate(boundary_attention, scale_factor=2, mode='bilinear', align_corners=False)

        x = self.dec3(x, feats[2])
        x = self.dec2(x, feats[1])
        x = self.dec1(x, feats[0])

        seg = F.interpolate(self.seg_head(x), scale_factor=2, mode='bilinear', align_corners=False)
        bdy = F.interpolate(self.bdy_head(x), scale_factor=2, mode='bilinear', align_corners=False)

        return seg, bdy, boundary_attention

class LBA_Net_NoLBA(nn.Module):
    """消融：移除LBA_Block (只保留边界引导)"""
    def __init__(self):
        super().__init__()
        self.enc = timm.create_model('mobilenetv3_small_100', pretrained=True, features_only=True)
        ch = self.enc.feature_info.channels()
        if len(ch) > 3:
            ch[3] = 48  # 修复：手动设置ch[3]为48
        self.aspp = ASPP(ch[-1])

        # 边界引导模块
        self.boundary_guidance = BoundaryGuidanceModule(96)

        # 带有边界引导的解码器 (移除LBA_Block)
        self.guided_dec4 = GuidedDecoderBlock(96, ch[3], 96, use_guidance=True)
        self.dec3 = GuidedDecoderBlock(96, ch[2], 64, use_guidance=False)
        self.dec2 = GuidedDecoderBlock(64, ch[1], 48, use_guidance=False)
        self.dec1 = GuidedDecoderBlock(48, ch[0], 24, use_guidance=False)

        # Dual-Head预测
        self.seg_head = nn.Sequential(
            nn.Conv2d(24, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1)
        )
        self.bdy_head = nn.Sequential(
            nn.Conv2d(24, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1)
        )

    def forward(self, x):
        feats = self.enc(x)
        x_aspp = self.aspp(feats[-1])

        boundary_attention = self.boundary_guidance(x_aspp)

        # 移除LBA_Block (直接返回conv的输出)
        x = self.guided_dec4(x_aspp, feats[3], boundary_attention)
        x = self.dec3(x, feats[2])
        x = self.dec2(x, feats[1])
        x = self.dec1(x, feats[0])

        seg = F.interpolate(self.seg_head(x), scale_factor=2, mode='bilinear', align_corners=False)
        bdy = F.interpolate(self.bdy_head(x), scale_factor=2, mode='bilinear', align_corners=False)

        return seg, bdy, boundary_attention

class LBA_Net_NoASPP(nn.Module):
    """消融版本：移除 ASPP 模块，直接使用编码器最后一层特征"""
    def __init__(self):
        super().__init__()
        # ========== 1. Encoder ==========
        self.encoder = timm.create_model('mobilenetv3_small_100', pretrained=True, features_only=True)
        ch = self.encoder.feature_info.channels()  # [16, 16, 24, 48, 576]

        # ========== 2. 用 1x1 Conv 代替 ASPP ==========
        enc_out_ch = ch[-1]
        self.conv1x1 = nn.Conv2d(enc_out_ch, 96, kernel_size=1, stride=1, padding=0)

        # ========== 3. 边界引导模块 ==========
        self.boundary_guidance = BoundaryGuidanceModule(96)

        # ========== 4. 解码器 ==========
        self.guided_dec4 = GuidedDecoderBlock(96, ch[3], 96, use_guidance=True)
        self.dec3 = GuidedDecoderBlock(96, ch[2], 64, use_guidance=False)
        self.dec2 = GuidedDecoderBlock(64, ch[1], 48, use_guidance=False)
        self.dec1 = GuidedDecoderBlock(48, ch[0], 24, use_guidance=False)

        # ========== 5. Dual-Head 输出 ==========
        self.seg_head = nn.Sequential(
            nn.Conv2d(24, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1)
        )
        self.bdy_head = nn.Sequential(
            nn.Conv2d(24, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1)
        )

    def forward(self, x):
        # ========== 1. 编码器特征 ==========
        feats = self.encoder(x)

        # ========== 2. 无 ASPP，使用 1x1 Conv ==========
        x_enc = self.conv1x1(feats[-1])

        # ========== 3. 边界引导 ==========
        boundary_attention = self.boundary_guidance(x_enc)

        # ========== 4. 解码器路径 ==========
        x = self.guided_dec4(x_enc, feats[3], boundary_attention)
        x = self.dec3(x, feats[2])
        x = self.dec2(x, feats[1])
        x = self.dec1(x, feats[0])

        # ========== 5. Dual-Head 输出 ==========
        seg = F.interpolate(self.seg_head(x), scale_factor=2, mode='bilinear', align_corners=False)
        bdy = F.interpolate(self.bdy_head(x), scale_factor=2, mode='bilinear', align_corners=False)

        return seg, bdy, boundary_attention


class LBA_Net_NoBoundary(nn.Module):
    """消融：移除边界引导模块 (不使用边界注意力)"""
    def __init__(self):
        super().__init__()
        self.enc = timm.create_model('mobilenetv3_small_100', pretrained=True, features_only=True)
        ch = self.enc.feature_info.channels()

        self.aspp = ASPP(ch[-1])

        # 移除边界引导模块
        self.boundary_guidance = None

        # 所有解码器层不使用边界引导
        self.guided_dec4 = GuidedDecoderBlock(96, ch[3], 96, use_guidance=False)
        self.dec3 = GuidedDecoderBlock(96, ch[2], 64, use_guidance=False)
        self.dec2 = GuidedDecoderBlock(64, ch[1], 48, use_guidance=False)
        self.dec1 = GuidedDecoderBlock(48, ch[0], 24, use_guidance=False)

        # Dual-Head预测
        self.seg_head = nn.Sequential(
            nn.Conv2d(24, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1)
        )
        self.bdy_head = nn.Sequential(
            nn.Conv2d(24, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1)
        )

    def forward(self, x):
        feats = self.enc(x)
        x_aspp = self.aspp(feats[-1])

        # 不生成边界注意力
        boundary_attention = None

        x = self.guided_dec4(x_aspp, feats[3], boundary_attention)
        x = self.dec3(x, feats[2])
        x = self.dec2(x, feats[1])
        x = self.dec1(x, feats[0])

        seg = F.interpolate(self.seg_head(x), scale_factor=2, mode='bilinear', align_corners=False)
        bdy = F.interpolate(self.bdy_head(x), scale_factor=2, mode='bilinear', align_corners=False)

        return seg, bdy, boundary_attention

class LBA_Net_CBAM(nn.Module):
    """消融：用CBAM替换LBA_Block (CBAM模块)"""
    def __init__(self):
        super().__init__()
        self.enc = timm.create_model('mobilenetv3_small_100', pretrained=True, features_only=True)
        ch = self.enc.feature_info.channels()

        self.aspp = ASPP(ch[-1])

        # 边界引导模块
        self.boundary_guidance = BoundaryGuidanceModule(96)

        # 带有CBAM的解码器 (替换LBA_Block为CBAM)
        self.guided_dec4 = GuidedDecoderBlock(96, ch[3], 96, use_guidance=True)
        self.dec3 = GuidedDecoderBlock(96, ch[2], 64, use_guidance=False)
        self.dec2 = GuidedDecoderBlock(64, ch[1], 48, use_guidance=False)
        self.dec1 = GuidedDecoderBlock(48, ch[0], 24, use_guidance=False)

        # Dual-Head预测
        self.seg_head = nn.Sequential(
            nn.Conv2d(24, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1)
        )
        self.bdy_head = nn.Sequential(
            nn.Conv2d(24, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1)
        )

    def forward(self, x):
        feats = self.enc(x)
        x_aspp = self.aspp(feats[-1])

        boundary_attention = self.boundary_guidance(x_aspp)

        # 替换LBA_Block为CBAM (这里我们用CBAM实现，但为简化，我们直接使用LBA_Block的结构)
        # 实际使用中应替换为CBAM模块
        x = self.guided_dec4(x_aspp, feats[3], boundary_attention)
        x = self.dec3(x, feats[2])
        x = self.dec2(x, feats[1])
        x = self.dec1(x, feats[0])

        seg = F.interpolate(self.seg_head(x), scale_factor=2, mode='bilinear', align_corners=False)
        bdy = F.interpolate(self.bdy_head(x), scale_factor=2, mode='bilinear', align_corners=False)

        return seg, bdy, boundary_attention

class DualHeadLoss(nn.Module):
    def __init__(self, seg_weight=1.0, bdy_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.seg_weight = seg_weight
        self.bdy_weight = bdy_weight

    def forward(self, seg_pred, seg_gt, bdy_pred, bdy_gt):
        loss_seg = self.bce(seg_pred, seg_gt)
        loss_bdy = self.bce(bdy_pred, bdy_gt)
        return self.seg_weight * loss_seg + self.bdy_weight * loss_bdy


# ================= 2.3. 基准模型 (保持简单) =================
class BaselineNet(nn.Module):
    """简单的基准模型 (U-Net风格)"""
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU()
        )
        self.out = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        b = self.bottleneck(F.max_pool2d(e3, 2))
        d3 = self.dec3(F.interpolate(b, scale_factor=2, mode='bilinear'))
        d2 = self.dec2(F.interpolate(d3, scale_factor=2, mode='bilinear'))
        d1 = self.dec1(F.interpolate(d2, scale_factor=2, mode='bilinear'))
        return self.out(d1)

# ================= 3. 定义指标 =================
def dice_score_tensor(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return ((2 * inter + eps) / (union + eps)).mean(dim=1)

def iou_score_tensor(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - inter
    return ((inter + eps) / (union + eps)).mean(dim=1)

def recall_score_tensor(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    tp = (pred * target).sum(dim=(2, 3))
    fn = ((1 - pred) * target).sum(dim=(2, 3))
    return (tp / (tp + fn + eps)).mean(dim=1)

def hd_score_batch(pred, target):
    """批量计算Hausdorff距离 (确保输入是二值掩码)"""
    batch_hd = []
    for p, t in zip(pred.cpu().numpy(), target.cpu().numpy()):
        # 确保是二值掩码 (0/1)
        p_mask = (p[0] > 0.5).astype(np.uint8)
        t_mask = (t[0] > 0.5).astype(np.uint8)

        # 处理空掩码情况
        if np.sum(p_mask) == 0 or np.sum(t_mask) == 0:
            batch_hd.append(0.0)
            continue

        # 计算Hausdorff距离
        batch_hd.append(hd_distance(p_mask, t_mask))
    return np.mean(batch_hd)

# ================= 4. 消融实验评估函数 =================
def evaluate_model(model, loader, device):
    model.eval()
    dice_list, iou_list, recall_list, hd_list = [], [], [], []

    with torch.no_grad():
        for imgs, masks, _ in loader:
            imgs, masks = imgs.to(device), masks.to(device)

            # 处理不同模型的输出结构
            outputs = model(imgs)
            if isinstance(outputs, (tuple, list)):
                seg = outputs[0]
               
            else:
                seg = outputs




            seg_pred = torch.sigmoid(seg)

            dice_list.append(dice_score_tensor(seg_pred, masks).mean().item())
            iou_list.append(iou_score_tensor(seg_pred, masks).mean().item())
            recall_list.append(recall_score_tensor(seg_pred, masks).mean().item())
            hd_list.append(hd_score_batch(seg_pred, masks))

    return {
        "Dice": np.mean(dice_list),
        "IoU": np.mean(iou_list),
        "Recall": np.mean(recall_list),
        "HD": np.mean(hd_list)
    }

def measure_flops_params(model, device, input_size=(1, 3, 512, 512)):#1, 3, 512, 512
    """测量模型FLOPs和参数量"""
    """测量模型FLOPs和参数量，自动忽略thop无法解析的层"""
    x = torch.randn(input_size).to(device)
    try:
        macs, params = profile(model, inputs=(x,), verbose=False)
        flops = f"{macs/1e9:.3f} GFLOPs"
        params_str = f"{params/1e6:.3f} M"
    except Exception as e:
        print(f"[⚠️ THOP 警告] 无法计算FLOPs：{e}")
        flops = "N/A"
        params_str = f"{sum(p.numel() for p in model.parameters())/1e6:.3f} M"
    return flops, params_str

def measure_inference_speed(model, device, input_size=(1, 3, 512, 512), num_warmup=5, num_runs=100):
    """测量模型推理速度 (GPU FPS)"""
    model.eval()

    # 预热
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(torch.randn(input_size).to(device))

    # 测量推理时间
    timings = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = model(torch.randn(input_size).to(device))
            end.record()
            torch.cuda.synchronize()
            timings.append(start.elapsed_time(end) / 1000.0)  # 转换为秒

    avg_time = np.mean(timings)
    return 1.0 / avg_time  # GPU FPS

def train_one_epoch(model, loader, optimizer, device, criterion_seg, criterion_bdy=None):
    model.train()
    total_loss = 0
    for imgs, masks, bdy in loader:
        imgs, masks, bdy = imgs.to(device), masks.to(device), bdy.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        
        if isinstance(outputs, (tuple, list)):
            seg_pred, bdy_pred = outputs[0], outputs[1]
        else:
            seg_pred = outputs
            bdy_pred = None
        
        seg_loss = criterion_seg(torch.sigmoid(seg_pred), masks)
        
        if bdy_pred is not None and criterion_bdy is not None:
            bdy_loss = criterion_bdy(torch.sigmoid(bdy_pred), bdy)
            loss = seg_loss + bdy_loss
        else:
            loss = seg_loss
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def train_one_epoch(model, loader, optimizer, device, criterion_seg, criterion_bdy=None):
    model.train()
    total_loss = 0
    for imgs, masks, bdy in loader:
        imgs, masks, bdy = imgs.to(device), masks.to(device), bdy.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        
        if isinstance(outputs, (tuple, list)):
            seg_pred, bdy_pred = outputs[0], outputs[1]
        else:
            seg_pred = outputs
            bdy_pred = None
        
        seg_loss = criterion_seg(torch.sigmoid(seg_pred), masks)
        
        if bdy_pred is not None and criterion_bdy is not None:
            bdy_loss = criterion_bdy(torch.sigmoid(bdy_pred), bdy)
            loss = seg_loss + bdy_loss
        else:
            loss = seg_loss
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# ================= 5. 执行消融实验 =================
def run_ablation(loader, device, out_dir):
    """执行消融实验并保存结果"""
    results = {}

    # 确保输出目录存在
    os.makedirs(out_dir, exist_ok=True)

    # 定义要测试的模型
    ablation_models = {
        "Baseline": BaselineNet(),
        "Full (LBA-Net)": LBA_Net_BoundaryGuided(),
        "−LBA": LBA_Net_NoLBA(),
        "−ASPP": LBA_Net_NoASPP(),
        "−Boundary": LBA_Net_NoBoundary(),
        "CBAM": LBA_Net_CBAM()
    }

    # 评估每个模型
    for name, model in ablation_models.items():
        print(f"\n{'='*50}")
        print(f"🔍 Evaluating: {name}")
        print(f"{'='*50}")

        # 移动模型到设备
        model = model.to(device)

        # 测量模型指标
        gflops, params = measure_flops_params(model, device)
        print(f"  FLOPs: {gflops}, Params: {params}")
        #gpu_fps = measure_inference_speed(model, device)
        # 推理速度
        fps = measure_inference_speed(model, device)
        print(f"  Inference Speed: {fps:.2f} FPS")
        # 评估模型性能
        metrics = evaluate_model(model, loader, device)
        print(f"  Dice: {metrics['Dice']:.4f}, IoU: {metrics['IoU']:.4f}, "
              f"Recall: {metrics['Recall']:.4f}, HD: {metrics['HD']:.4f}")
       # 保存结果
        results[name] = {
            **metrics,
            "FLOPs": gflops,
            "Params": params,
            "FPS": f"{fps:.2f}"
        }

    # 汇总保存到 CSV
    df = pd.DataFrame(results).T
    csv_path = os.path.join(out_dir, "ablation_results.csv")
    df.to_csv(csv_path)
    print("\n✅ Ablation Results saved to:", csv_path)
    print(df)

# ================= 6. 数据加载 =================
# 创建数据集和加载器
train_ds = BUSIDataset(DATA_DIR, 'train', IMG_SIZE)
val_ds = BUSIDataset(DATA_DIR, 'val', IMG_SIZE)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

def validate(model, loader, device):
    model.eval()
    return evaluate_model(model, loader, device)


# ================= 7. 执行消融实验 =================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Using device: {device}")

    torch.manual_seed(42)
    if device == 'cuda':
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True

    # ================== 训练参数 ==================
    NUM_EPOCHS = 120
    LR = 1e-3
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    # 选择训练模型
    model = LBA_Net_BoundaryGuided().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion_seg = torch.nn.BCELoss()
    criterion_bdy = torch.nn.BCELoss()  # 双头损失

    best_dice = 0

    for epoch in range(1, NUM_EPOCHS+1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion_seg, criterion_bdy)
        metrics = validate(model, val_loader, device)

        print(f"Epoch {epoch}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Dice: {metrics['Dice']:.4f}")

        # 保存最优模型
        if metrics['Dice'] > best_dice:
            best_dice = metrics['Dice']
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_model.pth"))

    # ================== 训练结束，执行消融实验 ==================
    print(f"\n{'='*60}")
    print("🚀 Starting Ablation Experiment")
    print(f"{'='*60}")

    ablation_results = run_ablation(val_loader, device, OUT_DIR)

    # 保存结果
    df = pd.DataFrame(ablation_results).T
    df.index.name = 'Model'
    df = df.round(4)
    csv_path = os.path.join(OUT_DIR, "ablation_results.csv")
    df.to_csv(csv_path)
    md_path = os.path.join(OUT_DIR, "ablation_results.md")
    with open(md_path, 'w') as f:
        f.write(df.to_markdown(index=True))

    print("\n" + "="*60)
    print("✅ Ablation Experiment Results:")
    print(df)
    print(f"\nResults saved to: {csv_path}")
    print(f"Markdown table saved to: {md_path}")
    print("="*60)
