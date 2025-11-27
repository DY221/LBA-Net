#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LBA-Net: Lightweight Boundary-Aware Network
for Breast Ultrasound Image Segmentation

Backbone: MobileNetV3-Small
Module: Lightweight Boundary-Aware (ECA + Spatial Attention)
Author: Deng
"""
# ================= 0. 环境 & 数据路径 =================
#pip - q install segmentation - models - pytorch timm albumentations opencv - python thop matplotlib
import os, cv2, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from tqdm import tqdm
import re
from sklearn.model_selection import KFold
from thop import profile, clever_format
import time
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import KFold
from scipy.ndimage import distance_transform_edt as edt
from scipy.spatial.distance import directed_hausdorff
from monai.metrics import compute_hausdorff_distance
SEED = 42
random.seed(SEED);
np.random.seed(SEED);
torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = "/home/wang/ultrasound/Dataset_BUSI_with_GT-new"#BUET_BUSD-new
#DATA_DIR = "/home/wang/ultrasound/Dataset_BUSI_with_GT"#BUET_BUSD-new
#DATA_DIR = "/home/wang/ultrasound/BUET_BUSD"
#OUT_DIR  = "/home/wang/ultrasound/LBA-CBAMTestBUETplus"
OUT_DIR  = "/home/wang/ultrasound/LBA-CBAMTestBUSI1"
os.makedirs(OUT_DIR, exist_ok=True)

IMG_SIZE = 512
BATCH_SIZE = 24
EPOCHS = 300
NUM_WORKERS = 8
USE_AMP = True

# ================= 1. Dataset (保持不变) =================
class BUSIDataset(Dataset):
    def __init__(self, root, split='train', img_size=512):
        self.root = root
        self.split = split
        self.img_size = img_size
        cls_list = ['benign', 'malignant']
        self.imgs, self.masks = [], []
        for cls in cls_list:
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir): continue

            # 先收集所有图像文件（排除mask文件）
            image_files = []
            for fname in os.listdir(cls_dir):
                if 'mask' in fname.lower():
                    continue  # 跳过所有mask文件
                image_files.append(fname)

            # 为每个图像文件处理对应的mask
            for img_fname in image_files:
                img_path = os.path.join(cls_dir, img_fname)    
                # 基础文件名，不带后缀
                base_name = os.path.splitext(img_fname)[0]   # → 例如 benign (4)
                # 扩展名（自动适配 .png/.bmp/.jpg/.jpeg/.tif）
                ext = os.path.splitext(img_fname)[1].lower()      # 例如 ".bmp"        
                # 构造正则：匹配以下任意一种形式
                # benign (4)_mask.png
                # benign (4)_mask_1.png
                # benign (4)_mask_123.png
                pattern = re.compile(rf"^{re.escape(base_name)}_mask(_\d+)?{re.escape(ext)}$",re.IGNORECASE)
                # 查找所有相关的 mask 文件（精准匹配）
                mask_files = []
                for mask_fname in os.listdir(cls_dir):
                    if pattern.match(mask_fname):
                        mask_files.append(os.path.join(cls_dir, mask_fname))

                if not mask_files:
                    print(f"Warning: No mask found for {img_path}, skipping...")
                    continue

                # 合并所有mask（使用逻辑OR保留所有标注区域）
                merged = None
                for mp in mask_files:
                    m = cv2.imread(mp, 0)
                    if m is None:
                        print(f"Warning: Cannot read mask {mp}, skipping...")
                        continue
                    m = (m > 127).astype(np.uint8)

                    if merged is None:
                        merged = m
                    else:
                        # 使用逻辑OR合并，保留所有mask区域
                        merged = np.logical_or(merged, m).astype(np.uint8)

                if merged is not None:
                    self.imgs.append(img_path)
                    self.masks.append(merged)

                    # 调试信息：显示多mask情况
                    if len(mask_files) > 1:
                        print(f"Multi-mask: {img_fname} -> {len(mask_files)} masks merged")

        ids = list(range(len(self.imgs)));
        random.shuffle(ids)
        if split == 'all':
            self.imgs = self.imgs
            self.masks = self.masks
        else:
            ids = list(range(len(self.imgs)))
            random.shuffle(ids)
            split_idx = int(0.8 * len(ids))
            if split == 'train':
                ids = ids[:split_idx]
            else:
                ids = ids[split_idx:]
            self.imgs = [self.imgs[i] for i in ids]
            self.masks = [self.masks[i] for i in ids]
        #self.masks = [self.masks[i] for i in ids]

        if split == 'train':
            self.aug = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.3), A.Rotate(limit=30, p=0.5),
                A.RandomBrightnessContrast(p=0.4), A.GaussianBlur(blur_limit=3, p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()
            ], is_check_shapes=False)
        else:
            self.aug = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()
            ], is_check_shapes=False)
        assert len(self.imgs) == len(self.masks), f"Length mismatch: {len(self.imgs)} vs {len(self.masks)}"
        print(f"Dataset {split}: {len(self.imgs)} samples loaded.")

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.imgs[idx]), cv2.COLOR_BGR2RGB)
    # 直接使用已有的mask数组，而不是试图用cv2.imread重新读取
        mask = self.masks[idx]
        #if img.shape[:2] != mask.shape[:2]:
        #    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        if mask.sum() == 0:
            print(f"Warning: All-zero mask at {self.imgs[idx]}")
    
        aug = self.aug(image=img, mask=mask)
        img, mask = aug['image'], aug['mask']

        if isinstance(mask, torch.Tensor):
            mask = mask.unsqueeze(0).float()
        else:
            mask = torch.from_numpy(mask).unsqueeze(0).float()
    # === 新增：确保 mask 是 [0,1] 二值 ===
        mask = (mask > 0.5).float()
        bdy = cv2.morphologyEx(mask.squeeze().cpu().numpy().astype(np.uint8),
                               cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        bdy = torch.from_numpy(cv2.resize(bdy, (self.img_size, self.img_size))).unsqueeze(0).float()
    # === 新增：检查是否有 NaN/Inf ===
        if torch.isnan(img).any() or torch.isinf(img).any():
            raise ValueError(f"NaN/Inf in image: {self.imgs[idx]}")
        if torch.isnan(mask).any() or torch.isnan(bdy).any():
            raise ValueError(f"NaN in mask/bdy: {self.imgs[idx]}")
        return img, mask, bdy

    def __len__(self):
        return len(self.imgs)


# ================= 2. Metrics  =================
def dice_score(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return ((2 * inter + eps) / (union + eps)).mean().item()


def iou_score(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - inter
    return ((inter + eps) / (union + eps)).mean().item()

def recall_score(pred, target, eps=1e-6):
    """Recall = TP / (TP + FN)"""
    pred = (pred > 0.5).float()
    tp = (pred * target).sum(dim=(2, 3))
    fn = ((1 - pred) * target).sum(dim=(2, 3))
    recall = (tp + eps) / (tp + fn + eps)
    return recall.mean().item()


def hd95_score(pred, target):
    """
    使用 MONAI 计算 HD95（95% Hausdorff Distance）
    - 输入: pred, target 形状为 [B, 1, H, W] 或 [B, H, W]，值域任意（会二值化）
    - 输出: 标量 float，单位：像素（假设 spacing=1）
    """
    # 确保输入是 [B, 1, H, W]
    if pred.ndim == 3:
        pred = pred.unsqueeze(1)
    if target.ndim == 3:
        target = target.unsqueeze(1)

    # 二值化
    pred_bin = (pred > 0.5).float()
    target_bin = (target > 0.5).float()

    # MONAI 要求 one-hot 编码: [B, C, H, W], C=2 (背景 + 前景)
    pred_onehot = torch.cat([1 - pred_bin, pred_bin], dim=1)      # [B, 2, H, W]
    target_onehot = torch.cat([1 - target_bin, target_bin], dim=1)  # [B, 2, H, W]

    # 计算 HD95（前景类，即 channel=1）
    hd95_vals = compute_hausdorff_distance(
        y_pred=pred_onehot,
        y=target_onehot,
        include_background=False,  
        percentile=95,
        directed=False,             # 双向 max(hd1, hd2)
        spacing=None                # 默认像素单位（spacing=1）
    )  # 返回 [B, 2]，第0列背景，第1列前景

    # 只取前景类（index=1）的 HD95，并求 batch 平均
    hd95_foreground = hd95_vals[:, 0]  # [B]
    
    # 处理 NaN（如预测和真值都为空时 MONAI 返回 inf/NaN）
    hd95_foreground = torch.nan_to_num(hd95_foreground, nan=0.0, posinf=0.0, neginf=0.0)
    
    return hd95_foreground.mean().item()
    
    def compute_hd95(p, t):
        # Ensure numpy binary arrays
        p = (p > 0.5).astype(np.uint8)
        t = (t > 0.5).astype(np.uint8)

        if p.sum() == 0 and t.sum() == 0:
            return 0.0
        if p.sum() == 0 or t.sum() == 0:
            # Use max possible distance in image
            return np.sqrt(p.shape[0]**2 + p.shape[1]**2)

        # Get surface points
        p_surface = p - cv2.erode(p, np.ones((3,3), np.uint8), iterations=1)
        t_surface = t - cv2.erode(t, np.ones((3,3), np.uint8), iterations=1)

        p_coords = np.argwhere(p_surface)
        t_coords = np.argwhere(t_surface)

        if len(p_coords) == 0 or len(t_coords) == 0:
            return np.sqrt(p.shape[0]**2 + p.shape[1]**2)

        # Compute all pairwise distances
        from scipy.spatial.distance import cdist
        dists = cdist(p_coords, t_coords, metric='euclidean')
        hd1 = np.percentile(dists.min(axis=1), 95)
        hd2 = np.percentile(dists.min(axis=0), 95)
        return max(hd1, hd2)

    hd95_vals = []
    B = pred.shape[0]
    for i in range(B):
        p = pred[i, 0].cpu().numpy() if pred.ndim == 4 else pred[i].cpu().numpy()
        t = target[i, 0].cpu().numpy() if target.ndim == 4 else target[i].cpu().numpy()
        hd95_vals.append(compute_hd95(p, t))
    return np.mean(hd95_vals)
# ================= 3. 带边界引导的改进模型 =================
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
        self.eca = ECA(c);
        self.spa = SpatialAtt()
        self.alpha = nn.Parameter(torch.tensor(0.5));
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        return self.alpha * self.eca(x) + self.beta * self.spa(x)


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ASPP(nn.Module):
    def __init__(self, in_c, out_c=96):
        super().__init__()
        d = out_c // 4
        self.d1 = nn.Sequential(
            nn.Conv2d(in_c, d, 3, padding=6, dilation=6, bias=False), nn.BatchNorm2d(d), nn.ReLU(inplace=True)
        )
        self.d2 = nn.Sequential(
            nn.Conv2d(in_c, d, 3, padding=12, dilation=12, bias=False), nn.BatchNorm2d(d), nn.ReLU(inplace=True)
        )
        self.d3 = nn.Sequential(
            nn.Conv2d(in_c, d, 3, padding=18, dilation=18, bias=False), nn.BatchNorm2d(d), nn.ReLU(inplace=True)
        )
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_c, d, 1, bias=False), nn.ReLU(inplace=True)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_c, out_c, 1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True), nn.Dropout2d(0.1)
        )

    def forward(self, x):
        h, w = x.shape[2:]
        g = self.gap(x);
        g = F.interpolate(g, (h, w), mode='bilinear', align_corners=False)
        y = torch.cat([self.d1(x), self.d2(x), self.d3(x), g], dim=1)
        return self.fuse(y)


# ================= 边界引导模块 =================
class BoundaryGuidanceModule(nn.Module):
    """在解码器前加入边界引导，增强边界感知"""

    def __init__(self, in_channels=96, guidance_channels=32):
        super().__init__()
        self.boundary_predictor = nn.Sequential(
            nn.Conv2d(in_channels, guidance_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(guidance_channels), nn.ReLU(inplace=True),
            nn.Conv2d(guidance_channels, guidance_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(guidance_channels), nn.ReLU(inplace=True),
            nn.Conv2d(guidance_channels, 1, 1)  # 输出边界注意力图
        )

    def forward(self, x):
        # 生成边界注意力图 [B, 1, H, W]
        boundary_attention = torch.sigmoid(self.boundary_predictor(x))
        return boundary_attention


class GuidedDecoderBlock(nn.Module):
    """带有边界引导和 LBA 可选的解码块"""

    def __init__(self, in_ch, skip_ch, out_ch,
                 use_guidance=True,
                 use_lba=True):
        super().__init__()
        self.use_guidance = use_guidance
        self.use_lba = use_lba

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        if self.use_guidance:
            # 边界引导调制
            self.guidance_modulation = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1, bias=False),
                nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                nn.Conv2d(16, skip_ch, 1, bias=False),
                nn.Sigmoid()
            )

        self.se = SEBlock(in_ch + skip_ch)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )

       
        if self.use_lba:
            self.lba = LBA_Block(out_ch)
        else:
            self.lba = nn.Identity()

    def forward(self, x, skip, boundary_attention=None):
        x = self.up(x)

        # 应用边界引导（如果启用）
        if self.use_guidance and (boundary_attention is not None):
            boundary_attention_resized = F.interpolate(
                boundary_attention,
                size=skip.shape[2:],  # 匹配跳跃连接的空间尺寸
                mode='bilinear',
                align_corners=False
            )
            guidance_weight = self.guidance_modulation(boundary_attention_resized)
            guided_skip = skip * (1 + guidance_weight)  # 增强边界区域特征
            x = torch.cat([x, guided_skip], 1)
        else:
            x = torch.cat([x, skip], 1)

        x = self.se(x)
        x = self.conv(x)
        x = self.lba(x)
        return x


class LBA_Net_BoundaryGuided(nn.Module):
    """带有边界引导的改进版LBA-Net（支持消融开关）"""

    def __init__(self,
                 use_boundary_guidance=True,
                 use_boundary_head=True,
                 use_lba_block=True,
                 use_aspp=True):
        super().__init__()

        self.use_boundary_guidance = use_boundary_guidance
        self.use_boundary_head = use_boundary_head
        self.use_lba_block = use_lba_block
        self.use_aspp = use_aspp

        self.enc = timm.create_model(
            'mobilenetv3_small_100',
            pretrained=True,
            features_only=True
        )
        ch = self.enc.feature_info.channels()  # [C1, C2, C3, C4]

        # ===== ASPP =====
        if use_aspp:
            self.aspp = ASPP(ch[-1])
            aspp_out_ch = 96
        else:
            self.aspp = nn.Identity()
            aspp_out_ch = ch[-1]

        # ===== 边界引导模块 =====
        if use_boundary_guidance:
            self.boundary_guidance = BoundaryGuidanceModule(aspp_out_ch)
        else:
            self.boundary_guidance = None

        # ===== 解码器=====
        self.guided_dec4 = GuidedDecoderBlock(
            in_ch=aspp_out_ch,
            skip_ch=ch[3],
            out_ch=96,
            use_guidance=use_boundary_guidance,
            use_lba=use_lba_block
        )
        self.dec3 = GuidedDecoderBlock(
            in_ch=96,
            skip_ch=ch[2],
            out_ch=64,
            use_guidance=False,
            use_lba=use_lba_block
        )
        self.dec2 = GuidedDecoderBlock(
            in_ch=64,
            skip_ch=ch[1],
            out_ch=48,
            use_guidance=False,
            use_lba=use_lba_block
        )
        self.dec1 = GuidedDecoderBlock(
            in_ch=48,
            skip_ch=ch[0],
            out_ch=24,
            use_guidance=False,
            use_lba=use_lba_block
        )

        # ===== Dual / Single Head =====
        self.seg_head = nn.Sequential(
            nn.Conv2d(24, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1)
        )

        if use_boundary_head:
            self.bdy_head = nn.Sequential(
                nn.Conv2d(24, 16, 3, padding=1, bias=False),
                nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, 1)
            )
        else:
            self.bdy_head = None

    def forward(self, x):
        feats = self.enc(x)
        x_aspp = self.aspp(feats[-1])  # [B, aspp_out_ch, H/16, W/16]

        # 生成边界注意力图
        if self.use_boundary_guidance and (self.boundary_guidance is not None):
            boundary_attention = self.boundary_guidance(x_aspp)  # [B,1,H/16,W/16]
        else:
            # 占位张量，保证返回维度一致
            boundary_attention = torch.zeros(
                x_aspp.size(0), 1, x_aspp.size(2), x_aspp.size(3),
                device=x_aspp.device, dtype=x_aspp.dtype
            )

        # 解码过程（仅在 dec4 使用边界引导）
        x = self.guided_dec4(
            x_aspp,
            feats[3],
            boundary_attention if self.use_boundary_guidance else None
        )
        x = self.dec3(x, feats[2])
        x = self.dec2(x, feats[1])
        x = self.dec1(x, feats[0])

        seg = F.interpolate(
            self.seg_head(x),
            scale_factor=2,
            mode='bilinear',
            align_corners=False
        )

        if self.use_boundary_head and (self.bdy_head is not None):
            bdy = F.interpolate(
                self.bdy_head(x),
                scale_factor=2,
                mode='bilinear',
                align_corners=False
            )
        else:
            # 没有边界头时，用 0 占位
            bdy = torch.zeros_like(seg)

        return seg, bdy, boundary_attention



# ================= 4. 改进的损失函数 =================
bce = nn.BCEWithLogitsLoss()

def dice_loss(logits, target):
    pred = torch.sigmoid(logits)
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1 - ((2 * inter + 1e-6) / (union + 1e-6)).mean()


def focal_tversky_loss(p, g, alpha=0.3, beta=0.7, gamma=0.75):
    p = torch.sigmoid(p)
    tp = (p * g).sum(dim=(2, 3))
    fp = (p * (1 - g)).sum(dim=(2, 3))
    fn = ((1 - p) * g).sum(dim=(2, 3))
    tversky = (tp + 1e-6) / (tp + alpha * fp + beta * fn + 1e-6)
    return (1 - tversky).pow(gamma).mean()


def boundary_guided_total_loss(seg_logits, bdy_logits, boundary_attention,
                               mask, bdy,
                               use_boundary_head=True,
                               use_consistency_loss=True):
    """带有边界引导的总损失（支持消融开关）"""
    device = seg_logits.device

    # 主分割损失：始终保留
    seg_loss = bce(seg_logits, mask) + dice_loss(seg_logits, mask)

    # 边界分支损失：
    if use_boundary_head:
        bdy_loss = focal_tversky_loss(bdy_logits, bdy)
    else:
        bdy_loss = torch.zeros(1, device=device)

    # 边界一致性损失：
    if use_consistency_loss:
        boundary_consistency_loss = F.mse_loss(
            F.interpolate(boundary_attention, size=bdy.shape[2:], mode='bilinear'),
            bdy
        )
    else:
        boundary_consistency_loss = torch.zeros(1, device=device)

    total_loss = seg_loss + 0.3 * bdy_loss + 0.1 * boundary_consistency_loss

    return total_loss, {
        'seg_loss': float(seg_loss.item()),
        'bdy_loss': float(bdy_loss.item()),
        'boundary_consistency_loss': float(boundary_consistency_loss.item())
    }


# ================= 5. 可视化工具 =================
def visualize_with_guidance(model, val_loader, epoch, save_dir, num_samples=3):
    """可视化边界引导效果"""
    model.eval()
    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4 * num_samples))

    with torch.no_grad():
        for i, (imgs, masks, bdys) in enumerate(val_loader):
            if i >= num_samples: break

            imgs, masks = imgs.to(device), masks.to(device)
            seg_pred, bdy_pred, boundary_att = model(imgs)
            seg_pred = torch.sigmoid(seg_pred)

            # 原始图像
            img_np = imgs[0].cpu().permute(1, 2, 0).numpy()
            img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_np = np.clip(img_np, 0, 1)

            # 真实mask和边界
            true_mask = masks[0, 0].cpu().numpy()
            true_bdy = bdys[0, 0].cpu().numpy()

            # 预测结果
            pred_mask = seg_pred[0, 0].cpu().numpy()
            pred_bdy = torch.sigmoid(bdy_pred)[0, 0].cpu().numpy()
            boundary_att_np = boundary_att[0, 0].cpu().numpy()

            # 绘制
            axes[i, 0].imshow(img_np);
            axes[i, 0].set_title('Original');
            axes[i, 0].axis('off')
            axes[i, 1].imshow(true_mask, cmap='gray');
            axes[i, 1].set_title('GT Mask');
            axes[i, 1].axis('off')
            axes[i, 2].imshow(boundary_att_np, cmap='hot');
            axes[i, 2].set_title('Boundary Attention');
            axes[i, 2].axis('off')
            axes[i, 3].imshow(pred_mask, cmap='gray');
            axes[i, 3].set_title(f'Pred (Dice: {dice_score(seg_pred[0:1], masks[0:1]):.3f})');
            axes[i, 3].axis('off')
            axes[i, 4].imshow(img_np);
            axes[i, 4].imshow(pred_mask > 0.5, alpha=0.5, cmap='jet');
            axes[i, 4].set_title('Overlay');
            axes[i, 4].axis('off')

    plt.tight_layout()
    if isinstance(epoch, int):
        plt.savefig(f'{save_dir}/guided_visualization_epoch_{epoch:03d}.png', dpi=100, bbox_inches='tight')
    else:
        plt.savefig(f'{save_dir}/guided_visualization_{epoch}.png', dpi=100, bbox_inches='tight')
    
    plt.close()


# ================= 6. 改进的训练器 =================
def create_optimizer(model):
    """差分学习率"""
    backbone_params = [];
    decoder_params = [];
    head_params = [];
    guidance_params = []

    for name, param in model.named_parameters():
        if 'enc' in name:
            backbone_params.append(param)
        elif 'dec' in name:
            decoder_params.append(param)
        elif 'boundary_guidance' in name:
            guidance_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': 1e-4},
        {'params': decoder_params, 'lr': 2e-3},
        {'params': guidance_params, 'lr': 3e-3},  # 边界引导模块用较高学习率
        {'params': head_params, 'lr': 3e-3}
    ], weight_decay=1e-5)

    return optimizer

# ================= 7. 模型评估 =================
def evaluate_fold_model(model_path, val_indices, full_dataset):
    """评估单个fold的模型，返回该 fold 的指标均值（scalar）"""
    model = LBA_Net_BoundaryGuided().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    val_subset = torch.utils.data.Subset(full_dataset, val_indices)
    val_loader = DataLoader(val_subset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)
    
    model.eval()
    dice_scores, iou_scores, recall_scores, hd95_scores = [], [], [], []
    
    with torch.no_grad():
        for imgs, masks, _ in tqdm(val_loader, desc='Evaluating', leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            seg, _, _ = model(imgs)
            seg_pred = torch.sigmoid(seg)
            
            dice_scores.append(dice_score(seg_pred, masks))
            iou_scores.append(iou_score(seg_pred, masks))
            recall_scores.append(recall_score(seg_pred, masks))
            hd95_scores.append(hd95_score(seg_pred, masks))
    
    # 返回该 fold 的指标均值（scalar）
    return {
        'dice': np.mean(dice_scores),
        'iou': np.mean(iou_scores),
        'recall': np.mean(recall_scores),
        'hd95': np.mean(hd95_scores)
    }

def train_kfold_boundary_guided(k_folds=5):
    
    full_ds = BUSIDataset(DATA_DIR, 'all', IMG_SIZE)
    print(f"Total dataset size: {len(full_ds)}")

    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=SEED)
    fold_results = []
    fold_val_indices = []  # 保存每折验证集索引

    for fold, (train_ids, val_ids) in enumerate(kfold.split(full_ds.imgs)):
        print(f"\n{'='*50}")
        print(f"          FOLD {fold + 1}/{k_folds}")
        print(f"{'='*50}")
        
        fold_val_indices.append(val_ids)

        # 创建子数据集
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

        train_loader = DataLoader(
            full_ds, batch_size=BATCH_SIZE, sampler=train_subsampler,
            num_workers=NUM_WORKERS, pin_memory=True
        )
        val_loader = DataLoader(
            full_ds, batch_size=BATCH_SIZE, sampler=val_subsampler,
            num_workers=NUM_WORKERS, pin_memory=True, drop_last=False
        )

        # 初始化模型、优化器、调度器
        model = LBA_Net_BoundaryGuided().to(device)
        optimizer = create_optimizer(model)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=[1e-4, 2e-3, 3e-3, 3e-3],
            epochs=EPOCHS,
            steps_per_epoch=len(train_loader),
            pct_start=0.1
        )
        scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
        best_dice = 0.0
        train_losses, val_dices = [], []

        for epoch in range(EPOCHS):
            # ===== 训练阶段 =====
            model.train()
            running_loss = 0.0
            pbar = tqdm(train_loader, desc=f'Fold {fold+1} | Epoch {epoch+1}/{EPOCHS}')
            
            for imgs, masks, bdys in pbar:
                imgs, masks, bdys = imgs.to(device), masks.to(device), bdys.to(device)
                optimizer.zero_grad()
                
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    seg, bdy, boundary_att = model(imgs)
                    loss, _ = boundary_guided_total_loss(seg, bdy, boundary_att, masks, bdys)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                running_loss += loss.item() * imgs.size(0)
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

            epoch_loss = running_loss / len(train_ids)
            train_losses.append(epoch_loss)

            # ===== 验证阶段 =====
            model.eval()
            val_dice_accum, n_samples = 0.0, 0
            
            with torch.no_grad():
                for imgs, masks, _ in val_loader:
                    imgs, masks = imgs.to(device), masks.to(device)
                    with torch.cuda.amp.autocast(enabled=USE_AMP):
                        seg, _, _ = model(imgs)
                    seg_pred = torch.sigmoid(seg)
                    val_dice_accum += dice_score(seg_pred, masks) * imgs.size(0)
                    n_samples += imgs.size(0)

            # ✅ 关键点：确保 current_dice 在正确作用域内定义
            current_dice = val_dice_accum / n_samples
            val_dices.append(current_dice)
            
            print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {epoch_loss:.4f} | Val Dice: {current_dice:.4f}")

            # ===== 保存最佳模型 =====
            if current_dice > best_dice:
                best_dice = current_dice
                model_path = os.path.join(OUT_DIR, f'best_model_fold_{fold+1}.pth')
                torch.save({
                    'fold': fold,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_dice': best_dice,
                    'val_indices': val_ids,  # 保存验证索引
                }, model_path)

            # 定期可视化
            if (epoch + 1) % 25 == 0 or epoch == EPOCHS - 1:
                visualize_with_guidance(model, val_loader, f'fold{fold+1}_epoch{epoch+1}', OUT_DIR, num_samples=3)

        fold_results.append(best_dice)
        print(f"✅ Fold {fold+1} Best Val Dice: {best_dice:.4f}")

        # 绘制训练曲线
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1); plt.plot(train_losses); plt.title(f'Fold {fold+1} Train Loss')
        plt.subplot(1, 2, 2); plt.plot(val_dices); plt.title(f'Fold {fold+1} Val Dice')
        plt.savefig(os.path.join(OUT_DIR, f'training_curve_fold_{fold+1}.png'), dpi=100, bbox_inches='tight')
        plt.close()

    # 汇总结果
    mean_dice = np.mean(fold_results)
    std_dice = np.std(fold_results)
    print(f"\n{'='*60}")
    print(f"📊 5-Fold Cross Validation Results:")
    print(f"   Mean Dice: {mean_dice:.4f} ± {std_dice:.4f}")
    print(f"   Per-fold: {[f'{d:.4f}' for d in fold_results]}")
    print(f"{'='*60}")

    return mean_dice, fold_results, fold_val_indices

# ================= 主执行函数 =================
if __name__ == "__main__":
    print("=== Boundary-Guided LBA-Net with 5-Fold Cross Validation ===")
    print(f"输出目录: {OUT_DIR}")
    print(f"设备: {device}")
    print(f"图像尺寸: {IMG_SIZE}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"训练轮数: {EPOCHS}")

    # 1. 执行5折交叉验证训练
    mean_dice, fold_results, val_indices = train_kfold_boundary_guided(k_folds=5)

    # 2. 评估所有fold的最佳模型（每折一个 scalar）
    print("\n" + "="*60)
    print("📊 正在对每个 fold 的最佳模型进行 Inter-fold 评估...")
    print("="*60)
    
    # 存储每折的 scalar 指标
    fold_metrics_list = []  # list of dict, length = 5
    full_ds = BUSIDataset(DATA_DIR, 'all', IMG_SIZE)
    
    for fold in range(5):
        print(f"\n--- 评估 Fold {fold+1}/{5} ---")
        checkpoint_path = f'{OUT_DIR}/best_model_fold_{fold+1}.pth'
        
        if not os.path.exists(checkpoint_path):
            print(f"⚠️ 警告: {checkpoint_path} 不存在，跳过此 fold")
            # 可选：用 NaN 填充或中断
            fold_metrics_list.append({'dice': np.nan, 'iou': np.nan, 'recall': np.nan, 'hd95': np.nan})
            continue
            
        fold_metrics = evaluate_fold_model(checkpoint_path, val_indices[fold], full_ds)
        fold_metrics_list.append(fold_metrics)
        
        print(f"Fold {fold+1} 验证集指标:")
        print(f"  Dice:   {fold_metrics['dice']*100:.2f}%")
        print(f"  IoU:    {fold_metrics['iou']*100:.2f}%")
        print(f"  Recall: {fold_metrics['recall']*100:.2f}%")
        print(f"  HD95:   {fold_metrics['hd95']:.2f} px")

    # 3. 提取每折 scalar，计算 Inter-fold mean ± std
    dice_per_fold = [m['dice'] for m in fold_metrics_list]
    iou_per_fold = [m['iou'] for m in fold_metrics_list]
    recall_per_fold = [m['recall'] for m in fold_metrics_list]
    hd95_per_fold = [m['hd95'] for m in fold_metrics_list]

    def mean_std_str(vals, multiply=1, fmt=".2f"):
        vals = np.array(vals)
        if np.any(np.isnan(vals)):
            return "NaN"
        mean = np.mean(vals) * multiply
        std = np.std(vals, ddof=1) * multiply
        return f"{mean:{fmt}} ± {std:{fmt}}"

    print("\n" + "="*60)
    print("✅ 纯 Inter-fold 报告（每折一个 scalar）")
    print("="*60)
    print(f"Dice:   {mean_std_str(dice_per_fold, 100)} %")
    print(f"IoU:    {mean_std_str(iou_per_fold, 100)} %")
    print(f"Recall: {mean_std_str(recall_per_fold, 100)} %")
    print(f"HD95:   {mean_std_str(hd95_per_fold, 1, '.2f')} px")

    # 4. 保存结果
    summary_path = os.path.join(OUT_DIR, 'kfold_inter_fold_results.txt')
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("Pure Inter-fold Cross Validation Results\n")
        f.write("(One scalar per fold → mean ± std)\n")
        f.write("="*60 + "\n\n")
        f.write(f"Dice:   {mean_std_str(dice_per_fold, 100)} %\n")
        f.write(f"IoU:    {mean_std_str(iou_per_fold, 100)} %\n")
        f.write(f"Recall: {mean_std_str(recall_per_fold, 100)} %\n")
        f.write(f"HD95:   {mean_std_str(hd95_per_fold, 1, '.2f')} px\n\n")
        
        f.write("Per-fold scalars:\n")
        for i, m in enumerate(fold_metrics_list):
            f.write(f"Fold {i+1}: Dice={m['dice']*100:.2f}%, IoU={m['iou']*100:.2f}%, "
                    f"Recall={m['recall']*100:.2f}%, HD95={m['hd95']:.2f}px\n")
    
    print(f"\n✅ 纯 Inter-fold 结果已保存至: {summary_path}")

