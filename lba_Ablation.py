#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LBA-Net Ablation Study Script
"""

# ========================================
# 0. 自动安装依赖
# ========================================
# ========================================
# 0. 环境安装（Colab 兼容版）
# ========================================
!pip -q install torch torchvision --index-url https://download.pytorch.org/whl/cu118
!pip -q install timm albumentations segmentation-models-pytorch thop scikit-learn opencv-python torchmetrics

# ========================================
# 1. 挂载 Google Drive + 数据路径
# ========================================
from google.colab import drive
drive.mount('/content/drive')
DATA_DIR = "/content/drive/MyDrive/Dataset_BUSI_with_GT"
SAVE_DIR = "/content/drive/MyDrive/LBA-Net_Ablation"  # 保存消融结果
os.makedirs(SAVE_DIR, exist_ok=True)

# ========================================
# 2. 导入库
# ========================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time # For FPS calculation

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ========================================
# 3. Dataset 定义 (与主脚本同步更新)
# ========================================
class BUSIDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=512):
        #self.root, self.img_size, self.split = root, img_size, split
        self.root = root_dir
        self.img_size = img_size
        self.split = split

        cls_list = ['benign', 'malignant', 'normal']
        all_imgs, all_masks, all_labels = [], [], []
        for cls in cls_list:
            cls_dir = os.path.join(self.root, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in sorted(os.listdir(cls_dir)):
                if 'mask' in fname:
                    continue
                mask_name = fname.replace('.png', '_mask.png')
                mask_path = os.path.join(cls_dir, mask_name)
                img_path  = os.path.join(cls_dir, fname)
                if os.path.exists(mask_path) and os.path.exists(img_path):
                    all_imgs.append(img_path)
                    all_masks.append(mask_path)
                    all_labels.append(cls)

        # derive case id from filename prefix
        cases = [os.path.basename(p).split('_')[0] for p in all_imgs]
        unique_cases = list(dict.fromkeys(cases))  # keep order
        case_labels = [all_labels[cases.index(c)] for c in unique_cases]

        # Stratified split by cases
        train_cases, val_cases = train_test_split(
            unique_cases, test_size=0.2, random_state=42, stratify=case_labels
        )

        is_train = [c in train_cases for c in cases]
        self.imgs  = [all_imgs[i]  for i, flag in enumerate(is_train) if flag == (split == 'train')]
        self.masks = [all_masks[i] for i, flag in enumerate(is_train) if flag == (split == 'train')]

        # augmentations
        if split == 'train':
            self.aug = A.Compose([
                A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0), ratio=(0.9, 1.1), p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Rotate(limit=20, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.GaussianBlur(p=0.2),
                A.GaussNoise(p=0.1),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
                A.GridDistortion(p=0.5),
                A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                ToTensorV2()
            ])
        else:
            self.aug = A.Compose([
                A.Resize(height=img_size, width=img_size),
                A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path  = self.imgs[idx]
        mask_path = self.masks[idx]
        assert os.path.exists(mask_path), f"mask not found: {mask_path}"

        img  = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)
        mask = (mask > 127).astype(np.uint8)   # ensure binary 0/1
        aug  = self.aug(image=img, mask=mask)
        return aug['image'], aug['mask'].float().unsqueeze(0)

# 划分训练/验证数据集（使用Dataset类的split参数）
# dataset = BUSIDataset(DATA_DIR, split='train', img_size=512) # This was causing the TypeError
train_ds = BUSIDataset(DATA_DIR, split='train', img_size=512)
val_ds   = BUSIDataset(DATA_DIR, split='val', img_size=512)


train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

print(f"训练样本数: {len(train_ds)}, 验证样本数: {len(val_ds)}")


# ========================================
# 4. 模型变体定义 (LBA-Net及其消融变体)
# ========================================
# 基础组件 (与主脚本同步)
class ECABlock(nn.Module):
    def __init__(self, channels, k=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.gap(x).squeeze(-1).transpose(-1,-2)
        y = self.conv(y)
        y = self.sigmoid(y).transpose(-1,-2).unsqueeze(-1)
        return x * y.expand_as(x)

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        attn = self.pointwise(self.depthwise(x))
        attn = torch.sigmoid(attn)
        return x * attn

class LBA_Block(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.eca = ECABlock(in_channels)
        self.spatial = SpatialAttention(in_channels)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        c = self.eca(x)
        s = self.spatial(x)
        return self.alpha * c + self.beta * s

# 解码器块 (与主脚本同步)
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, use_lba=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
        self.use_lba = use_lba
        if use_lba:
            self.lba = LBA_Block(out_ch)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.lba(self.conv(x))

# CBAM (用于变体4)
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        # 通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels//reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels//reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        # 空间注意力
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        channel_att = self.channel_att(x)
        x = x * channel_att
        # 空间注意力（平均+最大池化）
        spatial_att = torch.cat([torch.mean(x, dim=1, keepdim=True), torch.max(x, dim=1, keepdim=True)[0]], dim=1)
        spatial_att = self.spatial_att(spatial_att)
        x = x * spatial_att
        return x


# ------------------------------
# 变体0: Baseline (完整LBA-Net)
# ------------------------------
class LBANet_Baseline(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.encoder = timm.create_model("mobilenetv3_small_100", pretrained=True, features_only=True)
        enc_channels = self.encoder.feature_info.channels() # [16, 24, 40, 112, 960]

        self.aspp = torchvision.models.segmentation.deeplabv3.DeepLabHead(enc_channels[-1], 256)

        self.up4 = DecoderBlock(256, enc_channels[3], enc_channels[3], use_lba=True) # 1/32 -> 1/16
        self.up3 = DecoderBlock(enc_channels[3], enc_channels[2], enc_channels[2], use_lba=True) # 1/16 -> 1/8
        self.up2 = DecoderBlock(enc_channels[2], enc_channels[1], enc_channels[1], use_lba=True) # 1/8 -> 1/4
        self.up1 = DecoderBlock(enc_channels[1], enc_channels[0], enc_channels[0], use_lba=True) # 1/4 -> 1/2

        # Final upsampling to match input size (512x512)
        self.final_up_conv = nn.Sequential(
             nn.ConvTranspose2d(enc_channels[0], 64, kernel_size=2, stride=2), # 1/2 -> 1/1
             nn.BatchNorm2d(64),
             nn.ReLU(inplace=True)
        )


        self.seg_head = nn.Conv2d(64, num_classes, kernel_size=1)
        self.boundary_head = nn.Conv2d(64, 1, kernel_size=1)


    def forward(self, x):
        feats = self.encoder(x) # f0: 1/2, f1: 1/4, f2: 1/8, f3: 1/16, f4: 1/32
        f0, f1, f2, f3, f4 = feats

        bottleneck = self.aspp(f4) # 1/32

        d4 = self.up4(bottleneck, f3) # 1/32 -> 1/16, fuse f3
        d3 = self.up3(d4, f2)       # 1/16 -> 1/8, fuse f2
        d2 = self.up2(d3, f1)       # 1/8 -> 1/4, fuse f1
        d1 = self.up1(d2, f0)       # 1/4 -> 1/2, fuse f0

        final_feat = self.final_up_conv(d1) # 1/2 -> 1/1 (512x512)

        seg = torch.sigmoid(self.seg_head(final_feat))
        boundary = torch.sigmoid(self.boundary_head(final_feat))
        return seg, boundary

# ------------------------------
# 变体1: w/o LBA-Block
# ------------------------------
class LBANet_NoLBA(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.encoder = timm.create_model("mobilenetv3_small_100", pretrained=True, features_only=True)
        enc_channels = self.encoder.feature_info.channels()

        self.aspp = torchvision.models.segmentation.deeplabv3.DeepLabHead(enc_channels[-1], 256)

        # Decoder blocks without LBA
        self.up4 = DecoderBlock(256, enc_channels[3], enc_channels[3], use_lba=False)
        self.up3 = DecoderBlock(enc_channels[3], enc_channels[2], enc_channels[2], use_lba=False)
        self.up2 = DecoderBlock(enc_channels[2], enc_channels[1], enc_channels[1], use_lba=False)
        self.up1 = DecoderBlock(enc_channels[1], enc_channels[0], enc_channels[0], use_lba=False)

        self.final_up_conv = nn.Sequential(
             nn.ConvTranspose2d(enc_channels[0], 64, kernel_size=2, stride=2), # 1/2 -> 1/1
             nn.BatchNorm2d(64),
             nn.ReLU(inplace=True)
        )

        self.seg_head = nn.Conv2d(64, num_classes, kernel_size=1)
        self.boundary_head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        feats = self.encoder(x)
        f0, f1, f2, f3, f4 = feats

        bottleneck = self.aspp(f4)

        d4 = self.up4(bottleneck, f3)
        d3 = self.up3(d4, f2)
        d2 = self.up2(d3, f1)
        d1 = self.up1(d2, f0)

        final_feat = self.final_up_conv(d1)

        seg = torch.sigmoid(self.seg_head(final_feat))
        boundary = torch.sigmoid(self.boundary_head(final_feat))
        return seg, boundary

# ------------------------------
# 变体2: w/o ASPP
# ------------------------------
class LBANet_NoASPP(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.encoder = timm.create_model("mobilenetv3_small_100", pretrained=True, features_only=True)
        enc_channels = self.encoder.feature_info.channels()

        # Replace ASPP with a simple 1x1 conv + BN + ReLU
        self.bottleneck = nn.Sequential(
            nn.Conv2d(enc_channels[-1], 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.up4 = DecoderBlock(256, enc_channels[3], enc_channels[3], use_lba=True)
        self.up3 = DecoderBlock(enc_channels[3], enc_channels[2], enc_channels[2], use_lba=True)
        self.up2 = DecoderBlock(enc_channels[2], enc_channels[1], enc_channels[1], use_lba=True)
        self.up1 = DecoderBlock(enc_channels[1], enc_channels[0], enc_channels[0], use_lba=True)

        self.final_up_conv = nn.Sequential(
             nn.ConvTranspose2d(enc_channels[0], 64, kernel_size=2, stride=2), # 1/2 -> 1/1
             nn.BatchNorm2d(64),
             nn.ReLU(inplace=True)
        )

        self.seg_head = nn.Conv2d(64, num_classes, kernel_size=1)
        self.boundary_head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        feats = self.encoder(x)
        f0, f1, f2, f3, f4 = feats

        bottleneck = self.bottleneck(f4) # Replace ASPP

        d4 = self.up4(bottleneck, f3)
        d3 = self.up3(d4, f2)
        d2 = self.up2(d3, f1)
        d1 = self.up1(d2, f0)

        final_feat = self.final_up_conv(d1)

        seg = torch.sigmoid(self.seg_head(final_feat))
        boundary = torch.sigmoid(self.boundary_head(final_feat))
        return seg, boundary

# ------------------------------
# 变体3: w/o Boundary Head
# ------------------------------
class LBANet_NoBoundary(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.encoder = timm.create_model("mobilenetv3_small_100", pretrained=True, features_only=True)
        enc_channels = self.encoder.feature_info.channels()

        self.aspp = torchvision.models.segmentation.deeplabv3.DeepLabHead(enc_channels[-1], 256)

        self.up4 = DecoderBlock(256, enc_channels[3], enc_channels[3], use_lba=True)
        self.up3 = DecoderBlock(enc_channels[3], enc_channels[2], enc_channels[2], use_lba=True)
        self.up2 = DecoderBlock(enc_channels[2], enc_channels[1], enc_channels[1], use_lba=True)
        self.up1 = DecoderBlock(enc_channels[1], enc_channels[0], enc_channels[0], use_lba=True)

        self.final_up_conv = nn.Sequential(
             nn.ConvTranspose2d(enc_channels[0], 64, kernel_size=2, stride=2), # 1/2 -> 1/1
             nn.BatchNorm2d(64),
             nn.ReLU(inplace=True)
        )

        # Only Segmentation Head
        self.seg_head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        feats = self.encoder(x)
        f0, f1, f2, f3, f4 = feats

        bottleneck = self.aspp(f4)

        d4 = self.up4(bottleneck, f3)
        d3 = self.up3(d4, f2)
        d2 = self.up2(d3, f1)
        d1 = self.up1(d2, f0)

        final_feat = self.final_up_conv(d1)

        seg = torch.sigmoid(self.seg_head(final_feat))
        # Return only segmentation output
        return seg

# ------------------------------
# 变体4: LBA -> CBAM
# ------------------------------
class LBANet_CBAM(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.encoder = timm.create_model("mobilenetv3_small_100", pretrained=True, features_only=True)
        enc_channels = self.encoder.feature_info.channels()

        self.aspp = torchvision.models.segmentation.deeplabv3.DeepLabHead(enc_channels[-1], 256)

        # Decoder blocks with CBAM instead of LBA
        self.up4 = DecoderBlock(256, enc_channels[3], enc_channels[3], use_lba=False) # Use DecoderBlock without LBA
        self.cbam4 = CBAM(enc_channels[3]) # Add CBAM after DecoderBlock
        self.up3 = DecoderBlock(enc_channels[3], enc_channels[2], enc_channels[2], use_lba=False)
        self.cbam3 = CBAM(enc_channels[2])
        self.up2 = DecoderBlock(enc_channels[2], enc_channels[1], enc_channels[1], use_lba=False)
        self.cbam2 = CBAM(enc_channels[1])
        self.up1 = DecoderBlock(enc_channels[1], enc_channels[0], enc_channels[0], use_lba=False)
        self.cbam1 = CBAM(enc_channels[0])

        self.final_up_conv = nn.Sequential(
             nn.ConvTranspose2d(enc_channels[0], 64, kernel_size=2, stride=2), # 1/2 -> 1/1
             nn.BatchNorm2d(64),
             nn.ReLU(inplace=True)
        )

        self.seg_head = nn.Conv2d(64, num_classes, kernel_size=1)
        self.boundary_head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        feats = self.encoder(x)
        f0, f1, f2, f3, f4 = feats

        bottleneck = self.aspp(f4)

        d4 = self.up4(bottleneck, f3)
        d4 = self.cbam4(d4) # Apply CBAM
        d3 = self.up3(d4, f2)
        d3 = self.cbam3(d3)
        d2 = self.up2(d3, f1)
        d2 = self.cbam2(d2)
        d1 = self.up1(d2, f0)
        d1 = self.cbam1(d1)

        final_feat = self.final_up_conv(d1)

        seg = torch.sigmoid(self.seg_head(final_feat))
        boundary = torch.sigmoid(self.boundary_head(final_feat))
        return seg, boundary


# ========================================
# 5. 损失函数 + 评估指标 (适配所有变体)
# ========================================
def dice_loss(pred, target, eps=1e-6):
    intersection = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    dice = (2. * intersection + eps) / (union + eps)
    return 1 - dice.mean()

def bce_loss(pred, target):
    return F.binary_cross_entropy(pred, target)

def tversky_loss(pred, target, alpha=0.7, beta=0.3, eps=1e-6):
    TP = (pred * target).sum()
    FP = ((1-target) * pred).sum()
    FN = (target * (1-pred)).sum()
    tversky = (TP + eps) / (TP + alpha*FP + beta*FN + eps)
    return 1 - tversky

def total_loss(seg_pred, seg_gt, boundary_pred=None, boundary_gt=None, lam=0.3, has_boundary=True):
    """适配有无边界头的变体：has_boundary=False时仅计算分割损失"""
    seg_loss = bce_loss(seg_pred, seg_gt) + dice_loss(seg_pred, seg_gt)
    if not has_boundary:
        return seg_loss
    # Generate boundary GT if not provided for this batch (only for models with boundary head)
    if boundary_gt is None:
         boundary_gt = generate_boundary_gt(seg_gt)

    boundary_loss = tversky_loss(torch.sigmoid(boundary_pred), boundary_gt) # Apply sigmoid here
    return seg_loss + lam * boundary_loss

def generate_boundary_gt(mask):
    """Generate boundary ground truth from mask"""
    b, c, h, w = mask.shape
    batch_bdy = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    for i in range(b):
        m_np = (mask[i,0].cpu().numpy()*255).astype(np.uint8)
        bdy_np = cv2.morphologyEx(m_np, cv2.MORPH_GRADIENT, kernel)
        batch_bdy.append((bdy_np / 255.0).astype(np.float32))
    return torch.from_numpy(np.stack(batch_bdy, axis=0)).unsqueeze(1).to(mask.device)


def calculate_metrics(pred, target, threshold=0.5):
    """计算Dice、IoU（评估核心指标）"""
    pred = (pred > threshold).float()
    target = target.float()
    intersection = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    dice = (2. * intersection + 1e-6) / (union + 1e-6)
    iou = (intersection + 1e-6) / (union - intersection + 1e-6)
    return dice.mean().item(), iou.mean().item()

def count_params_flops(model, input_size=(3,512,512)):
    """统计模型参数（M）和FLOPs（G）"""
    # Ensure the model is on the correct device before profiling
    original_device = next(model.parameters()).device
    model.to(device)
    input_tensor = torch.randn(1, *input_size).to(device)

    # Need to handle models with different output shapes (like NoBoundary)
    # Profile the part of the model that is common or adjust based on variant
    # For simplicity, we will profile the full forward pass and note differences for NoBoundary
    if isinstance(model, LBANet_NoBoundary):
         # Profile the forward pass up to the segmentation head
         # This is an approximation; a more precise way would be to profile submodules
         # Or define a custom profiler for each variant.
         # For now, we'll just profile the whole thing and acknowledge the bdy_head FLOPs are small
         # and params are zero for NoBoundary variant.
         flops, params = profile(model, inputs=(input_tensor,), verbose=False)

    elif isinstance(model, (LBANet_Baseline, LBANet_NoLBA, LBANet_NoASPP, LBANet_CBAM)):
         flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    else:
         # Fallback for other potential model types
         flops, params = profile(model, inputs=(input_tensor,), verbose=False)


    # Move model back to original device if necessary
    model.to(original_device)

    return params / 1e6, flops / 1e9  # 转换为百万和十亿

def calculate_inference_fps(model, input_size=(3, 512, 512), num_runs=100):
    """计算推理FPS"""
    model.eval()
    input_tensor = torch.randn(1, *input_size).to(device)

    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            model(input_tensor)

    # Measure time
    torch.cuda.synchronize() # Ensure CUDA operations are finished
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            model(input_tensor)
    torch.cuda.synchronize() # Ensure CUDA operations are finished
    end_time = time.time()

    fps = num_runs / (end_time - start_time)
    return round(fps, 1)


# ========================================
# 6. 消融实验：统一训练与评估函数
# ========================================
def train_evaluate_model(model_name, model_class, train_loader, val_loader, has_boundary=True, epochs=80):
    """
    训练并评估单个消融变体
    model_name: 变体名称（用于记录）
    model_class: 模型类（如LBANet_Baseline）
    train_loader, val_loader: 数据加载器
    has_boundary: 是否有边界头（适配变体3）
    epochs: 训练轮次
    """
    model = model_class().to(device)

    # Separate learning rates for encoder and others
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': 1e-5},
        {'params': [p for n, p in model.named_parameters() if 'encoder' not in n], 'lr': 1e-4}
    ], weight_decay=1e-5)

    # Using ReduceLROnPlateau scheduler based on validation loss
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    old_lr = optimizer.param_groups[0]['lr']
    scheduler.step(val_loss)
    new_lr = optimizer.param_groups[0]['lr']
    if new_lr != old_lr:
      print(f"Epoch {epoch+1}: lr adjusted to {new_lr}")


    # Record training progress
    train_losses = []
    val_losses = []
    val_dices = []
    val_ious = []
    best_val_dice = 0.0
    best_model_path = os.path.join(SAVE_DIR, f"{model_name}_best.pth")
    early_stop_patience = 20 # Patience for early stopping
    early_stop_counter = 0


    print(f"\n开始训练：{model_name}")
    print("-" * 30)

    for epoch in range(epochs):
        # ------------------------------
        # 训练阶段
        # ------------------------------
        model.train()
        epoch_train_loss = 0.0
        train_loop = tqdm(train_loader, leave=False, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for imgs, masks in train_loop:
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            if has_boundary:
                seg_pred, boundary_pred = model(imgs)
                loss = total_loss(seg_pred, masks, boundary_pred=boundary_pred, has_boundary=True)
            else:
                seg_pred = model(imgs)
                loss = total_loss(seg_pred, masks, has_boundary=False)

            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

            train_loop.set_postfix(loss=loss.item())


        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ------------------------------
        # 验证阶段
        # ------------------------------
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_dice = 0.0
        epoch_val_iou = 0.0
        val_loop = tqdm(val_loader, leave=False, desc=f"Epoch {epoch+1}/{epochs} [Val]")
        with torch.no_grad():
            for imgs, masks in val_loop:
                imgs, masks = imgs.to(device), masks.to(device)

                if has_boundary:
                    seg_pred, boundary_pred = model(imgs)
                    loss = total_loss(seg_pred, masks, boundary_pred=boundary_pred, has_boundary=True)
                else:
                    seg_pred = model(imgs)
                    loss = total_loss(seg_pred, masks, has_boundary=False)

                # Calculate metrics
                dice, iou = calculate_metrics(seg_pred, masks)
                epoch_val_loss += loss.item()
                epoch_val_dice += dice
                epoch_val_iou += iou

                val_loop.set_postfix(loss=loss.item(), dice=dice, iou=iou)


        # Average validation metrics
        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_val_dice = epoch_val_dice / len(val_loader)
        avg_val_iou = epoch_val_iou / len(val_loader)
        val_losses.append(avg_val_loss)
        val_dices.append(avg_val_dice)
        val_ious.append(avg_val_iou)

        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)

        # Check for best model and early stopping
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save(model.state_dict(), best_model_path)
            early_stop_counter = 0 # Reset patience
            print(f"\nEpoch {epoch+1:3d} | Saved best model with Val Dice: {best_val_dice:.4f}")
        else:
            early_stop_counter += 1

        # Print epoch summary
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Dice: {avg_val_dice:.4f} | "
              f"Val IoU: {avg_val_iou:.4f}")

        # Check early stopping condition
        if early_stop_counter >= early_stop_patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break


    # ------------------------------
    # Final Evaluation (Load best model)
    # ------------------------------
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"\nLoaded best model for {model_name} for final evaluation.")
    else:
        print(f"\nWarning: Best model checkpoint not found for {model_name}. Using last epoch's model for final evaluation.")

    model.eval()
    final_val_dice = 0.0
    final_val_iou = 0.0
    final_eval_loop = tqdm(val_loader, leave=False, desc=f"Final Evaluation [{model_name}]")
    with torch.no_grad():
        for imgs, masks in final_eval_loop:
            imgs, masks = imgs.to(device), masks.to(device)
            if has_boundary:
                seg_pred, _ = model(imgs)
            else:
                seg_pred = model(imgs)
            dice, iou = calculate_metrics(seg_pred, masks)
            final_val_dice += dice * imgs.size(0)
            final_val_iou += iou * imgs.size(0)

    final_val_dice /= len(val_loader.dataset)
    final_val_iou /= len(val_loader.dataset)

    # Calculate efficiency metrics (Params, FLOPs, FPS)
    # Note: FPS calculation includes moving model to/from CPU/GPU if needed
    params, flops = count_params_flops(model)
    fps = calculate_inference_fps(model)


    # ------------------------------
    # Save results
    # ------------------------------
    result = {
        "Model Name": model_name,
        "Final Val Dice (%)": round(final_val_dice * 100, 2),
        "Final Val IoU (%)": round(final_val_iou * 100, 2),
        "Params (M)": round(params, 2),
        "FLOPs (G)": round(flops, 2),
        "Best Val Dice (%)": round(best_val_dice * 100, 2),
        "Train Losses": train_losses,
        "Val Losses": val_losses,
        "Val Dices": val_dices,
        "Val IoUs": val_ious,
        "FPS": fps
    }

    print(f"\n[{model_name}] 最终评估结果: "
          f"Dice={final_val_dice:.2%}, "
          f"IoU={final_val_iou:.2%}, "
          f"Params={params:.2f}M, "
          f"FLOPs={flops:.2f}G, "
          f"FPS={fps:.1f}\n")

    return result

# ========================================
# 7. 执行消融实验：训练所有变体
# ========================================
# Define all ablation variants
ablation_variants = [
    ("Baseline (Full LBA-Net)", LBANet_Baseline, True),  # 基准
    ("V1 (w/o LBA-Block)", LBANet_NoLBA, True),          # 无LBA
    ("V2 (w/o ASPP)", LBANet_NoASPP, True),              # 无ASPP
    ("V3 (w/o Boundary Head)", LBANet_NoBoundary, False),# 无边界头
    ("V4 (LBA→CBAM)", LBANet_CBAM, True)                 # LBA换CBAM
]

# Batch train and collect results
ablation_results = []
# You might want to run these one by one in Colab to manage memory and time
# For a full run, uncomment the loop below:
for model_name, model_class, has_boundary in ablation_variants:
     result = train_evaluate_model(model_name, model_class, train_loader, val_loader, has_boundary, epochs=80) # Reduced epochs for faster test
     ablation_results.append(result)


# ========================================
# 8. 消融实验结果可视化：表格 + 曲线
# ========================================
# ------------------------------
# 8.1 生成结果表格
# ------------------------------
# Extract key metrics
table_data = []
for res in ablation_results:
    table_data.append({
        "Model Variant": res["Model Name"],
        "Final Val Dice (%)": res["Final Val Dice (%)"],
        "Final Val IoU (%)": res["Final Val IoU (%)"],
        "Params (M)": res["Params (M)"],
        "FLOPs (G)": res["FLOPs (G)"],
        "FPS (GPU)": res["FPS"],
        "Best Val Dice (%)": res["Best Val Dice (%)"]
    })

# Convert to DataFrame and save
df_ablation = pd.DataFrame(table_data)
df_ablation = df_ablation.sort_values("Final Val Dice (%)", ascending=False)
csv_path = os.path.join(SAVE_DIR, "ablation_results.csv")
df_ablation.to_csv(csv_path, index=False, encoding="utf-8-sig")

# Display styled table
def highlight_max(s):
    if s.dtype == np.object_: # Handle string columns
        return [''] * len(s)
    is_max = s == s.max()
    return [f"font-weight: bold; background-color: #f0f8ff" if v else "" for v in is_max]

styled_table = df_ablation.style.apply(highlight_max, subset=["Final Val Dice (%)", "Final Val IoU (%)", "FPS (GPU)"])
# For Params and FLOPs, lower is better, so highlight min
def highlight_min(s):
     if s.dtype == np.object_:
         return [''] * len(s)
     is_min = s == s.min()
     return [f"font-weight: bold; background-color: #fffafa" if v else "" for v in is_min]

styled_table = styled_table.apply(highlight_min, subset=["Params (M)", "FLOPs (G)"])

styled_table = styled_table.set_caption("LBA-Net 消融实验结果")
print("\n消融实验结果表格：")
display(styled_table)
files.download(csv_path)

# ------------------------------
# 8.2 绘制训练曲线对比 (Loss + Dice)
# ------------------------------
plt.rcParams['font.size'] = 10
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Subplot 1: Train/Val Loss Comparison
for res in ablation_results:
    ax1.plot(res["Train Losses"], label=f"{res['Model Name']} (Train)", linestyle="-")
    ax1.plot(res["Val Losses"], label=f"{res['Model Name']} (Val)", linestyle="--")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("训练/验证损失对比")
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(alpha=0.3)

# Subplot 2: Val Dice Comparison
for res in ablation_results:
    ax2.plot(res["Val Dices"], label=res["Model Name"], marker="o", markersize=2)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("验证集 Dice")
ax2.set_title("验证集 Dice 对比")
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.grid(alpha=0.3)

plt.tight_layout()
curve_path = os.path.join(SAVE_DIR, "ablation_curves.png")
plt.savefig(curve_path, dpi=300, bbox_inches='tight')
plt.show()
files.download(curve_path)

print("消融实验代码已生成。请运行此单元格来执行实验。")
