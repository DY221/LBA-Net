#消融实验
#Model Variant	       Val Dice (%)	Test Dice (%)	Test IoU (%)	Test HD95 (mm)	Params (M)	FLOPs (G)
#Baseline (Full LBA-Net)
#V1 (w/o LBA-Block)
#V2 (w/o ASPP)
#V3 (w/o Boundary Head)
#V4 (LBA→CBAM)

# ========================================
# 1. 安装依赖（保持不变）
# ========================================
!pip install -q timm albumentations segmentation-models-pytorch matplotlib opencv-python
!pip install -q thop  # 新增：安装统计参数/FLOPs的thop库
# 验证thop安装
try:
    from thop import profile
    print("thop库导入成功！")
except ImportError:
    print("thop库导入失败，请重新运行安装命令！")

import os
import cv2
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from google.colab import drive, files


# ========================================
# 2. 挂载 Google Drive + 数据路径（保持不变）
# ========================================
drive.mount('/content/drive')
DATA_DIR = "/content/drive/MyDrive/Dataset_BUSI_with_GT"
SAVE_DIR = "/content/drive/MyDrive/LBA-Net_Ablation"  # 保存消融结果
os.makedirs(SAVE_DIR, exist_ok=True)

# ========================================
# 3. 定义 Dataset 类（保持不变，确保数据一致性）
# ========================================
class BUSIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.image_paths = []
        self.mask_paths = []

        for subfolder in ["benign", "malignant", "normal"]:
            img_dir = os.path.join(root_dir, subfolder)
            for img_path in glob.glob(os.path.join(img_dir, "*.png")):
                if "_mask" in img_path:
                    continue
                mask_path = img_path.replace(".png", "_mask.png")
                if os.path.exists(mask_path):
                    self.image_paths.append(img_path)
                    self.mask_paths.append(mask_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        mask = (mask > 127).astype(np.float32)
        boundary = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3,3), np.uint8))

        if self.transform:
            augmented = self.transform(image=image, mask=mask, masks=[mask, boundary])
            image = augmented['image']
            mask = augmented['masks'][0].unsqueeze(0)  # [1,H,W]
            boundary = augmented['masks'][1].unsqueeze(0)  # [1,H,W]

        return image, mask, boundary

# ========================================
# 4. 数据增强 + 加载器（保持不变，所有变体共享数据）
# ========================================
train_transform = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.Rotate(limit=20),
    A.RandomBrightnessContrast(),
    A.GaussNoise(),
    A.Normalize(),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(),
    ToTensorV2()
])

# 划分训练/验证（固定随机种子，确保所有变体数据划分一致）
dataset = BUSIDataset(DATA_DIR, transform=train_transform)
val_size = int(0.1 * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)  # 固定种子
)
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)

print(f"训练样本数: {len(train_dataset)}, 验证样本数: {len(val_dataset)}")

# ========================================
# 5. 消融实验：修复后的模型变体（核心修改部分）
# ========================================
# 基础组件（保持不变）
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

# ------------------------------
# 变体1：Baseline（完整LBA-Net，修复尺寸匹配）
# ------------------------------
class LBANet_Baseline(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        # 1. 加载MobileNetV3-Small编码器，获取特征图通道和尺度
        self.encoder = timm.create_model("mobilenetv3_small_100", pretrained=True, features_only=True)
        self.enc_feats = self.encoder.feature_info  # 包含特征图尺度和通道信息
        enc_channels = self.enc_feats.channels()  # [16, 24, 40, 112, 960]（对应尺度：256,128,64,32,16）

        # 2. ASPP瓶颈层（输入：编码器最后一层16×16，输出256通道）
        self.aspp = torchvision.models.segmentation.deeplabv3.DeepLabHead(enc_channels[-1], 256)

        # 3. 解码器：补充1次上采样（从256×256→512×512），确保输出尺寸匹配
        # 上采样块：ConvTranspose2d（步长2，将尺寸扩大2倍）
        self.up4 = self._up_block(256, enc_channels[3])  # 16×16→32×32（匹配f3尺度32×32）
        self.att4 = LBA_Block(enc_channels[3])
        self.up3 = self._up_block(enc_channels[3], enc_channels[2])  # 32×32→64×64（匹配f2尺度64×64）
        self.att3 = LBA_Block(enc_channels[2])
        self.up2 = self._up_block(enc_channels[2], enc_channels[1])  # 64×64→128×128（匹配f1尺度128×128）
        self.att2 = LBA_Block(enc_channels[1])
        self.up1 = self._up_block(enc_channels[1], enc_channels[0])  # 128×128→256×256（匹配f0尺度256×256）
        self.att1 = LBA_Block(enc_channels[0])
        # 新增：最终上采样（256×256→512×512，与输入尺寸一致）
        self.final_up = self._up_block(enc_channels[0], enc_channels[0])

        # 4. 双头部输出（输入：最终上采样后的512×512特征图）
        self.seg_head = nn.Conv2d(enc_channels[0], num_classes, kernel_size=1)
        self.boundary_head = nn.Conv2d(enc_channels[0], 1, kernel_size=1)

    def _up_block(self, in_c, out_c):
        """上采样块：ConvTranspose2d（步长2）+ BN + ReLU（确保尺寸扩大2倍）"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0),  # 步长2→尺寸×2
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 1. 编码器提取多尺度特征（f0:256×256, f1:128×128, f2:64×64, f3:32×32, f4:16×16）
        feats = self.encoder(x)  # feats[0]→f0, feats[1]→f1, feats[2]→f2, feats[3]→f3, feats[4]→f4
        f0, f1, f2, f3, f4 = feats[0], feats[1], feats[2], feats[3], feats[4]

        # 2. ASPP瓶颈处理（f4:16×16→256通道→保持16×16）
        bottleneck = self.aspp(f4)

        # 3. 解码器：逐步上采样+特征融合（确保每步尺寸匹配）
        d4 = self.up4(bottleneck)  # 16×16→32×32（匹配f3尺寸）
        d4 = d4 + f3  # 特征融合（上采样结果 + 编码器对应层特征）
        d4 = self.att4(d4)  # LBA-Block增强

        d3 = self.up3(d4)  # 32×32→64×64（匹配f2尺寸）
        d3 = d3 + f2
        d3 = self.att3(d3)

        d2 = self.up2(d3)  # 64×64→128×128（匹配f1尺寸）
        d2 = d2 + f1
        d2 = self.att2(d2)

        d1 = self.up1(d2)  # 128×128→256×256（匹配f0尺寸）
        d1 = d1 + f0
        d1 = self.att1(d1)

        # 4. 最终上采样（256×256→512×512，与输入/标签尺寸一致）
        final_feat = self.final_up(d1)  # 256×256→512×512

        # 5. 双头部输出（512×512）
        seg = torch.sigmoid(self.seg_head(final_feat))
        boundary = torch.sigmoid(self.boundary_head(final_feat))
        return seg, boundary  # 输出尺寸：[B,1,512,512]，与标签匹配

# ------------------------------
# 变体2：w/o LBA-Block（修复尺寸匹配）
# ------------------------------
class LBANet_NoLBA(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.encoder = timm.create_model("mobilenetv3_small_100", pretrained=True, features_only=True)
        enc_channels = self.encoder.feature_info.channels()

        self.aspp = torchvision.models.segmentation.deeplabv3.DeepLabHead(enc_channels[-1], 256)
        # 解码器结构与Baseline一致，仅替换LBA-Block为1×1卷积
        self.up4 = self._up_block(256, enc_channels[3])
        self.up3 = self._up_block(enc_channels[3], enc_channels[2])
        self.up2 = self._up_block(enc_channels[2], enc_channels[1])
        self.up1 = self._up_block(enc_channels[1], enc_channels[0])
        self.final_up = self._up_block(enc_channels[0], enc_channels[0])

        # 移除LBA-Block，用1×1卷积保持通道一致性
        self.no_att = lambda c: nn.Conv2d(c, c, kernel_size=1, bias=False)
        self.att4 = self.no_att(enc_channels[3])
        self.att3 = self.no_att(enc_channels[2])
        self.att2 = self.no_att(enc_channels[1])
        self.att1 = self.no_att(enc_channels[0])

        self.seg_head = nn.Conv2d(enc_channels[0], num_classes, kernel_size=1)
        self.boundary_head = nn.Conv2d(enc_channels[0], 1, kernel_size=1)

    def _up_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        feats = self.encoder(x)
        f0, f1, f2, f3, f4 = feats[0], feats[1], feats[2], feats[3], feats[4]
        bottleneck = self.aspp(f4)

        # 解码器流程与Baseline一致
        d4 = self.up4(bottleneck) + f3
        d4 = self.att4(d4)
        d3 = self.up3(d4) + f2
        d3 = self.att3(d3)
        d2 = self.up2(d3) + f1
        d2 = self.att2(d2)
        d1 = self.up1(d2) + f0
        d1 = self.att1(d1)
        final_feat = self.final_up(d1)  # 关键：最终上采样到512×512

        seg = torch.sigmoid(self.seg_head(final_feat))
        boundary = torch.sigmoid(self.boundary_head(final_feat))
        return seg, boundary

# ------------------------------
# 变体3：w/o ASPP（修复尺寸匹配）
# ------------------------------
class LBANet_NoASPP(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.encoder = timm.create_model("mobilenetv3_small_100", pretrained=True, features_only=True)
        enc_channels = self.encoder.feature_info.channels()

        # 用1×1卷积替代ASPP（保持16×16尺度）
        self.bottleneck = nn.Sequential(
            nn.Conv2d(enc_channels[-1], 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # 解码器结构与Baseline一致
        self.up4 = self._up_block(256, enc_channels[3])
        self.att4 = LBA_Block(enc_channels[3])
        self.up3 = self._up_block(enc_channels[3], enc_channels[2])
        self.att3 = LBA_Block(enc_channels[2])
        self.up2 = self._up_block(enc_channels[2], enc_channels[1])
        self.att2 = LBA_Block(enc_channels[1])
        self.up1 = self._up_block(enc_channels[1], enc_channels[0])
        self.att1 = LBA_Block(enc_channels[0])
        self.final_up = self._up_block(enc_channels[0], enc_channels[0])

        self.seg_head = nn.Conv2d(enc_channels[0], num_classes, kernel_size=1)
        self.boundary_head = nn.Conv2d(enc_channels[0], 1, kernel_size=1)

    def _up_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        feats = self.encoder(x)
        f0, f1, f2, f3, f4 = feats[0], feats[1], feats[2], feats[3], feats[4]
        bottleneck = self.bottleneck(f4)  # 替代ASPP，保持16×16尺度

        # 解码器流程与Baseline一致，最终上采样到512×512
        d4 = self.up4(bottleneck) + f3
        d4 = self.att4(d4)
        d3 = self.up3(d4) + f2
        d3 = self.att3(d3)
        d2 = self.up2(d3) + f1
        d2 = self.att2(d2)
        d1 = self.up1(d2) + f0
        d1 = self.att1(d1)
        final_feat = self.final_up(d1)

        seg = torch.sigmoid(self.seg_head(final_feat))
        boundary = torch.sigmoid(self.boundary_head(final_feat))
        return seg, boundary

# ------------------------------
# 变体4：w/o Boundary Head（修复尺寸匹配）
# ------------------------------
class LBANet_NoBoundary(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.encoder = timm.create_model("mobilenetv3_small_100", pretrained=True, features_only=True)
        enc_channels = self.encoder.feature_info.channels()

        self.aspp = torchvision.models.segmentation.deeplabv3.DeepLabHead(enc_channels[-1], 256)
        # 解码器结构与Baseline一致
        self.up4 = self._up_block(256, enc_channels[3])
        self.att4 = LBA_Block(enc_channels[3])
        self.up3 = self._up_block(enc_channels[3], enc_channels[2])
        self.att3 = LBA_Block(enc_channels[2])
        self.up2 = self._up_block(enc_channels[2], enc_channels[1])
        self.att2 = LBA_Block(enc_channels[1])
        self.up1 = self._up_block(enc_channels[1], enc_channels[0])
        self.att1 = LBA_Block(enc_channels[0])
        self.final_up = self._up_block(enc_channels[0], enc_channels[0])

        # 仅保留分割头
        self.seg_head = nn.Conv2d(enc_channels[0], num_classes, kernel_size=1)

    def _up_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        feats = self.encoder(x)
        f0, f1, f2, f3, f4 = feats[0], feats[1], feats[2], feats[3], feats[4]
        bottleneck = self.aspp(f4)

        # 解码器流程与Baseline一致，最终上采样到512×512
        d4 = self.up4(bottleneck) + f3
        d4 = self.att4(d4)
        d3 = self.up3(d4) + f2
        d3 = self.att3(d3)
        d2 = self.up2(d3) + f1
        d2 = self.att2(d2)
        d1 = self.up1(d2) + f0
        d1 = self.att1(d1)
        final_feat = self.final_up(d1)

        seg = torch.sigmoid(self.seg_head(final_feat))
        return seg  # 输出尺寸：[B,1,512,512]，与标签匹配

# ------------------------------
# 变体5：LBA→CBAM（修复尺寸匹配）
# ------------------------------
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

class LBANet_CBAM(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.encoder = timm.create_model("mobilenetv3_small_100", pretrained=True, features_only=True)
        enc_channels = self.encoder.feature_info.channels()

        self.aspp = torchvision.models.segmentation.deeplabv3.DeepLabHead(enc_channels[-1], 256)
        # 解码器结构与Baseline一致，仅替换LBA-Block为CBAM
        self.up4 = self._up_block(256, enc_channels[3])
        self.att4 = CBAM(enc_channels[3])
        self.up3 = self._up_block(enc_channels[3], enc_channels[2])
        self.att3 = CBAM(enc_channels[2])
        self.up2 = self._up_block(enc_channels[2], enc_channels[1])
        self.att2 = CBAM(enc_channels[1])
        self.up1 = self._up_block(enc_channels[1], enc_channels[0])
        self.att1 = CBAM(enc_channels[0])
        self.final_up = self._up_block(enc_channels[0], enc_channels[0])

        self.seg_head = nn.Conv2d(enc_channels[0], num_classes, kernel_size=1)
        self.boundary_head = nn.Conv2d(enc_channels[0], 1, kernel_size=1)

    def _up_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        feats = self.encoder(x)
        f0, f1, f2, f3, f4 = feats[0], feats[1], feats[2], feats[3], feats[4]
        bottleneck = self.aspp(f4)

        # 解码器流程与Baseline一致，最终上采样到512×512
        d4 = self.up4(bottleneck) + f3
        d4 = self.att4(d4)
        d3 = self.up3(d4) + f2
        d3 = self.att3(d3)
        d2 = self.up2(d3) + f1
        d2 = self.att2(d2)
        d1 = self.up1(d2) + f0
        d1 = self.att1(d1)
        final_feat = self.final_up(d1)

        seg = torch.sigmoid(self.seg_head(final_feat))
        boundary = torch.sigmoid(self.boundary_head(final_feat))
        return seg, boundary
# ========================================
# 6. 损失函数 + 评估指标（适配所有变体）
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

def total_loss(seg_pred, seg_gt, boundary_pred=None, boundary_gt=None, lam=0.5, has_boundary=True):
    """适配有无边界头的变体：has_boundary=False时仅计算分割损失"""
    seg_loss = bce_loss(seg_pred, seg_gt) + dice_loss(seg_pred, seg_gt)
    if not has_boundary:
        return seg_loss
    boundary_loss = tversky_loss(boundary_pred, boundary_gt)
    return seg_loss + lam * boundary_loss

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
    from thop import profile
    device = next(model.parameters()).device
    input_tensor = torch.randn(1, *input_size).to(device)
    flops, params = profile(model, inputs=(input_tensor,))
    return params / 1e6, flops / 1e9  # 转换为百万和十亿

# ========================================
# 7. 消融实验：统一训练与评估函数（所有变体共享）
# ========================================
def train_evaluate_model(model_name, model_class, has_boundary=True, epochs=50):
    """
    训练并评估单个消融变体
    model_name: 变体名称（用于记录）
    model_class: 模型类（如LBANet_Baseline）
    has_boundary: 是否有边界头（适配变体4）
    epochs: 训练轮次（原代码20，建议改为50+以稳定收敛）
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 记录训练过程
    train_losses = []
    val_losses = []
    val_dices = []
    val_ious = []
    best_val_dice = 0.0
    best_model_path = os.path.join(SAVE_DIR, f"{model_name}_best.pth")

    for epoch in range(epochs):
        # ------------------------------
        # 训练阶段
        # ------------------------------
        model.train()
        epoch_train_loss = 0.0
        for imgs, masks, boundaries in train_loader:
            imgs, masks, boundaries = imgs.to(device), masks.to(device), boundaries.to(device)

            optimizer.zero_grad()
            if has_boundary:
                seg_pred, boundary_pred = model(imgs)
                loss = total_loss(seg_pred, masks, boundary_pred, boundaries, has_boundary=True)
            else:
                seg_pred = model(imgs)
                loss = total_loss(seg_pred, masks, has_boundary=False)

            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ------------------------------
        # 验证阶段
        # ------------------------------
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_dice = 0.0
        epoch_val_iou = 0.0
        with torch.no_grad():
            for imgs, masks, boundaries in val_loader:
                imgs, masks, boundaries = imgs.to(device), masks.to(device), boundaries.to(device)

                if has_boundary:
                    seg_pred, boundary_pred = model(imgs)
                    loss = total_loss(seg_pred, masks, boundary_pred, boundaries, has_boundary=True)
                else:
                    seg_pred = model(imgs)
                    loss = total_loss(seg_pred, masks, has_boundary=False)

                # 计算指标
                dice, iou = calculate_metrics(seg_pred, masks)
                epoch_val_loss += loss.item()
                epoch_val_dice += dice
                epoch_val_iou += iou

        # 平均验证指标
        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_val_dice = epoch_val_dice / len(val_loader)
        avg_val_iou = epoch_val_iou / len(val_loader)
        val_losses.append(avg_val_loss)
        val_dices.append(avg_val_dice)
        val_ious.append(avg_val_iou)

        # 保存最优模型
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save(model.state_dict(), best_model_path)

        # 更新学习率
        scheduler.step()

        # 打印日志
        print(f"[{model_name}] Epoch {epoch+1:2d}/{epochs}: "
              f"Train Loss={avg_train_loss:.4f}, "
              f"Val Loss={avg_val_loss:.4f}, "
              f"Val Dice={avg_val_dice:.4f}, "
              f"Val IoU={avg_val_iou:.4f}, "
              f"Best Dice={best_val_dice:.4f}")

    # ------------------------------
    # 最终评估（加载最优模型）
    # ------------------------------
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    final_val_dice = 0.0
    final_val_iou = 0.0
    with torch.no_grad():
        for imgs, masks, _ in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            if has_boundary:
                seg_pred, _ = model(imgs)
            else:
                seg_pred = model(imgs)
            dice, iou = calculate_metrics(seg_pred, masks)
            final_val_dice += dice
            final_val_iou += iou
    final_val_dice /= len(val_loader)
    final_val_iou /= len(val_loader)

    # 统计效率指标（参数、FLOPs）
    params, flops = count_params_flops(model)

    # ------------------------------
    # 保存结果
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
        "Val Dices": val_dices
    }

    print(f"\n[{model_name}] 最终结果: "
          f"Dice={final_val_dice:.2%}, "
          f"IoU={final_val_iou:.2%}, "
          f"Params={params:.2f}M, "
          f"FLOPs={flops:.2f}G\n")

    return result

# ========================================
# 8. 执行消融实验：训练所有变体
# ========================================
# 定义所有消融变体（顺序：基准→逐步移除/替换）
ablation_variants = [
    ("Baseline (Full LBA-Net)", LBANet_Baseline, True),  # 基准
    ("V1 (w/o LBA-Block)", LBANet_NoLBA, True),          # 无LBA
    ("V2 (w/o ASPP)", LBANet_NoASPP, True),              # 无ASPP
    ("V3 (w/o Boundary Head)", LBANet_NoBoundary, False),# 无边界头
    ("V4 (LBA→CBAM)", LBANet_CBAM, True)                 # LBA换CBAM
]

# 批量训练并收集结果（建议分批次运行，避免Colab会话超时）
ablation_results = []
for model_name, model_class, has_boundary in ablation_variants:
    print("="*50)
    print(f"开始训练：{model_name}")
    print("="*50)
    result = train_evaluate_model(model_name, model_class, has_boundary, epochs=50)
    ablation_results.append(result)

# ========================================
# 9. 消融实验结果可视化：表格 + 曲线
# ========================================
# ------------------------------
# 9.1 生成结果表格（保存为CSV+美化显示）
# ------------------------------
# 提取关键指标（排除训练曲线）
table_data = []
for res in ablation_results:
    table_data.append({
        "Model Variant": res["Model Name"],
        "Val Dice (%)": res["Final Val Dice (%)"],
        "Val IoU (%)": res["Final Val IoU (%)"],
        "Params (M)": res["Params (M)"],
        "FLOPs (G)": res["FLOPs (G)"],
        "Best Val Dice (%)": res["Best Val Dice (%)"]
    })

# 转换为DataFrame并保存
df_ablation = pd.DataFrame(table_data)
df_ablation = df_ablation.sort_values("Val Dice (%)", ascending=False)  # 按Dice降序
csv_path = os.path.join(SAVE_DIR, "ablation_results.csv")
df_ablation.to_csv(csv_path, index=False, encoding="utf-8")

# 美化表格显示（加粗最优值）
def highlight_max(s):
    is_max = s == s.max()
    return [f"font-weight: bold; background-color: #f0f8ff" if v else "" for v in is_max]

styled_table = df_ablation.style.apply(highlight_max, subset=["Val Dice (%)", "Val IoU (%)"])
styled_table = styled_table.set_caption("Ablation Study Results of LBA-Net")
print("\n消融实验结果表格：")
display(styled_table)
files.download(csv_path)  # 下载到本地

# ------------------------------
# 9.2 绘制训练曲线对比（Loss + Dice）
# ------------------------------
plt.rcParams['font.size'] = 10
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# 子图1：训练/验证Loss对比
for res in ablation_results:
    ax1.plot(res["Train Losses"], label=f"{res['Model Name']} (Train)", linestyle="-")
    ax1.plot(res["Val Losses"], label=f"{res['Model Name']} (Val)", linestyle="--")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Train/Val Loss Comparison")
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(alpha=0.3)

# 子图2：验证Dice对比
for res in ablation_results:
    ax2.plot(res["Val Dices"], label=res["Model Name"], marker="o", markersize=2)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Val Dice")
ax2.set_title("Val Dice Comparison")
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.grid(alpha=0.3)

plt.tight_layout()
curve_path = os.path.join(SAVE_DIR, "ablation_curves.png")
plt.savefig(curve_path, dpi=300, bbox_inches='tight')
plt.show()
files.download(curve_path)  # 下载到本地

# ------------------------------
# 9.3 绘制分割结果示例（对比基准与关键变体）
# ------------------------------
def plot_segmentation_examples(model_classes, model_names, num_examples=3):
    """绘制多个变体的分割结果对比"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imgs, masks, _ = next(iter(val_loader))
    imgs, masks = imgs.to(device), masks.to(device)

    # 加载每个变体的最优模型
    models = []
    for cls, name in zip(model_classes, model_names):
        model = cls().to(device)
        model.load_state_dict(torch.load(os.path.join(SAVE_DIR, f"{name}_best.pth")))
        model.eval()
        models.append(model)

    # 生成预测
    preds = []
    with torch.no_grad():
        for model, has_boundary in zip(models, [True, True, False]):  # 适配变体3
            if has_boundary:
                pred, _ = model(imgs)
            else:
                pred = model(imgs)
            preds.append((pred > 0.5).float())

    # 绘制对比图
    for i in range(num_examples):
        plt.figure(figsize=(18, 4))
        # 输入图像
        plt.subplot(1, len(models)+2, 1)
        plt.imshow(imgs[i].permute(1,2,0).cpu().numpy())
        plt.title("Input Image")
        plt.axis("off")
        # 真值
        plt.subplot(1, len(models)+2, 2)
        plt.imshow(masks[i][0].cpu().numpy(), cmap="gray")
        plt.title("GT Mask")
        plt.axis("off")
        # 各变体预测
        for j, (pred, name) in enumerate(zip(preds, model_names)):
            plt.subplot(1, len(models)+2, j+3)
            plt.imshow(pred[i][0].cpu().numpy(), cmap="gray")
            plt.title(f"{name} (Pred)")
            plt.axis("off")
        plt.tight_layout()
        example_path = os.path.join(SAVE_DIR, f"segmentation_example_{i+1}.png")
        plt.savefig(example_path, dpi=300, bbox_inches='tight')
        plt.show()
        files.download(example_path)

# 选择关键变体绘制（基准、无LBA、无边界头）
plot_segmentation_examples(
    model_classes=[LBANet_Baseline, LBANet_NoLBA, LBANet_NoBoundary],
    model_names=["Baseline (Full LBA-Net)", "V1 (w/o LBA-Block)", "V3 (w/o Boundary Head)"],
    num_examples=3
)