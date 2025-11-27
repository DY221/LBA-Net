# ================= 0. 环境 & 数据路径 =================
# pip -q install segmentation-models-pytorch timm albumentations opencv-python thop matplotlib monai

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
from scipy.ndimage import distance_transform_edt as edt
from scipy.spatial.distance import directed_hausdorff
from monai.metrics import compute_hausdorff_distance

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# DATA_DIR = "/home/wang/ultrasound/Dataset_BUSI_with_GT"#BUET_BUSD-new
DATA_DIR = "/home/wang/ultrasound/BUET_BUSD-new"
OUT_DIR = "/home/wang/ultrasound/segnet-BUET"
# OUT_DIR  = "/home/wang/ultrasound/LBA-CBAMTestBUSI1"
os.makedirs(OUT_DIR, exist_ok=True)

IMG_SIZE = 512
BATCH_SIZE = 24
EPOCHS = 300
NUM_WORKERS = 8
USE_AMP = True

# ================= 1. Dataset =================
class BUETDataset(Dataset):
    def __init__(self, root, split='train', img_size=512):
        self.root = root
        self.split = split
        self.img_size = img_size
        cls_list = ['benign', 'malignant']
        self.imgs, self.masks = [], []
        for cls in cls_list:
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir):
                continue

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
                base_name = os.path.splitext(img_fname)[0]  # → 例如 benign (4)
                # 扩展名（自动适配 .png/.bmp/.jpg/.jpeg/.tif）
                ext = os.path.splitext(img_fname)[1].lower()  # 例如 ".bmp"
                # 构造正则：匹配以下任意一种形式
                # benign (4)_mask.png
                # benign (4)_mask_1.png
                # benign (4)_mask_123.png
                pattern = re.compile(rf"^{re.escape(base_name)}_mask(_\d+)?{re.escape(ext)}$", re.IGNORECASE)
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

        ids = list(range(len(self.imgs)))
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
        # 直接使用已有的mask数组
        mask = self.masks[idx]

        if mask.sum() == 0:
            print(f"Warning: All-zero mask at {self.imgs[idx]}")

        aug = self.aug(image=img, mask=mask)
        img, mask = aug['image'], aug['mask']

        if isinstance(mask, torch.Tensor):
            mask = mask.unsqueeze(0).float()
        else:
            mask = torch.from_numpy(mask).unsqueeze(0).float()
        # 二值化
        mask = (mask > 0.5).float()
        bdy = cv2.morphologyEx(mask.squeeze().cpu().numpy().astype(np.uint8),
                               cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        bdy = torch.from_numpy(cv2.resize(bdy, (self.img_size, self.img_size))).unsqueeze(0).float()

        if torch.isnan(img).any() or torch.isinf(img).any():
            raise ValueError(f"NaN/Inf in image: {self.imgs[idx]}")
        if torch.isnan(mask).any() or torch.isnan(bdy).any():
            raise ValueError(f"NaN in mask/bdy: {self.imgs[idx]}")
        return img, mask, bdy

    def __len__(self):
        return len(self.imgs)


# ================= 2. Metrics =================
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


def hd95_score(pred, target, img_size=512):
    """
    计算并返回批次中所有样本的平均HD95分数。
    对极端情况（如预测或标签为空）具有鲁棒性。
    """
    if pred.ndim == 3:
        pred = pred.unsqueeze(1)
    if target.ndim == 3:
        target = target.unsqueeze(1)

    pred_bin = (pred > 0.5).float()
    target_bin = (target > 0.5).float()

    # 构造 one-hot: [B, 2, H, W]
    pred_oh = torch.cat([1 - pred_bin, pred_bin], dim=1)
    target_oh = torch.cat([1 - target_bin, target_bin], dim=1)

    # 理论最大距离（图像对角线）
    max_dist = np.sqrt(img_size ** 2 + img_size ** 2)

    try:
        hd95_vals = compute_hausdorff_distance(
            y_pred=pred_oh,
            y=target_oh,
            include_background=False,
            percentile=95,
            directed=False,
            spacing=None
        )  # shape: [B, 1]

        hd95_per_sample = hd95_vals[:, 0]
        hd95_per_sample = torch.nan_to_num(hd95_per_sample, nan=max_dist, posinf=max_dist, neginf=0.0)

        return hd95_per_sample.mean().item()

    except Exception as e:
        print(f"MONAI HD95 calculation failed: {e}. Falling back to max distance.")
        return torch.tensor(max_dist, device=pred.device, dtype=torch.float32).item()


def compute_hd95(p, t):
    # numpy 版本（未在主流程中使用）
    p = (p > 0.5).astype(np.uint8)
    t = (t > 0.5).astype(np.uint8)

    if p.sum() == 0 and t.sum() == 0:
        return 0.0
    if p.sum() == 0 or t.sum() == 0:
        return np.sqrt(p.shape[0] ** 2 + p.shape[1] ** 2)

    p_surface = p - cv2.erode(p, np.ones((3, 3), np.uint8), iterations=1)
    t_surface = t - cv2.erode(t, np.ones((3, 3), np.uint8), iterations=1)

    p_coords = np.argwhere(p_surface)
    t_coords = np.argwhere(t_surface)

    if len(p_coords) == 0 or len(t_coords) == 0:
        return np.sqrt(p.shape[0] ** 2 + p.shape[1] ** 2)

    from scipy.spatial.distance import cdist
    dists = cdist(p_coords, t_coords, metric='euclidean')
    hd1 = np.percentile(dists.min(axis=1), 95)
    hd2 = np.percentile(dists.min(axis=0), 95)
    return max(hd1, hd2)


# ================= 3. 标准 SegNet 实现 =================
class SegNet(nn.Module):
    """
    标准 SegNet：VGG 风格 encoder + 对称 decoder，
    使用 MaxPool2d(return_indices=True) + MaxUnpool2d 进行上采样。
    这里做了简化：每层 2 个 conv，最后一层用 1xConv 输出 num_classes 通道（logits）。
    """
    def __init__(self, in_channels=3, num_classes=1):
        super(SegNet, self).__init__()

        # Encoder
        self.enc1 = self.encoder_block(in_channels, 64)
        self.enc2 = self.encoder_block(64, 128)
        self.enc3 = self.encoder_block(128, 256)
        self.enc4 = self.encoder_block(256, 512)
        self.enc5 = self.encoder_block(512, 512)

        # Decoder
        self.dec5 = self.decoder_block(512, 512)
        self.dec4 = self.decoder_block(512, 256)
        self.dec3 = self.decoder_block(256, 128)
        self.dec2 = self.decoder_block(128, 64)
        # 最后一层只做一次卷积输出 num_classes（不再接 ReLU）
        self.dec1 = nn.Conv2d(64, num_classes, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

    def encoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def decoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # ========= Encoder =========
        x1 = self.enc1(x)            # [B, 64, H, W]
        x1p, idx1 = self.pool(x1)    # [B, 64, H/2, W/2]

        x2 = self.enc2(x1p)          # [B, 128, H/2, W/2]
        x2p, idx2 = self.pool(x2)    # [B, 128, H/4, W/4]

        x3 = self.enc3(x2p)          # [B, 256, H/4, W/4]
        x3p, idx3 = self.pool(x3)    # [B, 256, H/8, W/8]

        x4 = self.enc4(x3p)          # [B, 512, H/8, W/8]
        x4p, idx4 = self.pool(x4)    # [B, 512, H/16, W/16]

        x5 = self.enc5(x4p)          # [B, 512, H/16, W/16]
        x5p, idx5 = self.pool(x5)    # [B, 512, H/32, W/32]

        # ========= Decoder =========
        d5 = self.unpool(x5p, idx5, output_size=x5.size())   # [B, 512, H/16, W/16]
        d5 = self.dec5(d5)

        d4 = self.unpool(d5, idx4, output_size=x4.size())    # [B, 512, H/8, W/8]
        d4 = self.dec4(d4)

        d3 = self.unpool(d4, idx3, output_size=x3.size())    # [B, 256, H/4, W/4]
        d3 = self.dec3(d3)

        d2 = self.unpool(d3, idx2, output_size=x2.size())    # [B, 128, H/2, W/2]
        d2 = self.dec2(d2)

        d1 = self.unpool(d2, idx1, output_size=x1.size())    # [B, 64, H, W]
        out = self.dec1(d1)                                  # [B, num_classes, H, W] (logits)

        return out


# ================= 4. 损失函数 =================
bce = nn.BCEWithLogitsLoss()


def dice_loss(logits, target):
    pred = torch.sigmoid(logits)
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1 - ((2 * inter + 1e-6) / (union + 1e-6)).mean()


def segmentation_loss(seg_logits, mask):
    """BCE + Dice"""
    bce_loss = nn.BCEWithLogitsLoss()(seg_logits, mask)
    dice_loss_val = dice_loss(seg_logits, mask)
    return bce_loss + dice_loss_val


# ================= 5. 可视化工具 =================
def visualize_with_guidance(model, val_loader, epoch, save_dir, num_samples=3):
    model.eval()
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))

    with torch.no_grad():
        for i, (imgs, masks, _) in enumerate(val_loader):
            if i >= num_samples:
                break

            imgs, masks = imgs.to(device), masks.to(device)
            seg_logits = model(imgs)
            seg_pred = torch.sigmoid(seg_logits)

            img_np = imgs[0].cpu().permute(1, 2, 0).numpy()
            img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_np = np.clip(img_np, 0, 1)

            true_mask = masks[0, 0].cpu().numpy()
            pred_mask = seg_pred[0, 0].cpu().numpy()

            axes[i, 0].imshow(img_np)
            axes[i, 0].set_title('Original')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(true_mask, cmap='gray')
            axes[i, 1].set_title('GT Mask')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(pred_mask, cmap='gray')
            axes[i, 2].set_title(f'Pred (Dice: {dice_score(seg_pred[0:1], masks[0:1]):.3f})')
            axes[i, 2].axis('off')

            axes[i, 3].imshow(img_np)
            axes[i, 3].imshow(pred_mask > 0.5, alpha=0.5, cmap='jet')
            axes[i, 3].set_title('Overlay')
            axes[i, 3].axis('off')

    plt.tight_layout()
    if isinstance(epoch, int):
        plt.savefig(f'{save_dir}/segnet_visualization_epoch_{epoch:03d}.png', dpi=100, bbox_inches='tight')
    else:
        plt.savefig(f'{save_dir}/segnet_visualization_{epoch}.png', dpi=100, bbox_inches='tight')
    plt.close()


# ================= 6. 优化器 =================
def create_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)


# ================= 7. 模型评估（统一使用 SegNet） =================
def evaluate_fold_model(model_path, val_indices, full_dataset):
    """评估单个fold的模型，返回该 fold 的指标均值（scalar）"""
    model = SegNet(in_channels=3, num_classes=1).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    val_subset = torch.utils.data.Subset(full_dataset, val_indices)
    val_loader = DataLoader(val_subset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    model.eval()
    dice_scores, iou_scores, recall_scores, hd95_scores = [], [], [], []

    with torch.no_grad():
        for imgs, masks, _ in tqdm(val_loader, desc='Evaluating', leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            seg = model(imgs)
            seg_pred = torch.sigmoid(seg)

            dice_scores.append(dice_score(seg_pred, masks))
            iou_scores.append(iou_score(seg_pred, masks))
            recall_scores.append(recall_score(seg_pred, masks))
            hd95_scores.append(hd95_score(seg_pred, masks))

    return {
        'dice': np.mean(dice_scores),
        'iou': np.mean(iou_scores),
        'recall': np.mean(recall_scores),
        'hd95': np.mean(hd95_scores)
    }


# ================= 8. 5-fold 训练（SegNet） =================
def train_kfold_boundary_guided(k_folds=5):
    """执行5折交叉验证训练（模型为标准SegNet）"""
    full_ds = BUETDataset(DATA_DIR, 'all', IMG_SIZE)
    print(f"Total dataset size: {len(full_ds)}")

    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=SEED)
    fold_results = []
    fold_val_indices = []  # 保存每折验证集索引

    for fold, (train_ids, val_ids) in enumerate(kfold.split(full_ds.imgs)):
        print(f"\n{'=' * 50}")
        print(f"          FOLD {fold + 1}/{k_folds}")
        print(f"{'=' * 50}")

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
        model = SegNet(in_channels=3, num_classes=1).to(device)
        optimizer = create_optimizer(model)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=1e-3,
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
            pbar = tqdm(train_loader, desc=f'Fold {fold + 1} | Epoch {epoch + 1}/{EPOCHS}')

            for imgs, masks, bdys in pbar:
                imgs, masks, bdys = imgs.to(device), masks.to(device), bdys.to(device)
                optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    seg_logits = model(imgs)
                    loss = segmentation_loss(seg_logits, masks)

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
                        seg = model(imgs)
                    seg_pred = torch.sigmoid(seg)
                    val_dice_accum += dice_score(seg_pred, masks) * imgs.size(0)
                    n_samples += imgs.size(0)

            current_dice = val_dice_accum / n_samples
            val_dices.append(current_dice)

            print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {epoch_loss:.4f} | Val Dice: {current_dice:.4f}")

            # ===== 保存最佳模型 =====
            if current_dice > best_dice:
                best_dice = current_dice
                model_path = os.path.join(OUT_DIR, f'best_model_fold_{fold + 1}.pth')
                torch.save({
                    'fold': fold,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_dice': best_dice,
                    'val_indices': val_ids,
                }, model_path)

        fold_results.append(best_dice)
        print(f"✅ Fold {fold + 1} Best Val Dice: {best_dice:.4f}")

    mean_dice = np.mean(fold_results)
    std_dice = np.std(fold_results)
    print(f"\n{'=' * 60}")
    print(f"📊 5-Fold Cross Validation Results:")
    print(f"   Mean Dice: {mean_dice:.4f} ± {std_dice:.4f}")
    print(f"   Per-fold: {[f'{d:.4f}' for d in fold_results]}")

    return mean_dice, fold_results, fold_val_indices


# ================= 主执行函数 =================
if __name__ == "__main__":
    print("=== Standard SegNet with 5-Fold Cross Validation ===")
    print(f"输出目录: {OUT_DIR}")
    print(f"设备: {device}")
    print(f"图像尺寸: {IMG_SIZE}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"训练轮数: {EPOCHS}")

    # 1. 执行5折交叉验证训练
    mean_dice, fold_results, val_indices = train_kfold_boundary_guided(k_folds=5)

    # 2. 评估所有fold的最佳模型（每折一个 scalar）
    print("\n" + "=" * 60)
    print("📊 正在对每个 fold 的最佳模型进行 Inter-fold 评估...")
    print("=" * 60)

    fold_metrics_list = []  # list of dict, length = 5
    full_ds = BUETDataset(DATA_DIR, 'all', IMG_SIZE)

    for fold in range(5):
        print(f"\n--- 评估 Fold {fold + 1}/{5} ---")
        checkpoint_path = f'{OUT_DIR}/best_model_fold_{fold + 1}.pth'

        if not os.path.exists(checkpoint_path):
            print(f"⚠️ 警告: {checkpoint_path} 不存在，跳过此 fold")
            fold_metrics_list.append({'dice': np.nan, 'iou': np.nan, 'recall': np.nan, 'hd95': np.nan})
            continue

        fold_metrics = evaluate_fold_model(checkpoint_path, val_indices[fold], full_ds)
        fold_metrics_list.append(fold_metrics)

        print(f"Fold {fold + 1} 验证集指标:")
        print(f"  Dice:   {fold_metrics['dice'] * 100:.2f}%")
        print(f"  IoU:    {fold_metrics['iou'] * 100:.2f}%")
        print(f"  Recall: {fold_metrics['recall'] * 100:.2f}%")
        print(f"  HD95:   {fold_metrics['hd95']:.2f} px")

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

    print("\n" + "=" * 60)
    print("✅ 纯 Inter-fold 报告（每折一个 scalar）")
    print("=" * 60)
    print(f"Dice:   {mean_std_str(dice_per_fold, 100)} %")
    print(f"IoU:    {mean_std_str(iou_per_fold, 100)} %")
    print(f"Recall: {mean_std_str(recall_per_fold, 100)} %")
    print(f"HD95:   {mean_std_str(hd95_per_fold, 1, '.2f')} px")

    # 4. 保存结果
    summary_path = os.path.join(OUT_DIR, 'kfold_inter_fold_results.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Pure Inter-fold Cross Validation Results (SegNet)\n")
        f.write("(One scalar per fold → mean ± std)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dice:   {mean_std_str(dice_per_fold, 100)} %\n")
        f.write(f"IoU:    {mean_std_str(iou_per_fold, 100)} %\n")
        f.write(f"Recall: {mean_std_str(recall_per_fold, 100)} %\n")
        f.write(f"HD95:   {mean_std_str(hd95_per_fold, 1, '.2f')} px\n\n")

        f.write("Per-fold scalars:\n")
        for i, m in enumerate(fold_metrics_list):
            f.write(f"Fold {i + 1}: Dice={m['dice'] * 100:.2f}%, IoU={m['iou'] * 100:.2f}%, "
                    f"Recall={m['recall'] * 100:.2f}%, HD95={m['hd95']:.2f}px\n")

    print(f"\n✅ 纯 Inter-fold 结果已保存至: {summary_path}")

