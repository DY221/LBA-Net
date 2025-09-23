#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LBA-Net: Lightweight Boundary-Aware Network for Robust Breast Ultrasound Image Segmentation
One-file Colab runnable script
"""

# ===LBA-Net 完整实现（MobileNetV3-Small 编码器 + LBA-Block + 双头监督）
# ========================= 0. 自动安装依赖 =========================
import subprocess, sys, os

def pip_install(cmd):
    """cmd 为完整 pip 命令字符串"""
    subprocess.check_call(cmd, shell=True)

need = [
    "torch==2.0.0+cu118",
    "torchvision==0.15.1+cu118",
    "timm",
    "albumentations",
    "segmentation-models-pytorch"
]

for pkg in need:
    try:
        __import__(pkg.split("+")[0].split("==")[0])
    except ImportError:
        if "cu118" in pkg:
            pip_install(f"{sys.executable} -m pip -q install {pkg} --index-url https://download.pytorch.org/whl/cu118")
        else:
            pip_install(f"{sys.executable} -m pip -q install {pkg}")

# ========================= 1. 挂载 Drive =========================
from google.colab import drive
drive.mount('/content/drive')
DATA_DIR = "/content/drive/MyDrive/Dataset_BUSI_with_GT"



# ========================= 2. 导入库 =========================
import torch, cv2, random, time, math, numpy as np, matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import timm
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ========================= 3. 数据集 =========================
class BUSIDataset(Dataset):
    def __init__(self, root, split='train', img_size=512):
        self.root, self.img_size, self.split = root, img_size, split
        self.imgs, self.masks = [], []

        from sklearn.model_selection import train_test_split

        # 在 BUSIDataset.__init__ 里替换掉原来的「简单 8:2」代码
        cls_list = ['benign', 'malignant', 'normal']
        all_imgs, all_masks, all_labels = [], [], []
        for cls in cls_list:
          cls_dir = os.path.join(root, cls)
          for fname in sorted(os.listdir(cls_dir)):
           if 'mask' in fname: continue
           mask_name = fname.replace('.png', '_mask.png') 
           mask_path = os.path.join(cls_dir, mask_name)
           if os.path.exists(mask_path):
              all_imgs.append(os.path.join(cls_dir, fname))
              all_masks.append(mask_path)
              all_labels.append(cls)
        # 分层 + 按病例划分（这里用文件名前缀当病例 ID，可换成真实病例号）
          cases = [os.path.basename(p).split('_')[0] for p in all_imgs]
          unique_cases = list(set(cases))
          train_cases, val_cases = train_test_split(
            unique_cases, test_size=0.2, random_state=42,
            stratify=[all_labels[cases.index(c)] for c in unique_cases])

          flag = [c in train_cases for c in cases]
          self.imgs  = [all_imgs[i]  for i, f in enumerate(flag) if f == (split == 'train')]
          self.masks = [all_masks[i] for i, f in enumerate(flag) if f == (split == 'train')]
        
            
            
            

  
        # 用于分层




       # for cls in ['benign', 'malignant', 'normal']:
       #     cls_dir = os.path.join(root, cls)
       #    for fname in sorted(os.listdir(cls_dir)):
       #         if 'mask' in fname: continue
       #         mask_name = fname.replace('.png', '_mask.png')
       #         mask_path = os.path.join(cls_dir, mask_name)
       #         if os.path.exists(mask_path):
       #             self.imgs.append(os.path.join(cls_dir, fname))
       #             self.masks.append(mask_path)
       # split_idx = int(0.8 * len(self.imgs))
       # if split == 'train':
       #     self.imgs, self.masks = self.imgs[:split_idx], self.masks[:split_idx]
       # else:
       #     self.imgs, self.masks = self.imgs[split_idx:], self.masks[split_idx:]

        self.aug = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
            
        ]) if split == 'train' else A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def __len__(self): return len(self.imgs)


    def __getitem__(self, idx):
        img_path  = self.imgs[idx]
        mask_path = self.masks[idx]          # 这里重新声明
        assert os.path.exists(mask_path), f"mask not found: {mask_path}"

        img  = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)
        mask = (mask > 127).astype(np.uint8)   # 确保 0/1 二值
        aug  = self.aug(image=img, mask=mask)
        return aug['image'], aug['mask'].float().unsqueeze(0)

# ========================= 4. LBA-Block =========================
class ECA(nn.Module):
    def __init__(self, c, k=3):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.avg(x)
        y = self.conv(y.squeeze(-1).transpose(-1,-2)).transpose(-1,-2).unsqueeze(-1)
        return x * self.sigmoid(y)

class SpatialAtt(nn.Module):
    def __init__(self):
        super().__init__()
        self.dw = nn.Sequential(nn.Conv2d(1,1,3,padding=1,groups=1,bias=False), nn.Sigmoid())
    def forward(self, x):
        att = torch.mean(x, dim=1, keepdim=True)
        att = self.dw(att)
        return x * att

class LBA_Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.eca = ECA(c)
        self.spa = SpatialAtt()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta  = nn.Parameter(torch.tensor(0.5))
    def forward(self, x):
        return self.alpha * self.eca(x) + self.beta * self.spa(x)

# ========================= 5. 解码器 & 双头 =========================
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
        self.lba = LBA_Block(out_ch)
    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.lba(self.conv(x))

class LBA_Net(nn.Module):
    def __init__(self, n_cls=1):
        super().__init__()
        self.encoder = timm.create_model('mobilenetv3_small_100', pretrained=True, features_only=True)
        enc_ch = self.encoder.feature_info.channels()  # [16,24,48,96,576]
        self.aspp = nn.Sequential(
            nn.Conv2d(enc_ch[-1], 96, 3, dilation=6, padding=6, bias=False), nn.BatchNorm2d(96), nn.ReLU(),
            nn.Conv2d(96, 96, 3, dilation=12, padding=12, bias=False), nn.BatchNorm2d(96), nn.ReLU(),
            nn.Conv2d(96, 96, 3, dilation=18, padding=18, bias=False), nn.BatchNorm2d(96), nn.ReLU(),
            nn.Conv2d(96, 96, 1, bias=False), nn.BatchNorm2d(96), nn.ReLU()
        )
        self.dec4 = DecoderBlock(96, enc_ch[3], 96)
        self.dec3 = DecoderBlock(96, enc_ch[2], 64)
        self.dec2 = DecoderBlock(64, enc_ch[1], 32)
        self.dec1 = DecoderBlock(32, enc_ch[0], 16)
        self.seg_head = nn.Conv2d(16, n_cls, 1)
        self.bdy_head = nn.Conv2d(16, 1, 1)
    def forward(self, x):
        feats = self.encoder(x)          # 1/2, 1/4, 1/8, 1/16, 1/32
        x = self.aspp(feats[-1])         # 1/32
        x = self.dec4(x, feats[3])       # 1/16 -> 1/8
        x = self.dec3(x, feats[2])       # 1/8  -> 1/4
        x = self.dec2(x, feats[1])       # 1/4  -> 1/2
        x = self.dec1(x, feats[0])       # 1/2  -> 1/1 (512×512)
        # 再显式保证尺寸完全一致
        x = F.interpolate(x, size=(x.shape[2]*2, x.shape[3]*2), mode='bilinear', align_corners=False)
        seg = torch.sigmoid(self.seg_head(x))
        bdy = torch.sigmoid(self.bdy_head(x))
        return seg, bdy

# ========================= 6. 损失函数 =========================
def dice_loss(pred, target, smooth=1.):
    inter = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    dice  = (2.*inter + smooth) / (union + smooth)
    return 1 - dice.mean()          # 再 mean 一次，保证标量

def tversky_loss(pred, target, alpha=0.7, beta=0.3):
    tp = (pred * target).sum(dim=(2,3))
    fp = (pred * (1-target)).sum(dim=(2,3))
    fn = ((1-pred) * target).sum(dim=(2,3))
    tversky = (tp + 1) / (tp + alpha*fp + beta*fn + 1)
    return 1 - tversky.mean()       # 再 mean 一次，保证标量

def criterion(seg, bdy, y_seg, y_bdy, lambda_bdy=0.3):
    loss_seg = dice_loss(seg, y_seg) + nn.BCELoss()(seg, y_seg)
    loss_bdy = tversky_loss(bdy, y_bdy)
    return loss_seg + lambda_bdy * loss_bdy

# ========================= 7. 训练/验证 =========================
# ========================= 7. 带 Dice 指标的训练/验证 =========================
def dice_coef(pred, target, smooth=1.):
    pred = (pred > 0.5).float()
    inter = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    return ((2.*inter + smooth)/(union + smooth)).mean()

def run_epoch(model, loader, optimizer=None, device='cuda'):
    model.train() if optimizer else model.eval()
    total_loss, total_dice, n = 0., 0., 0
    for img, mask in tqdm(loader, leave=False):
        img, mask = img.to(device), mask.to(device)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        y_bdy = cv2.morphologyEx((mask[0,0].cpu().numpy()*255).astype(np.uint8),
                                 cv2.MORPH_GRADIENT, kernel)
        y_bdy = (torch.from_numpy(y_bdy).float()/255.).unsqueeze(0).unsqueeze(0).to(device)

        seg, bdy = model(img)
        loss = criterion(seg, bdy, mask, y_bdy)
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()*img.size(0)
        total_dice += dice_coef(seg, mask).item()*img.size(0)
        n += img.size(0)
    return total_loss/n, total_dice/n

# ========================= 8. 主训练 =========================
model = LBA_Net().to(device)
train_ds = BUSIDataset(DATA_DIR, 'train', 512)
val_ds   = BUSIDataset(DATA_DIR, 'val', 512)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False, pin_memory=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
best, counter = 1e9, 0
for epoch in range(200):
    train_loss, train_dice = run_epoch(model, train_loader, optimizer, device)
    val_loss, val_dice     = run_epoch(model, val_loader, None, device)
    scheduler.step()
    print(f'Epoch {epoch:03d} | train loss {train_loss:.4f} Dice {train_dice:.4f} '
          f'| val loss {val_loss:.4f} Dice {val_dice:.4f}')
    if val_loss < best:
        best, counter = val_loss, 0
        torch.save(model.state_dict(), '/content/drive/MyDrive/LBA_Net_best.pth')
    else:
        counter += 1
        if counter >= 20: break

# ========================= 9. FPS 测试 =========================
model.load_state_dict(torch.load('/content/drive/MyDrive/LBA_Net_best.pth', map_location=device))
model.eval()
dummy = torch.randn(1,3,512,512).to(device)
for _ in range(10): _ = model(dummy)  # warmup
torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    with torch.no_grad():
        _ = model(dummy)
torch.cuda.synchronize()
print('GPU FPS:', 100/(time.time()-t0))

model.cpu(); dummy = dummy.cpu()
t0 = time.time()
for _ in range(20):
    with torch.no_grad():
        _ = model(dummy)
print('CPU FPS:', 20/(time.time()-t0))

# ========================= 10. 可视化 =========================
img, mask = val_ds[5]
model = model.to(device)
with torch.no_grad():
    seg, bdy = model(img.unsqueeze(0).to(device))
plt.figure(figsize=(12,4))
plt.subplot(131); plt.imshow(img.permute(1,2,0)); plt.title('Input')
plt.subplot(132); plt.imshow(mask[0], cmap='gray'); plt.title('GT Mask')
plt.subplot(133); plt.imshow(seg[0,0].cpu()>0.5, cmap='gray'); plt.title('LBA-Net')
plt.tight_layout()
plt.savefig('/content/drive/MyDrive/LBA_result.png', dpi=150)