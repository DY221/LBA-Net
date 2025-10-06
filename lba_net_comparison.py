# ========= 0. 安装依赖 ==========
!pip -q install segmentation-models-pytorch albumentations timm opencv-python matplotlib scikit-learn

# ========= 1. 导入库 & 设置 ==========
import os, random, cv2, numpy as np, matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
import timm
from tqdm import tqdm

# reproducibility
SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", device)

# ========= 2. Drive 挂载 & 参数 ==========
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

DATA_DIR = "/content/drive/MyDrive/Dataset_BUSI_with_GT"
OUT_DIR  = "/content/drive/MyDrive/BUSI_lightweight_comparison"
os.makedirs(OUT_DIR, exist_ok=True)

IMG_SIZE = 512
BATCH_SIZE = 8
EPOCHS = 120
NUM_WORKERS = 0

# ========= 3. BUSIDataset (含 mask 合并) ==========
class BUSIDataset(Dataset):
    def __init__(self, root, split='train', img_size=512):
        self.root, self.split, self.img_size = root, split, img_size
        cls_list = ['benign','malignant','normal']
        self.imgs, self.masks = [], []

        for cls in cls_list:
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir): continue
            for fname in sorted(os.listdir(cls_dir)):
                if 'mask' in fname: continue
                img_path = os.path.join(cls_dir,fname)
                base_name = fname.replace('.png','')
                mask_candidates = [os.path.join(cls_dir,m) for m in os.listdir(cls_dir) if m.startswith(base_name) and 'mask' in m]
                if not mask_candidates: continue
                merged_mask = None
                for mp in mask_candidates:
                    mask = cv2.imread(mp,0)
                    mask_bin = (mask>127).astype(np.uint8)
                    merged_mask = mask_bin if merged_mask is None else np.maximum(merged_mask, mask_bin)
                if merged_mask is not None:
                    self.imgs.append(img_path)
                    self.masks.append(merged_mask)

        # train/val split 80/20
        indices = list(range(len(self.imgs)))
        random.shuffle(indices)
        split_idx = int(0.8*len(indices))
        if split=='train': indices=indices[:split_idx]
        else: indices=indices[split_idx:]
        self.imgs = [self.imgs[i] for i in indices]
        self.masks = [self.masks[i] for i in indices]

        # Augmentations
        if split=='train':
            self.aug = A.Compose([
                #A.RandomResizedCrop(img_size,img_size,scale=(0.8,1.0)),
                #A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.8,1.0)),
                A.Resize(img_size,img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Rotate(limit=20,p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225)),
                ToTensorV2()
            ])
        else:
            self.aug = A.Compose([
                A.Resize(img_size,img_size),
                A.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225)),
                ToTensorV2()
            ])
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        mask = self.masks[idx]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        aug = self.aug(image=img, mask=mask)
        img, mask = aug['image'], aug['mask']

    # mask 转 Tensor 并加 channel 维度
        if isinstance(mask, np.ndarray):
           mask = torch.from_numpy(mask).unsqueeze(0).float()
        elif isinstance(mask, torch.Tensor):
           mask = mask.unsqueeze(0).float()

        return img, mask

        """
    def __getitem__(self,idx):
        img = cv2.cvtColor(cv2.imread(self.imgs[idx]), cv2.COLOR_BGR2RGB)
        mask = self.masks[idx]
        aug = self.aug(image=img,mask=mask)
        img, mask = aug['image'], aug['mask']
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        return img, mask"""



# ========= 4. Metrics & Loss ==========
def dice_score(pred,target,eps=1e-6):
    pred=(pred>0.5).float()
    inter=(pred*target).sum(dim=(2,3))
    union=pred.sum(dim=(2,3))+target.sum(dim=(2,3))
    return ((2*inter+eps)/(union+eps)).mean().item()

def iou_score(pred,target,eps=1e-6):
    pred=(pred>0.5).float()
    inter=(pred*target).sum(dim=(2,3))
    union=pred.sum(dim=(2,3))+target.sum(dim=(2,3))-inter
    return ((inter+eps)/(union+eps)).mean().item()

bce_logit = nn.BCEWithLogitsLoss()
def loss_fn(logits,target):
    dice_loss = 1 - ((2*(torch.sigmoid(logits)*target).sum((2,3))+1e-6)/(torch.sigmoid(logits).sum((2,3))+target.sum((2,3))+1e-6)).mean()
    return bce_logit(logits,target) + dice_loss

# ========= 5. Lightweight LBA-Net ==========
class ECA(nn.Module):
    def __init__(self,c,k=3):
        super().__init__()
        self.avg=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1,1,k,padding=k//2,bias=False)
        self.sig=nn.Sigmoid()
    def forward(self,x):
        y=self.avg(x)
        y=self.conv(y.squeeze(-1).transpose(-1,-2)).transpose(-1,-2).unsqueeze(-1)
        return x*self.sig(y)

class SpatialAtt(nn.Module):
    def __init__(self): super().__init__(); self.dw=nn.Sequential(nn.Conv2d(1,1,3,padding=1,bias=False), nn.Sigmoid())
    def forward(self,x): return x*self.dw(torch.mean(x,dim=1,keepdim=True))

class LBA_Block(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.eca=ECA(c); self.spa=SpatialAtt()
        self.alpha=nn.Parameter(torch.tensor(0.5))
        self.beta=nn.Parameter(torch.tensor(0.5))
    def forward(self,x): return self.alpha*self.eca(x)+self.beta*self.spa(x)

class DecoderBlock(nn.Module):
    def __init__(self,in_ch,skip_ch,out_ch):
        super().__init__()
        self.up=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)
        self.conv=nn.Sequential(
            nn.Conv2d(in_ch+skip_ch,out_ch,3,padding=1,bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch,out_ch,3,padding=1,bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
        self.lba=LBA_Block(out_ch)
    def forward(self,x,skip):
        x=self.up(x)
        x=torch.cat([x,skip],dim=1)
        x=self.conv(x)
        x=self.lba(x)
        return x

class LBA_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc=timm.create_model('mobilenetv3_small_100',pretrained=True,features_only=True)
        ch=self.enc.feature_info.channels()
        self.aspp=nn.Sequential(
            nn.Conv2d(ch[-1],96,3,padding=6,dilation=6,bias=False), nn.BatchNorm2d(96), nn.ReLU(),
            nn.Conv2d(96,96,3,padding=12,dilation=12,bias=False), nn.BatchNorm2d(96), nn.ReLU(),
            nn.Conv2d(96,96,3,padding=18,dilation=18,bias=False), nn.BatchNorm2d(96), nn.ReLU(),
            nn.Conv2d(96,96,1,bias=False), nn.BatchNorm2d(96), nn.ReLU()
        )
        self.dec4=DecoderBlock(96,ch[3],96)
        self.dec3=DecoderBlock(96,ch[2],64)
        self.dec2=DecoderBlock(64,ch[1],48)
        self.dec1=DecoderBlock(48,ch[0],24)
        self.head=nn.Conv2d(24,1,1)
    def forward(self,x):
        feats=self.enc(x)
        x=self.aspp(feats[-1])
        x=self.dec4(x,feats[3])
        x=self.dec3(x,feats[2])
        x=self.dec2(x,feats[1])
        x=self.dec1(x,feats[0])
        x=F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False)
        return self.head(x)

# ========= 6. Train/Val Helper ==========
def train_one_model(model_fn,model_name,epochs=EPOCHS,out_dir=OUT_DIR):
    print(f"\n=== Training {model_name} ===")
    train_ds=BUSIDataset(DATA_DIR,'train',IMG_SIZE)
    val_ds=BUSIDataset(DATA_DIR,'val',IMG_SIZE)
    train_loader=DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS)
    val_loader=DataLoader(val_ds,batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS)
    model=model_fn().to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=0.5,patience=10)
    best_val_loss=1e9
    history={'train_loss':[],'val_loss':[],'val_dice':[],'val_iou':[]}

    for ep in range(epochs):
        model.train(); running_loss=0; n=0
        for imgs,masks in train_loader:
            imgs,masks=imgs.to(device),masks.to(device)
            logits=model(imgs)
            loss=loss_fn(logits,masks)
            opt.zero_grad(); loss.backward(); opt.step()
            running_loss+=loss.item()*imgs.size(0); n+=imgs.size(0)
        train_loss=running_loss/n

        model.eval(); vloss=0; vdice=0; viou=0; vn=0
        with torch.no_grad():
            for imgs,masks in val_loader:
                imgs,masks=imgs.to(device),masks.to(device)
                logits=model(imgs)
                loss=loss_fn(logits,masks)
                vloss+=loss.item()*imgs.size(0)
                p=torch.sigmoid(logits)
                vdice+=dice_score(p,masks)*imgs.size(0)
                viou+=iou_score(p,masks)*imgs.size(0)
                vn+=imgs.size(0)
        val_loss=vloss/vn; val_dice=vdice/vn; val_iou=viou/vn
        history['train_loss'].append(train_loss); history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice); history['val_iou'].append(val_iou)
        scheduler.step(val_loss)
        print(f"Epoch {ep:03d} | train {train_loss:.4f} | val {val_loss:.4f} | Dice {val_dice:.4f} | IoU {val_iou:.4f}")

        if val_loss<best_val_loss:
            best_val_loss=val_loss
            ckpt=os.path.join(out_dir,f"{model_name}_best.pth")
            torch.save(model.state_dict(),ckpt)

    return history, ckpt, val_ds

# ========= 7. 轻量化模型列表 ==========
"""
def get_lightweight_models():
    return {
        "Mobile-UNet": lambda: smp.Unet(encoder_name="mobilenet_v3_small_100", encoder_weights="imagenet", in_channels=3, classes=1),
        "Mobile-UNet++": lambda: smp.UnetPlusPlus(encoder_name="mobilenet_v3_small_100", encoder_weights="imagenet", in_channels=3, classes=1),
        "Mobile-FPN": lambda: smp.FPN(encoder_name="mobilenet_v3_small_100", encoder_weights="imagenet", in_channels=3, classes=1),
        "Mobile-DeepLabV3+": lambda: smp.DeepLabV3Plus(encoder_name="mobilenet_v3_small_100", encoder_weights="imagenet", in_channels=3, classes=1),
        "LBA-Net": lambda: LBA_Net()
    }
"""
def get_lightweight_models():
    return {
        "Mobile-UNet": lambda: smp.Unet(encoder_name="mobilenet_v2", encoder_weights="imagenet", in_channels=3, classes=1),
        "Mobile-UNet++": lambda: smp.UnetPlusPlus(encoder_name="mobilenet_v2", encoder_weights="imagenet", in_channels=3, classes=1),
        "Mobile-FPN": lambda: smp.FPN(encoder_name="mobilenet_v2", encoder_weights="imagenet", in_channels=3, classes=1),
        "LBA-Net": lambda: LBA_Net()
    }

# ========= 8. 训练 & 记录 ==========
models = get_lightweight_models()
all_histories = {}
results = {}

for name,fn in models.items():
    hist,best_ckpt,val_dataset=train_one_model(fn,name,EPOCHS,OUT_DIR)
    all_histories[name]=hist
    model=fn().to(device)
    model.load_state_dict(torch.load(best_ckpt,map_location=device))
    model.eval()
    val_loader=DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=False)
    total_d,total_i,n=0,0,0
    with torch.no_grad():
        for imgs,masks in val_loader:
            imgs,masks=imgs.to(device),masks.to(device)
            p=torch.sigmoid(model(imgs))
            total_d+=dice_score(p,masks)*imgs.size(0)
            total_i+=iou_score(p,masks)*imgs.size(0)
            n+=imgs.size(0)
    results[name]={'dice':total_d/n,'iou':total_i/n,'ckpt':best_ckpt}
    print(f"\n{name} final val Dice={total_d/n:.4f}, IoU={total_i/n:.4f}")

# ========= 9. 可视化网格 (5 张样本) ==========
import matplotlib.pyplot as plt
vloader_small = DataLoader(val_dataset,batch_size=5,shuffle=True)
imgs_batch,masks_batch = next(iter(vloader_small))

plt.figure(figsize=(3*(2+len(models)),3*5))
for i in range(5):
    img = imgs_batch[i]
    gt  = masks_batch[i][0].numpy()

    # Input
    plt.subplot(5, 2+len(models), i*(2+len(models))+1)
    plt.imshow(img.permute(1,2,0).cpu().numpy()); plt.title("Input"); plt.axis('off')

    # GT
    plt.subplot(5, 2+len(models), i*(2+len(models))+2)
    plt.imshow(gt,cmap='gray'); plt.title("GT"); plt.axis('off')

    # 模型预测
    col=3
    for mname,mfn in models.items():
        model = mfn().to(device)
        ckpt = os.path.join(OUT_DIR,f"{mname}_best.pth")
        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt,map_location=device))
            model.eval()
            with torch.no_grad():
                pred = torch.sigmoid(model(img.unsqueeze(0).to(device)))[0,0].cpu().numpy()
        else:
            pred = np.zeros_like(gt)
        plt.subplot(5,2+len(models),i*(2+len(models))+col)
        plt.imshow(pred>0.5,cmap='gray'); plt.title(mname); plt.axis('off')
        col+=1

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"model_comparison_grid.png"),dpi=200)
plt.show()
print("All outputs saved to:", OUT_DIR)
