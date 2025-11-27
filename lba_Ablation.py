# ablation-lba.py
import os
import time
import random
import re
import csv
import cv2
import numpy as np
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.model_selection import KFold

from thop import profile, clever_format
import pandas as pd
from LBA_Net import (
    LBA_Net_BoundaryGuided,
    boundary_guided_total_loss,
    dice_score,
    iou_score,
    recall_score,
    hd95_score,
    create_optimizer,
)

# ===================== 从主模型文件导入 =====================
DATA_DIR = "/home/wang/ultrasound/Dataset_BUSI_with_GT-new"#BUET_BUSD-new
#DATA_DIR = "/home/wang/ultrasound/BUET_BUSD-new"
#OUT_DIR = "/home/wang/ultrasound/unet-BUET"
OUT_DIR  = "/home/wang/ultrasound/LBA-BUSIablation"
os.makedirs(OUT_DIR, exist_ok=True)

IMG_SIZE = 512
BATCH_SIZE = 24
EPOCHS = 300
NUM_WORKERS = 12
USE_AMP = True

# ================ 若主文件里已经有 BUSIDataset，可以删掉下面整个类 =================
class BUSIDataset(Dataset):
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

            image_files = []
            for fname in os.listdir(cls_dir):
                if 'mask' in fname.lower():
                    continue
                image_files.append(fname)

            for img_fname in image_files:
                img_path = os.path.join(cls_dir, img_fname)
                base_name = os.path.splitext(img_fname)[0]
                ext = os.path.splitext(img_fname)[1].lower()
                pattern = re.compile(
                    rf"^{re.escape(base_name)}_mask(_\d+)?{re.escape(ext)}$",
                    re.IGNORECASE
                )
                mask_files = []
                for mask_fname in os.listdir(cls_dir):
                    if pattern.match(mask_fname):
                        mask_files.append(os.path.join(cls_dir, mask_fname))

                if not mask_files:
                    print(f"Warning: No mask found for {img_path}, skipping...")
                    continue

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
                        merged = np.logical_or(merged, m).astype(np.uint8)

                if merged is not None:
                    self.imgs.append(img_path)
                    self.masks.append(merged)
                    if len(mask_files) > 1:
                        print(f"Multi-mask: {img_fname} -> {len(mask_files)} masks merged")

        ids = list(range(len(self.imgs)))
        random.shuffle(ids)
        if split == 'all':
            # 使用全部数据
            pass
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
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.Rotate(limit=30, p=0.5),
                A.RandomBrightnessContrast(p=0.4),
                A.GaussianBlur(blur_limit=3, p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], is_check_shapes=False)
        else:
            self.aug = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], is_check_shapes=False)

        assert len(self.imgs) == len(self.masks), \
            f"Length mismatch: {len(self.imgs)} vs {len(self.masks)}"
        print(f"Dataset {split}: {len(self.imgs)} samples loaded.")

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.imgs[idx]), cv2.COLOR_BGR2RGB)
        mask = self.masks[idx]

        if mask.sum() == 0:
            print(f"Warning: All-zero mask at {self.imgs[idx]}")

        aug = self.aug(image=img, mask=mask)
        img, mask = aug["image"], aug["mask"]

        if isinstance(mask, torch.Tensor):
            mask = mask.unsqueeze(0).float()
        else:
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        mask = (mask > 0.5).float()

        bdy = cv2.morphologyEx(
            mask.squeeze().cpu().numpy().astype(np.uint8),
            cv2.MORPH_GRADIENT,
            np.ones((3, 3), np.uint8)
        )
        bdy = torch.from_numpy(cv2.resize(
            bdy, (self.img_size, self.img_size)
        )).unsqueeze(0).float()

        if torch.isnan(img).any() or torch.isinf(img).any():
            raise ValueError(f"NaN/Inf in image: {self.imgs[idx]}")
        if torch.isnan(mask).any() or torch.isnan(bdy).any():
            raise ValueError(f"NaN in mask/bdy: {self.imgs[idx]}")
        return img, mask, bdy

    def __len__(self):
        return len(self.imgs)


# =============================================================

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_DIR, exist_ok=True)

# ==================== Ablation Configs =======================

ABLATION_SETTINGS = {
    "full": {  # Full (LBA-Net)
        "name_show": "Full (LBA-Net)",
        "use_boundary_guidance": True,
        "use_boundary_head": True,
        "use_consistency_loss": True,
        "use_lba_block": True,
        "use_aspp": True,
    },
    "w/o_boundary_guidance": {
        "name_show": "w/o Boundary Guidance",
        "use_boundary_guidance": False,
        "use_boundary_head": True,
        "use_consistency_loss": True,
        "use_lba_block": True,
        "use_aspp": True,
    },
    "w/o_boundary_head": {  # single-head
        "name_show": "w/o Boundary Head (single-head)",
        "use_boundary_guidance": True,
        "use_boundary_head": False,
        "use_consistency_loss": False,
        "use_lba_block": True,
        "use_aspp": True,
    },
    "w/o_boundary_consistency": {
        "name_show": "w/o Boundary-Consistency Loss",
        "use_boundary_guidance": True,
        "use_boundary_head": True,
        "use_consistency_loss": False,
        "use_lba_block": True,
        "use_aspp": True,
    },
    "w/o_lba_block": {
        "name_show": "w/o LBA-Block",
        "use_boundary_guidance": True,
        "use_boundary_head": True,
        "use_consistency_loss": True,
        "use_lba_block": False,
        "use_aspp": True,
    },
    "w/o_aspp": {
        "name_show": "w/o ASPP",
        "use_boundary_guidance": True,
        "use_boundary_head": True,
        "use_consistency_loss": True,
        "use_lba_block": True,
        "use_aspp": False,
    },
}

# =============== 复杂度 & 速度测试函数 ======================


def compute_model_complexity(model, img_size=512):
    """计算 Params 和 FLOPs"""
    model = model.to(device)
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size).to(device)
    with torch.no_grad():
        flops, params = profile(model, inputs=(dummy,), verbose=False)
    flops_str, params_str = clever_format([flops, params], "%.3f")
    return flops_str, params_str  # e.g. '12.345G', '3.210M'


def measure_gpu_fps(model, img_size=512, runs=200):
    """GPU 上的推理 FPS"""
    if not torch.cuda.is_available():
        return 0.0
    model = model.to("cuda")
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size).cuda()

    with torch.no_grad():
        # warm-up
        for _ in range(20):
            _ = model(dummy)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(runs):
            _ = model(dummy)
        torch.cuda.synchronize()
        end = time.time()

    fps = runs / (end - start)
    return fps


def measure_cpu_fps(model, img_size=512, runs=100):
    """CPU 上的推理 FPS"""
    model = model.to("cpu")
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size)

    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy)
        start = time.time()
        for _ in range(runs):
            _ = model(dummy)
        end = time.time()

    fps = runs / (end - start)
    return fps


def evaluate_model_complexity_and_speed(cfg):
    """
    对某个消融配置实例化一个模型（未训练也没关系），
    计算 Params, FLOPs, GPU FPS, CPU FPS
    """
    model = LBA_Net_BoundaryGuided(
        use_boundary_guidance=cfg["use_boundary_guidance"],
        use_boundary_head=cfg["use_boundary_head"],
        use_lba_block=cfg["use_lba_block"],
        use_aspp=cfg["use_aspp"],
    )
    flops_str, params_str = compute_model_complexity(model, IMG_SIZE)
    gpu_fps = measure_gpu_fps(model, IMG_SIZE)
    cpu_fps = measure_cpu_fps(model, IMG_SIZE)
    return {
        "Params": params_str,
        "FLOPs": flops_str,
        "GPU_FPS": gpu_fps,
        "CPU_FPS": cpu_fps,
    }


# =================== 训练 & 评估（K-Fold） ====================
def train_and_eval_ablation(ablation_key, cfg, k_folds=5):
    """
    对单个消融配置做 K-Fold 训练 + 验证：
    - 每个 fold 按 best Val Dice 选最佳 epoch
    - 保存每个 fold 的最佳模型 checkpoint
    - 返回各项指标的 mean±std（基于每 fold 的 best 指标）
    """
    print(f"\n========== Ablation: {cfg['name_show']} ({ablation_key}) ==========")

    # 建议在主文件开头加上这两行：
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    full_ds = BUSIDataset(DATA_DIR, "all", IMG_SIZE)
    print(f"Total dataset size: {len(full_ds)}")

    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=SEED)

    dice_list, iou_list, recall_list, hd95_list = [], [], [], []
    fold_best_metrics = []  # 记录每个 fold 的最佳指标

    # ⚠️ 关键：用安全 key（把 "/" 换掉）
    safe_key = ablation_key.replace("/", "_")

    # checkpoint 保存目录：OUT_DIR/ablations/<safe_key>/
    ablation_ckpt_dir = os.path.join(OUT_DIR, "ablations", safe_key)
    os.makedirs(ablation_ckpt_dir, exist_ok=True)

    for fold, (train_ids, val_ids) in enumerate(kfold.split(full_ds.imgs)):
        print(f"\n{'=' * 50}")
        print(f"[{cfg['name_show']}] Fold {fold + 1}/{k_folds}")
        print(f"{'=' * 50}")

        train_sampler = SubsetRandomSampler(train_ids)
        val_sampler = SubsetRandomSampler(val_ids)

        train_loader = DataLoader(
            full_ds,
            batch_size=BATCH_SIZE,
            sampler=train_sampler,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            full_ds,
            batch_size=1,
            sampler=val_sampler,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        # ====== 初始化模型、优化器、调度器 ======
        model = LBA_Net_BoundaryGuided(
            use_boundary_guidance=cfg["use_boundary_guidance"],
            use_boundary_head=cfg["use_boundary_head"],
            use_lba_block=cfg["use_lba_block"],
            use_aspp=cfg["use_aspp"],
        ).to(device)

        optimizer = create_optimizer(model)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=[1e-4, 2e-3, 3e-3, 3e-3],
            epochs=EPOCHS,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
        )
        scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

        best_dice_fold = -1.0
        best_metrics_fold = None
        best_epoch_fold = -1
        best_ckpt_path = None

        for epoch in range(EPOCHS):
            # ----------------- Train -----------------
            model.train()
            running_loss = 0.0
            pbar = tqdm(
                train_loader,
                desc=f"[{ablation_key}] Fold {fold + 1} | Epoch {epoch + 1}/{EPOCHS}",
            )

            for imgs, masks, bdys in pbar:
                imgs = imgs.to(device)
                masks = masks.to(device)
                bdys = bdys.to(device)

                optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    seg_logits, bdy_logits, boundary_att = model(imgs)
                    loss, _ = boundary_guided_total_loss(
                        seg_logits,
                        bdy_logits,
                        boundary_att,
                        masks,
                        bdys,
                        use_boundary_head=cfg["use_boundary_head"],
                        use_consistency_loss=cfg["use_consistency_loss"],
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                running_loss += loss.item() * imgs.size(0)
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            epoch_loss = running_loss / len(train_ids)

            # ----------------- Validation -----------------
            model.eval()
            fold_dice_scores, fold_iou_scores = [], []
            fold_recall_scores, fold_hd95_scores = [], []

            with torch.no_grad():
                for imgs, masks, _ in val_loader:
                    imgs = imgs.to(device)
                    masks = masks.to(device)

                    with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                        seg_logits, _, _ = model(imgs)
                        seg_pred = torch.sigmoid(seg_logits)

                    fold_dice_scores.append(dice_score(seg_pred, masks))
                    fold_iou_scores.append(iou_score(seg_pred, masks))
                    fold_recall_scores.append(recall_score(seg_pred, masks))
                    fold_hd95_scores.append(hd95_score(seg_pred, masks))

            mean_dice_fold = float(np.mean(fold_dice_scores))
            mean_iou_fold = float(np.mean(fold_iou_scores))
            mean_recall_fold = float(np.mean(fold_recall_scores))
            mean_hd95_fold = float(np.mean(fold_hd95_scores))

            print(
                f"[{ablation_key}] Epoch {epoch + 1}/{EPOCHS} | "
                f"Train Loss: {epoch_loss:.4f} | "
                f"Val Dice: {mean_dice_fold:.4f} | "
                f"IoU: {mean_iou_fold:.4f} | "
                f"Recall: {mean_recall_fold:.4f} | "
                f"HD95: {mean_hd95_fold:.2f}"
            )

            # ====== 按 best Dice 选择最佳 epoch，并保存 checkpoint ======
            if mean_dice_fold > best_dice_fold:
                best_dice_fold = mean_dice_fold
                best_epoch_fold = epoch + 1
                best_metrics_fold = {
                    "dice": mean_dice_fold,
                    "iou": mean_iou_fold,
                    "recall": mean_recall_fold,
                    "hd95": mean_hd95_fold,
                }

                best_ckpt_path = os.path.join(
                    ablation_ckpt_dir,
                    f"best_{safe_key}_fold{fold + 1}.pth",
                )
                torch.save(
                    {
                        "ablation_key": ablation_key,  # 原始 key 继续保留
                        "cfg": cfg,
                        "fold": fold,
                        "epoch": best_epoch_fold,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_metrics": best_metrics_fold,
                        "val_indices": val_ids,
                    },
                    best_ckpt_path,
                )
                print(
                    f"  🔥 New best fold-{fold + 1} model saved at epoch {best_epoch_fold} "
                    f"(Dice={best_dice_fold:.4f}) → {best_ckpt_path}"
                )

        # ====== 一个 fold 训练完成后，使用 best epoch 的指标 ======
        if best_metrics_fold is None:
            raise RuntimeError(f"No best metrics recorded for fold {fold + 1} of {ablation_key}")

        print(
            f"\n✅ [{cfg['name_show']}] Fold {fold + 1} BEST @ epoch {best_epoch_fold}: "
            f"Dice={best_metrics_fold['dice'] * 100:.2f}%, "
            f"IoU={best_metrics_fold['iou'] * 100:.2f}%, "
            f"Recall={best_metrics_fold['recall'] * 100:.2f}%, "
            f"HD95={best_metrics_fold['hd95']:.2f}px"
        )

        dice_list.append(best_metrics_fold["dice"])
        iou_list.append(best_metrics_fold["iou"])
        recall_list.append(best_metrics_fold["recall"])
        hd95_list.append(best_metrics_fold["hd95"])
        fold_best_metrics.append(best_metrics_fold)

    # ====== 汇总：计算 mean ± std ======
    def mean_std(arr):
        arr = np.array(arr)
        return float(np.mean(arr)), float(np.std(arr, ddof=1))

    dice_mean, dice_std = mean_std(dice_list)
    iou_mean, iou_std = mean_std(iou_list)
    recall_mean, recall_std = mean_std(recall_list)
    hd95_mean, hd95_std = mean_std(hd95_list)

    print("\n============= Per-fold Best Results =============")
    for i, m in enumerate(fold_best_metrics):
        print(
            f"Fold {i + 1}: "
            f"Dice={m['dice'] * 100:.2f}%, "
            f"IoU={m['iou'] * 100:.2f}%, "
            f"Recall={m['recall'] * 100:.2f}%, "
            f"HD95={m['hd95']:.2f}px"
        )

    print("\n============= Cross-fold Summary (best-epoch per fold) =============")
    print(
        f"Dice:   {dice_mean * 100:.2f} ± {dice_std * 100:.2f} %\n"
        f"IoU:    {iou_mean * 100:.2f} ± {iou_std * 100:.2f} %\n"
        f"Recall: {recall_mean * 100:.2f} ± {recall_std * 100:.2f} %\n"
        f"HD95:   {hd95_mean:.2f} ± {hd95_std:.2f} px"
    )

    return {
        "Dice_mean": dice_mean,
        "Dice_std": dice_std,
        "IoU_mean": iou_mean,
        "IoU_std": iou_std,
        "Recall_mean": recall_mean,
        "Recall_std": recall_std,
        "HD95_mean": hd95_mean,
        "HD95_std": hd95_std,
    }



# ============================================================
# 6. 运行全部消融实验 + 保存结果
# ============================================================

def run_all_ablations():
    all_results = {}

    for ab_key, cfg in ABLATION_SETTINGS.items():
        print(f"\n\n================ Running Ablation: {cfg['name_show']} ================")

        # 复杂度 & 速度
        complexity = evaluate_model_complexity_and_speed(cfg)

        # 5-fold 训练 + 评估
        metrics = train_and_eval_ablation(ab_key, cfg, k_folds=5)

        all_results[ab_key] = {
            "name_show": cfg["name_show"],
            "mean_dice": metrics["Dice_mean"],
            "dice_std": metrics["Dice_std"],
            "iou_mean": metrics["IoU_mean"],
            "iou_std": metrics["IoU_std"],
            "recall_mean": metrics["Recall_mean"],
            "recall_std": metrics["Recall_std"],
            "hd95_mean": metrics["HD95_mean"],
            "hd95_std": metrics["HD95_std"],
            "Params": complexity["Params"],
            "FLOPs": complexity["FLOPs"],
            "GPU_FPS": complexity["GPU_FPS"],
            "CPU_FPS": complexity["CPU_FPS"],
        }

    return all_results


# ============================================================
# 7. 自动生成论文表格（CSV + LaTeX）
# ============================================================

def save_results_table(all_results, save_path):
    rows = []
    for key, res in all_results.items():
        rows.append({
            "Variant": res["name_show"],
            "Dice (%)": f"{res['mean_dice']*100:.2f}",
            "Dice std": f"{res['dice_std']*100:.2f}",
            "IoU (%)": f"{res['iou_mean']*100:.2f}",
            "IoU std": f"{res['iou_std']*100:.2f}",
            "Recall (%)": f"{res['recall_mean']*100:.2f}",
            "Recall std": f"{res['recall_std']*100:.2f}",
            "HD95 (px)": f"{res['hd95_mean']:.2f}",
            "HD95 std": f"{res['hd95_std']:.2f}",
            "Params": res["Params"],
            "FLOPs": res["FLOPs"],
            "GPU FPS": f"{res['GPU_FPS']:.2f}",
            "CPU FPS": f"{res['CPU_FPS']:.2f}",
        })

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"\nSaved ablation results table to: {save_path}")
    print(df)


def save_latex_table(all_results, latex_path):
    with open(latex_path, "w") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Ablation study of Boundary-Guided LBA-Net on BUSI*.}\n")
        f.write("\\begin{tabular}{lcccccccc}\n")
        f.write("\\toprule\n")
        f.write("Variant & Dice(\\%) & IoU(\\%) & Recall(\\%) & HD95(px) & Params & FLOPs & GPU~FPS & CPU~FPS \\\\\n")
        f.write("\\midrule\n")

        for key, res in all_results.items():
            name = res["name_show"].replace("_", "-")
            f.write(
                f"{name} & "
                f"{res['mean_dice']*100:.2f} $\\pm$ {res['dice_std']*100:.2f} & "
                f"{res['iou_mean']*100:.2f} $\\pm$ {res['iou_std']*100:.2f} & "
                f"{res['recall_mean']*100:.2f} $\\pm$ {res['recall_std']*100:.2f} & "
                f"{res['hd95_mean']:.2f} $\\pm$ {res['hd95_std']:.2f} & "
                f"{res['Params']} & {res['FLOPs']} & "
                f"{res['GPU_FPS']:.2f} & {res['CPU_FPS']:.2f} \\\\\n"
            )

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"LaTeX ablation table saved to: {latex_path}")


# ============================================================
# 8. 主入口（Main）
# ============================================================

if __name__ == "__main__":
    print("\n==============================")
    print(" Boundary-Guided LBA-Net Ablation Runner ")
    print("==============================\n")

    # 运行所有消融实验
    all_results = run_all_ablations()

    # 保存 CSV
    csv_path = os.path.join(OUT_DIR, "ablation_results.csv")
    save_results_table(all_results, csv_path)

    # 保存 LaTeX
    tex_path = os.path.join(OUT_DIR, "ablation_results.tex")
    save_latex_table(all_results, tex_path)

    # 终端简单打印一版汇总
    print("\n============= Ablation Summary (mean over 5-fold) =============")
    header = (
        f"{'Variant':35s} | {'Dice':>8s} | {'IoU':>8s} | "
        f"{'Recall':>8s} | {'HD95':>8s} | {'Params':>8s} | {'FLOPs':>8s} | "
        f"{'GPU FPS':>8s} | {'CPU FPS':>8s}"
    )
    print(header)
    print("-" * len(header))
    for key, res in all_results.items():
        print(
            f"{res['name_show'][:35]:35s} | "
            f"{res['mean_dice']*100:8.2f} | "
            f"{res['iou_mean']*100:8.2f} | "
            f"{res['recall_mean']*100:8.2f} | "
            f"{res['hd95_mean']:8.2f} | "
            f"{res['Params']:>8s} | {res['FLOPs']:>8s} | "
            f"{res['GPU_FPS']:8.2f} | {res['CPU_FPS']:8.2f}"
        )

    print("\n=== All Ablation Experiments Completed ===\n")


