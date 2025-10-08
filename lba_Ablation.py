# =====================
#  消融实验（6变体×1次×72epoch）
# =====================
import os, time, numpy as np, pandas as pd
import torch, torch.nn as nn
import matplotlib.pyplot as plt
from thop import profile, clever_format
from tqdm import tqdm

# 超参（在此处改）
EPOCHS = 72
PATIENCE = 15
NUM_RUNS = 1          # 只跑1次
OUT_DIR = "/content/drive/MyDrive/LBA-Net-Comprehensive-Ablation"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------
# 6个核心变体
# ---------------------
ablation_configs = {
    "Baseline":     lambda: LBA_Net_Enhanced(use_aspp=False, use_boundary=False, attention_type='none'),
    "Full":         lambda: LBA_Net_Enhanced(use_aspp=True, use_boundary=True, attention_type='lba'),
    "-LBA":         lambda: LBA_Net_Enhanced(use_aspp=True, use_boundary=True, attention_type='none'),
    "-ASPP":        lambda: LBA_Net_Enhanced(use_aspp=False, use_boundary=True, attention_type='lba'),
    "-Boundary":    lambda: LBA_Net_Enhanced(use_aspp=True, use_boundary=False, attention_type='lba'),
    "CBAM":         lambda: LBA_Net_Enhanced(use_aspp=True, use_boundary=True, attention_type='cbam'),
}

# ---------------------
# 训练一个模型（单次）
# ---------------------
def train_oneShot(model_fn, name):
    print(f'\n>>> {name} <<<')
    train_ds = BUSIDataset(DATA_DIR, 'train', IMG_SIZE)
    val_ds   = BUSIDataset(DATA_DIR, 'val', IMG_SIZE)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = model_fn().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    stopper = EarlyStop(patience=PATIENCE)
    best_dice = 0
    ckpt_path = os.path.join(OUT_DIR, f"{name}.pth")

    use_boundary = getattr(model, 'use_boundary', False)

    for epoch in range(EPOCHS):
        # ---- 训练 ----
        model.train()
        for imgs, masks, bdys in tqdm(train_loader, desc=f'{name} E{epoch}'):
            imgs, masks, bdys = imgs.to(device), masks.to(device), bdys.to(device)
            seg, bdy = model(imgs)
            loss = total_loss_ablation(seg, bdy, masks, bdys, use_boundary)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        # ---- 验证 ----
        model.eval()
        dice_val = 0.0
        with torch.no_grad():
            for imgs, masks, bdys in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                seg, _ = model(imgs)
                dice_val += dice_score(torch.sigmoid(seg), masks) * imgs.size(0)
        dice_val /= len(val_loader.dataset)

        if dice_val > best_dice:
            best_dice = dice_val
            torch.save(model.state_dict(), ckpt_path)
        if not stopper.step(dice_val):
            break
    return best_dice, ckpt_path

# ---------------------
# 测参数量 & FLOPs & FPS
# ---------------------
def measure_all(model_fn):
    model = model_fn().to(device).eval()
    dummy = torch.randn(1, 3, 512, 512).to(device)
    # 只计分割分支
    class SegWrap(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): return self.m(x)[0]
    wrap = SegWrap(model).to(device)
    flops, params = profile(wrap, inputs=(dummy,), verbose=False)
    flops_g, params_m = clever_format([flops, params], "%.3f")

    # FPS
    gpu_fps = measure_fps(model, 'cuda')
    cpu_fps = measure_fps(model.cpu(), 'cpu')
    return flops_g, params_m, gpu_fps, cpu_fps

# ---------------------
# 主流程：6变体×1次
# ---------------------
def run_light_ablation():
    results = []
    for name, model_fn in ablation_configs.items():
        dice, ckpt = train_oneShot(model_fn, name)
        flops, params, gpu_fps, cpu_fps = measure_all(model_fn)
        results.append({
            "Model": name,
            "Dice": round(dice, 4),
            "GFLOPs": flops,
            "Params": params,
            "GPU_FPS": round(gpu_fps, 2),
            "CPU_FPS": round(cpu_fps, 2)
        })
        print(f'{name}  Done | Dice={dice:.4f}  | {params}  | {flops}')
    return pd.DataFrame(results)

# ---------------------
# 一键执行
# -------------------------
if __name__ == "__main__":
    df = run_light_ablation()
    csv_path = os.path.join(OUT_DIR, "light_ablation.csv")
    df.to_csv(csv_path, index=False)
    print("\n=== 结果 ===")
    print(df)

    # 简单柱状图
    plt.figure(figsize=(8,3))
    plt.bar(df["Model"], df["Dice"], color='skyblue', edgecolor='navy')
    plt.xticks(rotation=45, ha='right'); plt.title("Dice"); plt.ylabel("Dice")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ablation_dice.png"), dpi=300, bbox_inches='tight')
    plt.show()
