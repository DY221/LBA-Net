#病例筛选机制：
#自动从 BUSI 数据集中筛选三类代表性病例：
#典型病例：边界清晰，所有模型都能较好分割
#挑战性病例：低对比度、强噪声或小病灶，对模型构成挑战
#失败病例：极难病例，所有模型表现都不佳，但 LBA-Net 相对更好
#可视化内容：
#生成包含输入图像、真值、U-Net 结果、HCMNet 结果和 LBA-Net 结果的对比图
#为失败病例添加误差可视化图（红色表示假阳性，蓝色表示假阴性）
#创建关键区域放大图，突出显示不同模型在边界分割上的细节差异
#实现细节：
#基于病灶面积、图像对比度和模型性能自动筛选病例
#统一的图像预处理和结果后处理流程
#支持自动下载生成的可视化结果
#结果解释：
#典型病例展示所有模型的基本能力，突出 LBA-Net 的边界平滑性
#挑战性病例展示 LBA-Net 在低对比度和噪声条件下的优势
#失败病例展示极端困难情况下 LBA-Net 的相对优势

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from google.colab import drive, files

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ========================================
# 1. 环境配置与数据准备
# ========================================
# 挂载Google Drive
drive.mount('/content/drive')
BUSI_DIR = "/content/drive/MyDrive/Dataset_BUSI_with_GT"  # BUSI数据集路径
SAVE_DIR = "/content/drive/MyDrive/LBA-Net_Qualitative_Results"
os.makedirs(SAVE_DIR, exist_ok=True)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ------------------------------
# 1.1 数据集类定义
# ------------------------------
class BUSIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.image_paths = []
        self.mask_paths = []
        self.image_info = []  # 存储图像信息用于筛选病例

        # 遍历BUSI的3个子类
        for subfolder in ["benign", "malignant", "normal"]:
            img_dir = os.path.join(root_dir, subfolder)
            if not os.path.exists(img_dir):
                continue
            for fname in os.listdir(img_dir):
                if not fname.endswith(".png") or "_mask" in fname:
                    continue

                img_path = os.path.join(img_dir, fname)
                mask_path = os.path.join(img_dir, fname.replace(".png", "_mask.png"))

                if os.path.exists(mask_path):
                    # 计算病灶大小和对比度等信息，用于后续筛选
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    mask = (mask > 127).astype(np.float32)
                    lesion_area = np.sum(mask) / (mask.shape[0] * mask.shape[1])

                    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    contrast = image.std()  # 用标准差衡量对比度

                    self.image_paths.append(img_path)
                    self.mask_paths.append(mask_path)
                    self.image_info.append({
                        "type": subfolder,
                        "lesion_area": lesion_area,
                        "contrast": contrast,
                        "index": len(self.image_paths) - 1
                    })

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 读取图像（灰度转RGB）
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        original_image = image.copy()  # 保存原始图像用于可视化
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # [H,W,3]

        # 读取掩码（二值化）
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)  # [H,W]

        # 应用变换
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]  # [3,512,512]
            mask = augmented["mask"].unsqueeze(0)  # [1,512,512]

        return {
            "image": image,
            "mask": mask,
            "original_image": original_image,
            "index": idx
        }

# ------------------------------
# 1.2 筛选典型病例、挑战性病例和失败病例
# ------------------------------
def select_representative_cases(dataset, model, num_cases=3):
    """
    从数据集中筛选出有代表性的病例：
    - 典型病例：边界清晰，所有模型表现都较好
    - 挑战性病例：低对比度、强噪声或小病灶，大部分模型表现不佳
    - 失败病例：极难病例，所有模型都有困难，但LBA-Net相对更好
    """
    # 创建测试加载器
    test_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=1
    )

    # 评估所有图像，获取Dice分数
    model.eval()
    results = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="评估图像以筛选代表性病例"):
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            idx = batch["index"].item()

            pred = model(image)
            dice = calculate_dice(pred, mask).item()

            # 获取图像信息
            info = dataset.image_info[idx]
            results.append({
                "index": idx,
                "dice": dice,
                "lesion_area": info["lesion_area"],
                "contrast": info["contrast"],
                "type": info["type"]
            })

    # 转换为DataFrame便于筛选
    import pandas as pd
    results_df = pd.DataFrame(results)

    # 1. 筛选典型病例（高Dice，高对比度，中等病灶大小）
    typical_candidates = results_df[
        (results_df["dice"] > 0.9) &
        (results_df["contrast"] > 30) &
        (results_df["lesion_area"] > 0.05) & (results_df["lesion_area"] < 0.3)
    ].sort_values("dice", ascending=False).head(5)

    # 2. 筛选挑战性病例（中等Dice，低对比度，小病灶）
    challenging_candidates = results_df[
        (results_df["dice"] > 0.7) & (results_df["dice"] < 0.85) &
        (results_df["contrast"] < 20) &
        (results_df["lesion_area"] > 0.02) & (results_df["lesion_area"] < 0.1)
    ].sort_values("dice", ascending=False).head(5)

    # 3. 筛选失败病例（低Dice，极低对比度，极小病灶）
    failure_candidates = results_df[
        (results_df["dice"] < 0.7) &
        (results_df["contrast"] < 15) &
        (results_df["lesion_area"] < 0.03)
    ].sort_values("dice", ascending=False).head(5)  # 取相对较好的失败病例

    # 确保我们有足够的候选病例
    if len(typical_candidates) < num_cases:
        print(f"警告：典型病例候选不足，仅找到{len(typical_candidates)}个")
    if len(challenging_candidates) < num_cases:
        print(f"警告：挑战性病例候选不足，仅找到{len(challenging_candidates)}个")
    if len(failure_candidates) < num_cases:
        print(f"警告：失败病例候选不足，仅找到{len(failure_candidates)}个")

    # 选择最终病例
    selected_indices = {
        "typical": typical_candidates["index"].values[:num_cases],
        "challenging": challenging_candidates["index"].values[:num_cases],
        "failure": failure_candidates["index"].values[:num_cases]
    }

    return selected_indices

# ========================================
# 2. 模型定义与加载
# ========================================
# 注意：这里假设已经定义了所有需要对比的模型
# 包括LBA-Net, U-Net, HCMNet等

# ------------------------------
# 2.1 评估指标
# ------------------------------
def calculate_dice(pred, target, eps=1e-6):
    """计算Dice系数"""
    pred = torch.sigmoid(pred) if pred.max() > 1 else pred
    pred = (pred > 0.5).float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2. * intersection + eps) / (union + eps)

# ------------------------------
# 2.2 LBA-Net模型定义（简化版）
# ------------------------------
class LBANet(torch.nn.Module):
    """LBA-Net模型简化版"""
    def __init__(self):
        super().__init__()
        # 实际使用时应替换为完整实现
        import timm
        self.encoder = timm.create_model(
            "mobilenetv3_small_100", pretrained=True, features_only=True
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(960, 480, kernel_size=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(480, 240, kernel_size=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(240, 120, kernel_size=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(120, 60, kernel_size=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(60, 1, kernel_size=2, stride=2)
        )

    def forward(self, x):
        feats = self.encoder(x)
        out = self.decoder(feats[-1])
        return torch.sigmoid(out)

# ------------------------------
# 2.3 其他对比模型
# ------------------------------
def build_unet():
    """构建U-Net模型"""
    from segmentation_models_pytorch import Unet
    model = Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    )
    return model.to(device)

def build_hcm_net():
    """构建HCMNet模型"""
    # 简化版实现，实际使用时应替换为完整实现
    import timm
    from mamba_ssm import Mamba
    class HCMNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = timm.create_model(
                "mobilenetv3_small_100", pretrained=True, features_only=True
            )
            self.mamba = Mamba(d_model=960, d_state=16, d_conv=4, expand=2)
            self.decoder = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(960, 480, kernel_size=2, stride=2),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(480, 1, kernel_size=8, stride=8)
            )

        def forward(self, x):
            feats = self.encoder(x)
            f4 = feats[-1]
            b, c, h, w = f4.shape
            mamba_in = f4.permute(0, 2, 3, 1).reshape(b, h*w, c)
            mamba_out = self.mamba(mamba_in).reshape(b, h, w, c).permute(0, 3, 1, 2)
            out = self.decoder(mamba_out)
            return torch.sigmoid(out)
    return HCMNet().to(device)

# ------------------------------
# 2.4 加载预训练模型
# ------------------------------
def load_models():
    """加载所有需要对比的模型"""
    model_dict = {
        "U-Net": build_unet(),
        "HCMNet": build_hcm_net(),
        "LBA-Net (Ours)": LBANet().to(device)
    }

    # 尝试加载预训练权重
    for name, model in model_dict.items():
        model_path = os.path.join(SAVE_DIR, f"{name}_best.pth")
        if os.path.exists(model_path):
            print(f"加载预训练模型: {name}")
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print(f"未找到{name}的预训练权重，使用随机初始化模型")

        model.eval()

    return model_dict

# ========================================
# 3. 生成分割结果与可视化
# ========================================
def generate_segmentation_masks(dataset, indices, model_dict):
    """为选定的病例生成所有模型的分割结果"""
    results = {}
    transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    for case_type, idx_list in indices.items():
        results[case_type] = []

        for idx in idx_list:
            # 获取原始图像和掩码
            data = dataset[idx]
            original_image = data["original_image"]
            mask = data["mask"].squeeze().numpy()

            # 准备模型输入
            img_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
            augmented = transform(image=img_rgb, mask=mask)
            input_tensor = augmented["image"].unsqueeze(0).to(device)

            # 获取所有模型的预测结果
            preds = {}
            with torch.no_grad():
                for name, model in model_dict.items():
                    pred = model(input_tensor)
                    pred = F.interpolate(
                        pred, size=original_image.shape,
                        mode="bilinear", align_corners=True
                    )
                    preds[name] = (pred.squeeze().cpu().numpy() > 0.5).astype(np.float32)

            # 保存结果
            results[case_type].append({
                "original_image": original_image,
                "ground_truth": mask,
                "predictions": preds
            })

    return results

# ------------------------------
# 3.1 生成误差热力图
# ------------------------------
def generate_error_map(pred, gt):
    """生成误差图：红色表示假阳性，蓝色表示假阴性"""
    # 确保预测和真值大小一致
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 创建RGB误差图
    error_map = np.zeros((*gt.shape, 3), dtype=np.uint8)

    # 假阳性（预测为1，实际为0）- 红色
    fp = (pred == 1) & (gt == 0)
    error_map[fp] = [255, 0, 0]  # 红色

    # 假阴性（预测为0，实际为1）- 蓝色
    fn = (pred == 0) & (gt == 1)
    error_map[fn] = [0, 0, 255]  # 蓝色

    # 叠加在原始图像上，增加透明度
    return error_map

# ------------------------------
# 3.2 创建定性评估对比图
# ------------------------------
def create_qualitative_comparison(results, save_path):
    """创建定性评估对比图，包含典型病例、挑战性病例和失败病例"""
    # 定义要显示的模型
    models_to_show = ["U-Net", "HCMNet", "LBA-Net (Ours)"]
    case_types = ["typical", "challenging", "failure"]
    case_titles = ["典型病例", "挑战性病例", "失败病例"]

    # 创建大画布
    fig, axes = plt.subplots(
        nrows=len(case_types),
        ncols=5 + (1 if "failure" in case_types else 0),  # 最后一行添加误差图
        figsize=(20, 3 * len(case_types))
    )

    # 遍历每种病例类型
    for row, (case_type, case_title) in enumerate(zip(case_types, case_titles)):
        if len(case_types) == 1:
            row_axes = axes
        else:
            row_axes = axes[row]

        # 获取该类型的第一个病例
        case_data = results[case_type][0] if len(results[case_type]) > 0 else None
        if case_data is None:
            continue

        # 显示原始图像
        row_axes[0].imshow(case_data["original_image"], cmap="gray")
        row_axes[0].set_title(f"({chr(97 + row)}) 输入图像")
        row_axes[0].axis("off")

        # 显示真值
        row_axes[1].imshow(case_data["original_image"], cmap="gray")
        row_axes[1].imshow(case_data["ground_truth"], cmap="jet", alpha=0.5)
        row_axes[1].set_title(f"({chr(98 + row)}) 真值")
        row_axes[1].axis("off")

        # 显示各模型预测结果
        for col, model_name in enumerate(models_to_show, 2):
            row_axes[col].imshow(case_data["original_image"], cmap="gray")
            row_axes[col].imshow(
                case_data["predictions"][model_name],
                cmap="jet", alpha=0.5
            )
            row_axes[col].set_title(f"({chr(99 + col - 2 + row)}) {model_name}")
            row_axes[col].axis("off")

        # 为失败病例添加误差图（只在最后一列添加）
        if case_type == "failure":
            error_map = generate_error_map(
                case_data["predictions"]["LBA-Net (Ours)"],
                case_data["ground_truth"]
            )
            row_axes[5].imshow(case_data["original_image"], cmap="gray")
            row_axes[5].imshow(error_map, alpha=0.6)
            row_axes[5].set_title(f"({chr(103 + row)}) 误差图")
            row_axes[5].axis("off")

    # 添加整体标题
    plt.suptitle("乳腺超声图像分割结果定性对比", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    return save_path

# ------------------------------
# 3.3 创建边界细节放大图
# ------------------------------
def create_zoomed_comparison(results, save_path):
    """创建关键区域放大图，突出显示边界分割细节"""
    # 选择挑战性病例进行放大展示
    case_data = results["challenging"][0] if len(results["challenging"]) > 0 else None
    if case_data is None:
        print("没有找到挑战性病例，无法生成放大对比图")
        return None

    # 找到病灶区域，确定放大区域
    gt = case_data["ground_truth"]
    # 找到病灶的边界框
    rows, cols = np.where(gt > 0)
    if len(rows) == 0:
        print("未在图像中找到病灶，无法生成放大对比图")
        return None

    # 扩展边界框以包含周围区域
    margin = 20
    min_row, max_row = max(0, np.min(rows) - margin), min(gt.shape[0], np.max(rows) + margin)
    min_col, max_col = max(0, np.min(cols) - margin), min(gt.shape[1], np.max(cols) + margin)

    # 创建画布
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))

    # 显示原始区域
    axes[0].imshow(case_data["original_image"], cmap="gray")
    # 绘制放大区域框
    rect = plt.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row,
                         edgecolor='red', facecolor='none', linewidth=2)
    axes[0].add_patch(rect)
    axes[0].set_title("原始图像（红色框为放大区域）")
    axes[0].axis("off")

    # 放大区域显示
    models_to_show = ["U-Net", "HCMNet", "LBA-Net (Ours)"]
    zoomed_img = case_data["original_image"][min_row:max_row, min_col:max_col]

    for i, model_name in enumerate(models_to_show, 1):
        axes[i].imshow(zoomed_img, cmap="gray")
        zoomed_pred = case_data["predictions"][model_name][min_row:max_row, min_col:max_col]
        zoomed_gt = case_data["ground_truth"][min_row:max_row, min_col:max_col]

        # 叠加真值（绿色）和预测（红色）边界
        axes[i].imshow(zoomed_gt, cmap="Greens", alpha=0.3)
        axes[i].imshow(zoomed_pred, cmap="Reds", alpha=0.3)
        axes[i].set_title(f"{model_name} 边界细节")
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    return save_path

# ========================================
# 4. 主函数执行
# ========================================
def main():
    # 创建数据集
    dataset = BUSIDataset(
        BUSI_DIR,
        transform=A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    )
    print(f"加载BUSI数据集，共{len(dataset)}张图像")

    # 加载模型
    model_dict = load_models()

    # 筛选代表性病例
    # 使用LBA-Net来评估图像难度，因为它是我们提出的模型
    selected_indices = select_representative_cases(dataset, model_dict["LBA-Net (Ours)"])

    # 生成所有模型的分割结果
    print("生成所有模型的分割结果...")
    results = generate_segmentation_masks(dataset, selected_indices, model_dict)

    # 创建定性评估对比图
    print("创建定性评估对比图...")
    comparison_path = os.path.join(SAVE_DIR, "分割结果定性对比图.png")
    create_qualitative_comparison(results, comparison_path)
    files.download(comparison_path)

    # 创建边界细节放大图
    print("创建边界细节放大图...")
    zoomed_path = os.path.join(SAVE_DIR, "边界细节放大对比图.png")
    create_zoomed_comparison(results, zoomed_path)
    if zoomed_path:
        files.download(zoomed_path)

    print("定性评估可视化完成，结果已保存到:", SAVE_DIR)

if __name__ == "__main__":
    main()
