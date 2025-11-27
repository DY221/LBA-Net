# Code of the LBA_Net Lightweight Boundary-Aware Network for Robust Breast Ultrasound Image Segmentation
## Introduction
Breast ultrasound (BUS) segmentation is challenged by strong noise, low contrast, and ambiguous lesion boundaries. Although deep models achieve high accuracy, their heavy computational cost limits deployment on portable ultrasound devices. In contrast, lightweight networks often struggle to preserve fine boundary details. To address this gap, we propose Lightweight Boundary-Aware Network (LBA-Net). A MobileNetV3-based encoder with Atrous Spatial Pyramid Pooling (ASPP) is integrated for efficient multi-scale representation learning. The applied Lightweight Boundary-Aware Block (LBA-Block) uses an adaptive fusion to combine efficient channel attention and depthwise spatial attention to enhance discriminative capability with minimal computational overhead. A boundary-guided dual-head decoding scheme injects explicit boundary priors and enforces boundary consistency to sharpen and stabilize margin delineation. Experiments on curated BUSI* and BUET* datasets demonstrate that LBA-Net achieves 82.8% Dice, 38 px HD95, and real-time inference speeds (123 FPS GPU / 19 FPS CPU) using only 1.76M parameters. They show that LBA-Net can offer a highly favorable balance between accuracy and efficiency.
# Architecture
![结构1]![结构1](https://github.com/user-attachments/assets/8a7f2905-0d9b-40f8-9bc9-e1f5c8582920)



## Requirements
python3.7-3.9 pytorch >= 1.5 torchvision >= 0.6.1 cuda >= 10.1
