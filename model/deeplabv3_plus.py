import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv6 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.conv12 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
        self.conv18 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False)
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )
        self.conv_final = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x1 = F.relu(self.bn(self.conv1(x)))
        x6 = F.relu(self.bn(self.conv6(x)))
        x12 = F.relu(self.bn(self.conv12(x)))
        x18 = F.relu(self.bn(self.conv18(x)))
        x_global = F.interpolate(self.global_avg_pool(x), size=x.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x1, x6, x12, x18, x_global], dim=1)
        x = F.relu(self.bn(self.conv_final(x)))
        return x


class DeepLabv3Plus(nn.Module):
    def __init__(self, num_classes=1):
        super(DeepLabv3Plus, self).__init__()
        self.backbone = resnet50(pretrained=True, replace_stride_with_dilation=[False, True, True])
        self.aspp = ASPP(2048, 256)
        self.low_level_conv = nn.Conv2d(256, 48, 1, bias=False)
        self.final_conv = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x):
        # Get low level features
        low_level = self.backbone.layer1(x)
        # Get high level features
        x = self.backbone.layer2(self.backbone.layer1(x))
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.aspp(x)

        # Upsample low level features
        low_level = self.low_level_conv(low_level)
        x = F.interpolate(x, size=low_level.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, low_level], dim=1)
        x = self.final_conv(x)
        return x