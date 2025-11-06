import torch
import torch.nn as nn
import torch.nn.functional as F


class U2NetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(U2NetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(x1)))
        x3 = self.relu(self.bn3(self.conv3(x2)))
        return x3


class U2Net(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(U2Net, self).__init__()
        # Encoder
        self.enc1 = U2NetBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = U2NetBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = U2NetBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = U2NetBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.enc5 = U2NetBlock(512, 512)

        # Decoder
        self.dec5 = U2NetBlock(1024, 512)
        self.dec4 = U2NetBlock(1024, 256)
        self.dec3 = U2NetBlock(512, 128)
        self.dec2 = U2NetBlock(256, 64)
        self.dec1 = U2NetBlock(128, 64)

        # Upsampling
        self.up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Final prediction
        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        e5 = self.enc5(self.pool4(e4))

        # Decoder
        d5 = self.dec5(torch.cat([self.up5(e5), e4], dim=1))
        d4 = self.dec4(torch.cat([self.up4(d5), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1))
        d1 = self.dec1(d2)

        return self.final(d1)