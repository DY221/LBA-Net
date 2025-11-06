import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResUnet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(ResUnet, self).__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResBlock(64, 64)
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            ResBlock(64, 128),
            ResBlock(128, 128)
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            ResBlock(128, 256),
            ResBlock(256, 256)
        )
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = nn.Sequential(
            ResBlock(256, 512),
            ResBlock(512, 512)
        )
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResBlock(512, 1024),
            ResBlock(1024, 1024)
        )

        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec1 = nn.Sequential(
            ResBlock(1024, 512),
            ResBlock(512, 512)
        )

        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec2 = nn.Sequential(
            ResBlock(512, 256),
            ResBlock(256, 256)
        )

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = nn.Sequential(
            ResBlock(256, 128),
            ResBlock(128, 128)
        )

        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec4 = nn.Sequential(
            ResBlock(128, 64),
            ResBlock(64, 64)
        )

        self.final_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.pool1(e1)
        e2 = self.enc2(e2)

        e3 = self.pool2(e2)
        e3 = self.enc3(e3)

        e4 = self.pool3(e3)
        e4 = self.enc4(e4)

        # Bottleneck
        b = self.pool4(e4)
        b = self.bottleneck(b)

        # Decoder
        d1 = self.up1(b)
        d1 = torch.cat([d1, e4], dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2, e3], dim=1)
        d2 = self.dec2(d2)

        d3 = self.up3(d2)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)

        d4 = self.up4(d3)
        d4 = torch.cat([d4, e1], dim=1)
        d4 = self.dec4(d4)

        return self.final_conv(d4)