# models/local_net.py

import torch
import torch.nn as nn
from models.cbam import CBAM

class LocalNet(nn.Module):
    def __init__(self, in_channels=6):
        super(LocalNet, self).__init__()
        # 與 GlobalNet 相同結構
        self.conv1_1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.cbam1_1 = CBAM(64)

        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.cbam1_2 = CBAM(64)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.cbam2_1 = CBAM(128)

        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.cbam2_2 = CBAM(128)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.cbam3_1 = CBAM(256)

        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.cbam3_2 = CBAM(256)

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.cbam4_1 = CBAM(512)

        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.cbam4_2 = CBAM(512)

        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # block1
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.cbam1_1(x)

        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.cbam1_2(x)

        x = self.pool1(x)

        # block2
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.cbam2_1(x)

        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.cbam2_2(x)

        x = self.pool2(x)

        # block3
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.cbam3_1(x)

        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.cbam3_2(x)

        x = self.pool3(x)

        # block4
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.cbam4_1(x)

        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.cbam4_2(x)

        x = self.pool4(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return x
