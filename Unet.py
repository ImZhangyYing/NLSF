import torch
import torch.nn as nn

import torch
import torch.nn as nn
from typing import Union
from torch.nn.modules.utils import _triple





class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool3d(2, 2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.sigmoid(x)
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch,out_ch):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch+skip_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels):
        super(UNet, self).__init__()
        features = [32,64,128,256]

        self.inc = InConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        self.down4 = Down(features[3], features[3])

        self.up1 = Up(features[3], features[3], features[2])
        self.up2 = Up(features[2], features[2], features[1])
        self.up3 = Up(features[1], features[1], features[0])
        self.up4 = Up(features[0], features[0], features[0])
        self.outc = OutConv(features[0], 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        xl = self.up4(x, x1)
        y = self.outc(xl)
        return x5, xl, y

# 定义简单的 Conv3d 测试
# class TestModel(nn.Module):
#     def __init__(self):
#         super(TestModel, self).__init__()
#         self.conv = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=1)
#
#     def forward(self, x):
#         return self.conv(x)



if __name__ == '__main__':

    # model = TestModel()
    # input_tensor = torch.randn(1, 1, 10, 10, 10)  # 示例输入
    # output = model(input_tensor)
    # print(output.shape)
    net = UNet(4)


    x = torch.randn(1, 4, 160, 160, 128)
    # y = net(x)
    print("params: ", sum(p.numel() for p in net.parameters()))
    # print(y.shape)