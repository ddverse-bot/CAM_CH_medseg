import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet import DoubleConv


class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, filters=(32,64,128,256,512)):
        super().__init__()
        f0,f1,f2,f3,f4 = filters
        self.pool = nn.MaxPool2d(2)
        self.up = lambda x: F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        self.x00 = DoubleConv(in_channels, f0)
        self.x10 = DoubleConv(f0, f1)
        self.x20 = DoubleConv(f1, f2)
        self.x30 = DoubleConv(f2, f3)
        self.x40 = DoubleConv(f3, f4)

        self.x01 = DoubleConv(f0+f1, f0)
        self.x11 = DoubleConv(f1+f2, f1)
        self.x21 = DoubleConv(f2+f3, f2)
        self.x31 = DoubleConv(f3+f4, f3)

        self.x02 = DoubleConv(f0*2+f1, f0)
        self.x12 = DoubleConv(f1*2+f2, f1)
        self.x22 = DoubleConv(f2*2+f3, f2)

        self.x03 = DoubleConv(f0*3+f1, f0)
        self.x13 = DoubleConv(f1*3+f2, f1)
        self.x04 = DoubleConv(f0*4+f1, f0)

        self.final = nn.Conv2d(f0, out_channels, 1)

    def forward(self, x, return_features=False):
        x00 = self.x00(x)
        x10 = self.x10(self.pool(x00))
        x20 = self.x20(self.pool(x10))
        x30 = self.x30(self.pool(x20))
        x40 = self.x40(self.pool(x30))

        x01 = self.x01(torch.cat([x00, self.up(x10)], 1))
        x11 = self.x11(torch.cat([x10, self.up(x20)], 1))
        x21 = self.x21(torch.cat([x20, self.up(x30)], 1))
        x31 = self.x31(torch.cat([x30, self.up(x40)], 1))

        x02 = self.x02(torch.cat([x00, x01, self.up(x11)], 1))
        x12 = self.x12(torch.cat([x10, x11, self.up(x21)], 1))
        x22 = self.x22(torch.cat([x20, x21, self.up(x31)], 1))

        x03 = self.x03(torch.cat([x00, x01, x02, self.up(x12)], 1))
        x13 = self.x13(torch.cat([x10, x11, x12, self.up(x22)], 1))

        x04 = self.x04(torch.cat([x00, x01, x02, x03, self.up(x13)], 1))

        out = self.final(x04)
        if return_features:
            return out, x40
        return out
