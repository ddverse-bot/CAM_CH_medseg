import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet import DoubleConv


class AttentionGate(nn.Module):
    def __init__(self, gate_channels, skip_channels, inter_channels):
        super().__init__()
        self.g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        self.x = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):
        g = self.g(gate)
        x = self.x(skip)
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        a = self.psi(self.relu(g+x))
        return skip*a


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=(64,128,256,512)):
        super().__init__()
        f1,f2,f3,f4 = features
        self.pool=nn.MaxPool2d(2)

        self.e1=DoubleConv(in_channels,f1)
        self.e2=DoubleConv(f1,f2)
        self.e3=DoubleConv(f2,f3)
        self.e4=DoubleConv(f3,f4)
        self.bottleneck=DoubleConv(f4,f4*2)

        self.u4=nn.ConvTranspose2d(f4*2,f4,2,2)
        self.a4=AttentionGate(f4,f4,f4//2)
        self.d4=DoubleConv(f4*2,f4)

        self.u3=nn.ConvTranspose2d(f4,f3,2,2)
        self.a3=AttentionGate(f3,f3,f3//2)
        self.d3=DoubleConv(f3*2,f3)

        self.u2=nn.ConvTranspose2d(f3,f2,2,2)
        self.a2=AttentionGate(f2,f2,f2//2)
        self.d2=DoubleConv(f2*2,f2)

        self.u1=nn.ConvTranspose2d(f2,f1,2,2)
        self.a1=AttentionGate(f1,f1,f1//2)
        self.d1=DoubleConv(f1*2,f1)

        self.final=nn.Conv2d(f1,out_channels,1)

    def forward(self,x,return_features=False):
        e1=self.e1(x)
        e2=self.e2(self.pool(e1))
        e3=self.e3(self.pool(e2))
        e4=self.e4(self.pool(e3))
        b=self.bottleneck(self.pool(e4))

        d4=self.u4(b); d4=self.d4(torch.cat([d4,self.a4(d4,e4)],1))
        d3=self.u3(d4); d3=self.d3(torch.cat([d3,self.a3(d3,e3)],1))
        d2=self.u2(d3); d2=self.d2(torch.cat([d2,self.a2(d2,e2)],1))
        d1=self.u1(d2); d1=self.d1(torch.cat([d1,self.a1(d1,e1)],1))

        out=self.final(d1)
        if return_features:
            return out,b
        return out
