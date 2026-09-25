import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, in_channels, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.net=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,1)
        )

    def forward(self,x):
        return self.net(x)
