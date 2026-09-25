import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskPriorAE(nn.Module):
    def __init__(self,latent_dim=128):
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Conv2d(1,32,4,2,1),nn.ReLU(),
            nn.Conv2d(32,64,4,2,1),nn.ReLU(),
            nn.Conv2d(64,128,4,2,1),nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc=nn.Linear(128,latent_dim)
        self.dec_fc=nn.Linear(latent_dim,128)
        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(128,64,4,2,1),nn.ReLU(),
            nn.ConvTranspose2d(64,32,4,2,1),nn.ReLU(),
            nn.ConvTranspose2d(32,1,4,2,1),nn.Sigmoid()
        )

    def encode(self,x):
        return self.fc(self.encoder(x).flatten(1))

    def forward(self,x):
        z=self.encode(x)
        h=self.dec_fc(z).view(x.size(0),128,1,1)
        y=self.decoder(h)
        return F.interpolate(y,size=x.shape[-2:],mode="bilinear",align_corners=False),z


def prior_uncertainty(pred_mask,prior_bank):
    if len(prior_bank)==0:
        return pred_mask.new_zeros(pred_mask.shape)
    z=pred_mask
    # fallback pixel-space distance to nearest stored anatomical prior
    d=[(z-p).abs().mean(dim=1,keepdim=True) for p in prior_bank]
    return torch.stack(d).min(0).values
