import torch
import torch.nn.functional as F


def dice_loss(logits, target, smooth=1e-6):
    prob=torch.sigmoid(logits)
    prob=prob.flatten(1)
    target=target.float().flatten(1)
    inter=(prob*target).sum(1)
    dice=(2*inter+smooth)/(prob.sum(1)+target.sum(1)+smooth)
    return 1-dice.mean()


def segmentation_loss(logits,target,bce_weight=0.5,dice_weight=0.5):
    bce=F.binary_cross_entropy_with_logits(logits,target.float())
    d=dice_loss(logits,target)
    return bce_weight*bce+dice_weight*d
