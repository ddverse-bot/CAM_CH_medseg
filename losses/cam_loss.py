import torch
import torch.nn.functional as F


def normalized_cam(features, classifier_weight):
    # features: B,C,H,W; classifier_weight: C
    cam=(features*classifier_weight.view(1,-1,1,1)).sum(1,keepdim=True)
    cam=F.relu(cam)
    b=cam.shape[0]
    flat=cam.flatten(1)
    mn=flat.min(1)[0].view(b,1,1,1)
    mx=flat.max(1)[0].view(b,1,1,1)
    return (cam-mn)/(mx-mn+1e-6)


def cam_loss(features, classifier_weight, target):
    cam=normalized_cam(features,classifier_weight)
    cam=F.interpolate(cam,size=target.shape[-2:],mode="bilinear",align_corners=False)
    return F.mse_loss(cam,target.float()),cam
