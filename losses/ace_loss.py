import torch


def ace_loss(logits, target, n_bins=10, eps=1e-6):
    # Soft uncertainty/error calibration approximation.
    p=torch.sigmoid(logits)
    u=4.0*p*(1.0-p)
    error=(p-target.float()).abs()
    total=p.new_tensor(0.0)
    valid=0
    for k in range(n_bins):
        lo=k/n_bins
        hi=(k+1)/n_bins
        center=(lo+hi)/2
        w=(1-(u-center).abs()/(1/n_bins)).clamp(min=0)
        mass=w.sum()
        if mass>eps:
            total=total+(w*error).sum()/mass
            valid+=1
    return total/max(valid,1)
