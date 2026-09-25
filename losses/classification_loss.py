import torch.nn.functional as F


def classification_loss(logits, target):
    return F.binary_cross_entropy_with_logits(
        logits.view(-1), target.float().view(-1)
    )
