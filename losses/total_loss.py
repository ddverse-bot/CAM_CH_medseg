from .segmentation_loss import segmentation_loss
from .classification_loss import classification_loss
from .cam_loss import cam_loss
from .ace_loss import ace_loss


def total_loss(seg_logits, target, cls_logits=None, cls_target=None,
               features=None, classifier_weight=None,
               lambda_cls=0.2, lambda_cam=0.3, lambda_ace=0.5,
               use_cls=False, use_cam=False, use_ace=False):

    loss=segmentation_loss(seg_logits,target)
    parts={"seg":float(loss.detach())}

    if use_cls:
        lc=classification_loss(cls_logits,cls_target)
        loss=loss+lambda_cls*lc
        parts["cls"]=float(lc.detach())

    if use_cam:
        lcam,cam=cam_loss(features,classifier_weight,target)
        loss=loss+lambda_cam*lcam
        parts["cam"]=float(lcam.detach())
    else:
        cam=None

    if use_ace:
        la=ace_loss(seg_logits,target)
        loss=loss+lambda_ace*la
        parts["ace"]=float(la.detach())

    parts["total"]=float(loss.detach())
    return loss,parts,cam
