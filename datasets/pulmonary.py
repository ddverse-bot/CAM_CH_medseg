
import os
import pandas as pd
from .base import list_files, SegmentationDataset, split_by_group

def find_pu_pairs(root):
    """
    PU2756 Figshare format:
      images/ : 2,756 PNG B-mode ultrasound images, one per patient
      masks/  : corresponding PNG masks with the same patient/image identifier
      five_fold_split.xlsx : patient-level fold metadata
    """
    imgdir=os.path.join(root,"images")
    maskdir=os.path.join(root,"masks")
    if not os.path.isdir(imgdir) or not os.path.isdir(maskdir):
        raise RuntimeError("Expected PU2756 root/images and root/masks directories.")

    imgs=list_files(imgdir,{".png",".jpg",".jpeg"})
    masks=list_files(maskdir,{".png",".jpg",".jpeg"})
    mm={os.path.splitext(os.path.basename(x))[0]:x for x in masks}
    pairs=[]
    for ip in imgs:
        stem=os.path.splitext(os.path.basename(ip))[0]
        mp=mm.get(stem)
        if mp is None:
            # allow common annotation suffixes
            for s in ["_mask","_seg","_label"]:
                if stem+s in mm:
                    mp=mm[stem+s]; break
        if mp:
            pairs.append((ip,mp))
    if not pairs:
        raise RuntimeError("No PU2756 image/mask matches found.")
    return pairs

def make_datasets(root,image_size=256,seed=42):
    pairs=find_pu_pairs(root)
    # If official five_fold_split.xlsx exists, users can replace this with
    # the official fold assignment. Default remains patient/image-level 80/10/10.
    tr,va,te=split_by_group(pairs,seed=seed,group_fn=lambda p:p[0])
    return SegmentationDataset(tr,image_size),SegmentationDataset(va,image_size),SegmentationDataset(te,image_size)
