
import os
from .base import list_files, SegmentationDataset, split_by_group

def find_busi_pairs(root):
    # Official BUSI layout:
    # Dataset_BUSI_with_GT/{benign,malignant,normal}/
    # class image: benign (1).png
    # mask:        benign (1)_mask.png
    pairs=[]
    for cls in ["benign","malignant","normal"]:
        d=os.path.join(root,cls)
        if not os.path.isdir(d):
            continue
        files=list_files(d)
        masks={os.path.basename(x).lower():x for x in files if "_mask" in os.path.basename(x).lower()}
        for ip in files:
            name=os.path.basename(ip)
            if "_mask" in name.lower():
                continue
            stem,ext=os.path.splitext(name)
            mp=masks.get((stem+"_mask"+ext).lower())
            if mp:
                pairs.append((ip,mp,cls))
    if not pairs:
        raise RuntimeError(
            "BUSI pairs not found. Expected Dataset_BUSI_with_GT/benign, "
            "malignant, normal with '<name>.png' and '<name>_mask.png'."
        )
    return pairs

def make_datasets(root,image_size=256,seed=42):
    pairs=find_busi_pairs(root)
    # Patient/image-level split. Do not split an image and its mask independently.
    tr,va,te=split_by_group(pairs,seed=seed,group_fn=lambda p:p[0])
    return (SegmentationDataset([(a,b) for a,b,_ in tr],image_size),
            SegmentationDataset([(a,b) for a,b,_ in va],image_size),
            SegmentationDataset([(a,b) for a,b,_ in te],image_size))
