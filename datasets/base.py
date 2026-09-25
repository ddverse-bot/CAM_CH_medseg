
import os
import random
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

IMG_EXT={".png",".jpg",".jpeg",".bmp",".tif",".tiff",".npy"}

def list_files(root, exts=None):
    exts = exts or IMG_EXT
    out=[]
    for dp,_,fs in os.walk(root):
        for f in fs:
            if os.path.splitext(f)[1].lower() in exts:
                out.append(os.path.join(dp,f))
    return sorted(out)

def patient_key(path):
    b=os.path.basename(path)
    s=os.path.splitext(b)[0]
    s=s.replace("_mask","").replace("_seg","").replace("_label","")
    # BUSI/PU: file stem is already a patient/image identifier.
    return s

def split_by_group(items, train=0.8, val=0.1, seed=42, group_fn=lambda x:x):
    groups=defaultdict(list)
    for item in items:
        groups[group_fn(item)].append(item)
    keys=list(groups)
    random.Random(seed).shuffle(keys)
    n=len(keys)
    a=int(train*n); b=int((train+val)*n)
    def flatten(keys_):
        return [z for k in keys_ for z in groups[k]]
    return flatten(keys[:a]), flatten(keys[a:b]), flatten(keys[b:])

class SegmentationDataset(Dataset):
    def __init__(self,pairs,image_size=256, normalize=True):
        self.pairs=pairs
        self.image_size=image_size
        self.normalize=normalize

    def __len__(self): return len(self.pairs)

    def __getitem__(self,i):
        ip,mp,*rest=self.pairs[i]
        im=Image.open(ip).convert("L")
        mask=Image.open(mp).convert("L")
        im=TF.resize(im,[self.image_size,self.image_size])
        mask=TF.resize(mask,[self.image_size,self.image_size],
                       interpolation=TF.InterpolationMode.NEAREST)
        im=TF.to_tensor(im)
        mask=(TF.to_tensor(mask)>0.5).float()
        return {"image":im,"mask":mask,"path":ip}

def normalize_array(x):
    x=np.asarray(x,dtype=np.float32)
    lo=np.percentile(x,1)
    hi=np.percentile(x,99)
    x=np.clip(x,lo,hi)
    return (x-lo)/(hi-lo+1e-6)
