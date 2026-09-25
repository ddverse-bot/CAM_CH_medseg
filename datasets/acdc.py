
import os, re
import numpy as np
import nibabel as nib
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from .base import split_by_group, normalize_array

def read_info(path):
    info={}
    if not os.path.exists(path):
        return info
    with open(path,encoding="utf-8",errors="ignore") as f:
        for line in f:
            if ":" in line:
                k,v=line.strip().split(":",1)
                info[k.strip()]=v.strip()
    return info

def discover_patient_frames(root, include_training_only=True):
    base=os.path.join(root,"training") if os.path.isdir(os.path.join(root,"training")) else root
    patients=[]
    for name in sorted(os.listdir(base)):
        p=os.path.join(base,name)
        if not os.path.isdir(p) or not name.startswith("patient"):
            continue
        info=read_info(os.path.join(p,"Info.cfg"))
        frames=[]
        for phase in ["ED","ES"]:
            if phase not in info: continue
            num=int(info[phase])
            stem=f"{name}_frame{num:02d}"
            ip=os.path.join(p,stem+".nii.gz")
            mp=os.path.join(p,stem+"_gt.nii.gz")
            if os.path.exists(ip) and os.path.exists(mp):
                frames.append((ip,mp))
        if frames:
            patients.append((name,frames))
    return patients

class ACDCSliceDataset(Dataset):
    def __init__(self,items,image_size=256,target_labels=(1,2,3)):
        self.items=items
        self.image_size=image_size
        self.target_labels=set(target_labels)

    def __len__(self): return len(self.items)

    def __getitem__(self,i):
        ip,mp,slice_idx,patient,phase=self.items[i]
        vol=nib.load(ip).get_fdata().astype(np.float32)
        gt=nib.load(mp).get_fdata()
        x=normalize_array(vol[:,:,slice_idx])
        y=np.isin(gt[:,:,slice_idx],list(self.target_labels)).astype(np.uint8)
        x=Image.fromarray((x*255).astype(np.uint8))
        y=Image.fromarray(y*255)
        x=TF.resize(x,[self.image_size,self.image_size])
        y=TF.resize(y,[self.image_size,self.image_size],
                     interpolation=TF.InterpolationMode.NEAREST)
        return {"image":TF.to_tensor(x),"mask":(TF.to_tensor(y)>0.5).float(),
                "path":ip,"patient":patient,"phase":phase,"slice":slice_idx}

def make_datasets(root,image_size=256,seed=42,target_labels=(1,2,3)):
    patients=discover_patient_frames(root)
    if not patients:
        raise RuntimeError("ACDC training/patientXXX with frameXX and frameXX_gt files not found.")

    # Patient-level 80/10/10 split BEFORE expanding into slices.
    import random
    keys=[p[0] for p in patients]
    random.Random(seed).shuffle(keys)
    n=len(keys); a=int(.8*n); b=int(.9*n)
    split_names={"train":set(keys[:a]),"val":set(keys[a:b]),"test":set(keys[b:])}
    byname={p[0]:p[1] for p in patients}
    result=[]
    for split in ["train","val","test"]:
        items=[]
        for patient in sorted(split_names[split]):
            for ip,mp in byname[patient]:
                z=nib.load(ip).shape[2]
                phase="ED" if "ED" in "" else ("ED" if "frame"+re.search(r"frame(\d+)",os.path.basename(ip)).group(1) == f"frame{read_info(os.path.join(os.path.dirname(ip),'Info.cfg')).get('ED','-1'):02d}" else "ES")
                for s in range(z):
                    items.append((ip,mp,s,patient,phase))
        result.append(ACDCSliceDataset(items,image_size,target_labels))
    return tuple(result)
