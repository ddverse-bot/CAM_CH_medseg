
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
from .base import normalize_array

class COVIDNpyDataset(Dataset):
    """
    Exact format of the Kaggle COVID-19 CT Images Segmentation competition:
      images_medseg.npy       -> (100,512,512) axial slices
      masks_medseg.npy        -> (100,512,512,4), channels:
                                  0 ground glass
                                  1 consolidations
                                  2 lungs other
                                  3 background
      images_radiopedia.npy   -> (829,512,512)
      masks_radiopedia.npy    -> (829,512,512,4)
      test_images_medseg.npy  -> (10,512,512)

    For binary infection segmentation, channels 0+1+2 are foreground.
    Background channel 3 is excluded.
    """
    def __init__(self,images,masks,image_size=256):
        self.images=np.asarray(images)
        self.masks=np.asarray(masks)
        assert len(self.images)==len(self.masks)
        self.image_size=image_size

    def __len__(self): return len(self.images)

    def __getitem__(self,i):
        x=normalize_array(self.images[i])
        m=self.masks[i]
        if m.ndim==3 and m.shape[-1]==4:
            y=(m[...,:3].sum(axis=-1)>0).astype(np.uint8)
        else:
            y=(m>0).astype(np.uint8)
        x=Image.fromarray((x*255).astype(np.uint8))
        y=Image.fromarray(y*255)
        x=TF.resize(x,[self.image_size,self.image_size])
        y=TF.resize(y,[self.image_size,self.image_size],
                     interpolation=TF.InterpolationMode.NEAREST)
        return {"image":TF.to_tensor(x),"mask":(TF.to_tensor(y)>0.5).float(),"index":i}

def load_covid_arrays(root, part="combined"):
    def load(n): return np.load(os.path.join(root,n))
    ims=[]; mks=[]
    if part in ("medseg","combined"):
        ims.append(load("images_medseg.npy")); mks.append(load("masks_medseg.npy"))
    if part in ("radiopedia","combined"):
        ims.append(load("images_radiopedia.npy")); mks.append(load("masks_radiopedia.npy"))
    if not ims:
        raise ValueError(part)
    return np.concatenate(ims),np.concatenate(mks)

def make_datasets(root,image_size=256,seed=42,part="combined"):
    from torch.utils.data import Subset
    images,masks=load_covid_arrays(root,part)
    n=len(images)
    rng=np.random.default_rng(seed)
    idx=rng.permutation(n)
    a=int(.8*n); b=int(.9*n)
    ds=COVIDNpyDataset(images,masks,image_size)
    return Subset(ds,idx[:a]),Subset(ds,idx[a:b]),Subset(ds,idx[b:])
