
import argparse, os, random, numpy as np, torch
from torch.utils.data import DataLoader
from models.unet import UNet
from models.unetplusplus import UNetPlusPlus
from models.attunet import AttentionUNet
from models.classification_head import ClassificationHead
from losses.total_loss import total_loss
from datasets import make_datasets

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def build_model(name):
    return {"unet":UNet,"unet++":UNetPlusPlus,"attunet":AttentionUNet}[name]()

@torch.no_grad()
def dice_score(logits,target):
    p=(torch.sigmoid(logits)>0.5).float()
    inter=(p*target).sum((1,2,3))
    den=p.sum((1,2,3))+target.sum((1,2,3))
    return ((2*inter+1e-6)/(den+1e-6)).mean().item()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",choices=["busi","acdc","pulmonary","covid"],required=True)
    ap.add_argument("--data-root",required=True)
    ap.add_argument("--model",choices=["unet","unet++","attunet"],default="unet")
    ap.add_argument("--experiment",choices=["baseline","ch","ch_cam","ch_ace","full"],default="baseline")
    ap.add_argument("--epochs",type=int,default=100)
    ap.add_argument("--batch-size",type=int,default=8)
    ap.add_argument("--lr",type=float,default=1e-4)
    ap.add_argument("--image-size",type=int,default=256)
    ap.add_argument("--lambda-cls",type=float,default=0.2)
    ap.add_argument("--lambda-cam",type=float,default=0.3)
    ap.add_argument("--lambda-ace",type=float,default=0.5)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save-dir",default="runs")
    args=ap.parse_args()

    seed_everything(args.seed)
    device=torch.device(args.device)

    tr,va,te=make_datasets(args.dataset,args.data_root,image_size=args.image_size,seed=args.seed)
    train_loader=DataLoader(tr,args.batch_size,shuffle=True,num_workers=2,pin_memory=True)
    val_loader=DataLoader(va,args.batch_size,shuffle=False,num_workers=2,pin_memory=True)

    model=build_model(args.model).to(device)
    use_cls=args.experiment in {"ch","ch_cam","ch_ace","full"}
    use_cam=args.experiment in {"ch_cam","full"}
    use_ace=args.experiment in {"ch_ace","full"}
    head=ClassificationHead(1024 if args.model=="unet" else (512 if args.model=="unet++" else 1024)).to(device) if use_cls else None

    params=list(model.parameters())+(list(head.parameters()) if head else [])
    opt=torch.optim.Adam(params,lr=args.lr,weight_decay=1e-5)

    os.makedirs(args.save_dir,exist_ok=True)
    best=-1
    for epoch in range(1,args.epochs+1):
        model.train()
        if head: head.train()
        running=0.0
        for batch in train_loader:
            x=batch["image"].to(device); y=batch["mask"].to(device)
            opt.zero_grad()
            seg,feat=model(x,return_features=True)
            cls=head(feat) if head else None
            cls_target=(y.flatten(1).sum(1)>0).float()
            classifier_weight=head.net[-1].weight[0] if head else None
            loss,parts,_=total_loss(
                seg,y,cls,cls_target,feat,classifier_weight,
                args.lambda_cls,args.lambda_cam,args.lambda_ace,
                use_cls,use_cam,use_ace)
            loss.backward(); opt.step()
            running+=loss.item()

        score=0.0; model.eval()
        for batch in val_loader:
            x=batch["image"].to(device); y=batch["mask"].to(device)
            score+=dice_score(model(x),y)
        score/=max(len(val_loader),1)
        print(f"Epoch {epoch:03d} loss={running/max(len(train_loader),1):.5f} val_dice={score:.5f}")

        if score>best:
            best=score
            ck={"model":model.state_dict(),"head":head.state_dict() if head else None,
                "epoch":epoch,"val_dice":score,"dataset":args.dataset,"model_name":args.model}
            torch.save(ck,os.path.join(args.save_dir,f"{args.dataset}_{args.model}_{args.experiment}_best.pt"))

if __name__=="__main__":
    main()
