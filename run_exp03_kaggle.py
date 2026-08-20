"""Experiment 3: restoration test for existing M_global and B checkpoints only."""
from __future__ import annotations
import argparse, copy, csv, json, logging, random
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import matplotlib.pyplot as plt

class Model(nn.Module):
    def __init__(self):
        super().__init__(); self.backbone=models.resnet50(weights=None); self.backbone.fc=nn.Identity()
        self.head=nn.Sequential(nn.Linear(2048,512),nn.ReLU(),nn.Dropout(.5),nn.Linear(512,2))
    def forward(self,x): return self.head(self.backbone(x))

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def load(path, device):
    payload=torch.load(path,map_location="cpu",weights_only=False); model=Model()
    model.load_state_dict(payload["model_state_dict"],strict=True); return model.to(device)

def metadata(root):
    f=pd.read_csv(root/"metadata.csv").rename(columns={"img_path":"image","img_filename":"image","waterbird":"y","water_background":"place"})
    if f.split.dtype==object: f["split"]=f.split.map({"train":0,"val":1,"test":2})
    required={"image","y","place","split"}
    if not required <= set(f): raise ValueError(f"metadata missing {required-set(f)}")
    f["path"]=f.image.map(lambda x: str(x) if Path(str(x)).is_absolute() else str(root/x))
    return f

TF=transforms.Compose([transforms.Resize((160,160)),transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
class Records(Dataset):
    def __init__(self, rows): self.rows=rows
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        with Image.open(r["path"]) as im: return TF(im.convert("RGB")),int(r["y"])
class Pairs(Dataset):
    def __init__(self, frame, seed=0):
        self.items=[]; rng=np.random.default_rng(seed)
        # Fixed, label-conditioned proxy pairs.  The original Waterbirds data has
        # no identity-preserving background edits; this is reproducible but not a
        # claim of same-subject counterfactual generation.
        for y,first,second in ((1,1,0),(0,0,1)):
            a=frame[(frame.y==y)&(frame.place==first)].sort_values("path").to_dict("records")
            b=frame[(frame.y==y)&(frame.place==second)].sort_values("path").to_dict("records")
            n=min(len(a),len(b),200)
            if not n: raise ValueError("test split lacks a required label/background group")
            ia=rng.choice(len(a),n,replace=False); ib=rng.choice(len(b),n,replace=False)
            # Each pair is returned water-background first, then land-background.
            for x,z in zip(ia,ib): self.items.append((a[x] if first==1 else b[z], b[z] if first==1 else a[x], y))
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        water,land,y=self.items[i]; out=[]
        for r in (water,land):
            with Image.open(r["path"]) as im: out.append(TF(im.convert("RGB")))
        return out[0],out[1],y

def backbone_check(a,b):
    diffs=[]
    for (n,p),(m,q) in zip(a.backbone.named_parameters(),b.backbone.named_parameters()):
        if n!=m or p.shape!=q.shape: raise RuntimeError(f"backbone mismatch: {n}/{m}")
        diffs.append(float((p.detach().cpu()-q.detach().cpu()).abs().max()))
    d=max(diffs); return {"all_backbone_parameters_equal":d==0.0,"max_parameter_difference":d}

@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval(); wp=[]; lp=[]; ys=[]
    for w,l,y in loader:
        wp.append(model(w.to(device)).softmax(1).cpu()); lp.append(model(l.to(device)).softmax(1).cpu()); ys.append(y)
    w,l,y=torch.cat(wp),torch.cat(lp),torch.cat(ys); wb=y==1; lb=y==0
    dw=(w[wb,1]-l[wb,1]).mean().item(); dl=(l[lb,0]-w[lb,0]).mean().item(); pw,pl=w.argmax(1),l.argmax(1)
    acc=torch.cat((pw==y,pl==y)).float().mean().item()
    group=[(pw[wb]==1).float().mean(),(pl[wb]==1).float().mean(),(pl[lb]==0).float().mean(),(pw[lb]==0).float().mean()]
    return {"delta_waterbird":dw,"delta_landbird":dl,"background_gap":(abs(dw)+abs(dl))/2,
            "original_flip_rate":(pw[wb]!=pl[wb]).float().mean().item(),"reverse_flip_rate":(pw[lb]!=pl[lb]).float().mean().item(),
            "overall_accuracy":acc,"worst_group_accuracy":min(x.item() for x in group)}

def train_head(model, loader, opt, device):
    model.eval(); model.head.train(); loss=nn.CrossEntropyLoss()
    for x,y in loader:
        opt.zero_grad(set_to_none=True); z=loss(model(x.to(device)),y.to(device)); z.backward(); opt.step()

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data-dir",type=Path,required=True); p.add_argument("--checkpoint-global",type=Path,required=True); p.add_argument("--checkpoint-b",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--device",default="cuda"); p.add_argument("--seed",type=int,default=42); p.add_argument("--epochs",type=int,default=5); p.add_argument("--batch-size",type=int,default=128); p.add_argument("--num-workers",type=int,default=2); p.add_argument("--lr",type=float,default=1e-3); a=p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True); logging.basicConfig(level=logging.INFO,format="%(asctime)s %(message)s",handlers=[logging.FileHandler(a.output_dir/"run.log"),logging.StreamHandler()]); log=logging.getLogger("exp03")
    device=torch.device("cuda" if a.device=="cuda" and torch.cuda.is_available() else "cpu"); log.info("torch=%s cuda=%s device=%s",torch.__version__,torch.version.cuda,device)
    if device.type=="cuda": log.info("gpu=%s",torch.cuda.get_device_name(0))
    seed_all(a.seed); g,b=load(a.checkpoint_global,device),load(a.checkpoint_b,device); check=backbone_check(g,b); (a.output_dir/"backbone_check.json").write_text(json.dumps(check,indent=2)); log.info("backbone check=%s",check)
    # One initialization, copied bit-for-bit to both heads; no backbone buffer or parameter is trainable.
    fresh=Model().head; init=copy.deepcopy(fresh.state_dict())
    for model in (g,b):
        model.head=copy.deepcopy(fresh); model.head.load_state_dict(init)
        for q in model.backbone.parameters(): q.requires_grad_(False)
        for q in model.head.parameters(): q.requires_grad_(True)
    f=metadata(a.data_dir); train=f[f.split==0]; groups={(y,k):x for (y,k),x in train.groupby(["y","place"])}
    units=min(len(groups[1,1])//19,len(groups[0,0])//19,len(groups[1,0]),len(groups[0,1]));
    if units<1: raise ValueError("not enough samples for exact 95/5 restoration data")
    rng=np.random.default_rng(a.seed); rows=[]
    for key,n in [((1,1),19*units),((1,0),units),((0,0),19*units),((0,1),units)]: rows += groups[key].iloc[rng.permutation(len(groups[key]))[:n]].to_dict("records")
    rng.shuffle(rows) # one fixed order, reused without shuffle by both models in every epoch
    train_loader=DataLoader(Records(rows),batch_size=a.batch_size,shuffle=False,num_workers=a.num_workers,pin_memory=device.type=="cuda")
    audit_loader=DataLoader(Pairs(f[f.split==2]),batch_size=a.batch_size,shuffle=False,num_workers=a.num_workers,pin_memory=device.type=="cuda")
    opts=[torch.optim.SGD(m.head.parameters(),lr=a.lr,momentum=.9) for m in (g,b)]; history=[]
    for epoch in range(a.epochs+1):
        gm,bm=evaluate(g,audit_loader,device),evaluate(b,audit_loader,device); history.append({"epoch":epoch,"global_bg_gap":gm["background_gap"],"B_bg_gap":bm["background_gap"],"global_delta_WB":gm["delta_waterbird"],"B_delta_WB":bm["delta_waterbird"],"global_delta_LB":gm["delta_landbird"],"B_delta_LB":bm["delta_landbird"],**{f"global_{k}":v for k,v in gm.items()},**{f"B_{k}":v for k,v in bm.items()}})
        if epoch<a.epochs:
            # Reset RNG for each model so Dropout receives identical masks too.
            seed_all(a.seed + epoch)
            train_head(g,train_loader,opts[0],device)
            seed_all(a.seed + epoch)
            train_head(b,train_loader,opts[1],device)
    with (a.output_dir/"restoration_metrics.csv").open("w",newline="") as h: w=csv.DictWriter(h,fieldnames=list(history[0])); w.writeheader(); w.writerows(history)
    diffs=[abs(r["global_bg_gap"]-r["B_bg_gap"]) for r in history]; conclusion=("Balanced Head Retraining only changes the current decision head; it does not remove recoverable background-label relation information from the backbone. B is a relation suppression baseline, not representation-level relation unlearning." if check["all_backbone_parameters_equal"] and max(diffs)<1e-4 else "Restoration curves differ beyond tolerance. Check head initialization, batch order, optimizer, backbone freezing, and checkpoint loading before making a scientific claim.")
    plt.plot([r["epoch"] for r in history],[r["global_bg_gap"] for r in history],marker="o",label="M_global"); plt.plot([r["epoch"] for r in history],[r["B_bg_gap"] for r in history],marker="o",label="B"); plt.xlabel("Epoch"); plt.ylabel("BG Gap"); plt.legend(); plt.grid(); plt.savefig(a.output_dir/"restoration_curve.png",dpi=160,bbox_inches="tight"); plt.close()
    summary={"backbone_check":check,"max_BGGap_difference_over_epochs":max(diffs),"mean_BGGap_difference_over_epochs":float(np.mean(diffs)),"tolerance":1e-4,"conclusion":conclusion}; (a.output_dir/"summary.json").write_text(json.dumps(summary,indent=2))
    print(f"Backbone equal: {check['all_backbone_parameters_equal']}\nMax backbone parameter difference: {check['max_parameter_difference']}\nEpoch 0 BGGap: M_global = {history[0]['global_bg_gap']:.6f}, B = {history[0]['B_bg_gap']:.6f}\nEpoch 5 BGGap: M_global = {history[-1]['global_bg_gap']:.6f}, B = {history[-1]['B_bg_gap']:.6f}\nMax restoration curve difference = {max(diffs):.6g}\nConclusion:\n{conclusion}")
if __name__=="__main__": main()
