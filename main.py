import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, Resize, ToTensor
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd

from models.efficientnet import get_efficientnet
from models.gat import GAT
from models.gru import GRU
from utils.dataset import DeepfakeDataset
from utils.losses import CombinedLoss

# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

EDGES_68 = [
    *[(i, i+1) for i in range(0,16)],
    *[(i, i+1) for i in range(17,21)],
    *[(i, i+1) for i in range(22,26)],
    *[(i, i+1) for i in range(27,30)],
    *[(i, i+1) for i in range(31,35)],
    *[(i, i+1) for i in range(36,41)] + [(41,36)],
    *[(i, i+1) for i in range(42,47)] + [(47,42)],
    *[(i, i+1) for i in range(48,59)] + [(59,48)],
    *[(i, i+1) for i in range(60,67)] + [(67,60)],
]
FACEMESH_EDGES = torch.tensor(EDGES_68, dtype=torch.long).t().contiguous()

def create_batched_edge_index(base_edge_index, batch_size, num_nodes, device):
    edge_index = base_edge_index.clone().to(device)
    edge_index = edge_index.repeat(1, batch_size)
    offsets = torch.arange(batch_size, device=device) * num_nodes
    offsets = offsets.unsqueeze(0).repeat(2, base_edge_index.size(1))
    edge_index += offsets
    return edge_index

class DeepfakeModel(nn.Module):
    def __init__(self, seq_len=40, dropout_rate=0.3, n_nodes=68):
        super().__init__()
        self.seq_len = seq_len
        self.n_nodes = n_nodes

        self.eff = get_efficientnet()
        self.proj = nn.Linear(1280, 256)
        self.lm_proj = nn.Linear(2, 64)
        self.gat = GAT(in_channels=256+64, out_channels=8, heads=1)
        self.gru = GRU(input_size=8, hidden_size=32, num_layers=1, dropout=dropout_rate)
        self.attn = nn.Linear(32, 1)
        self.fc   = nn.Linear(32, 1)

    def forward(self, x, lm):
        B, S, C, H, W = x.shape
        x = x.view(B*S, C, H, W)
        feats = self.eff(x).view(B, S, 1280)
        proj  = self.proj(feats)

        lm_flat = lm.view(B*S*self.n_nodes, 2)
        lm_feat = self.lm_proj(lm_flat)

        node_feats = proj.unsqueeze(2).repeat(1,1,self.n_nodes,1)
        node_feats = node_feats.view(B*S*self.n_nodes, 256)
        node_feats = torch.cat([node_feats, lm_feat], dim=1)

        batched_edges = create_batched_edge_index(FACEMESH_EDGES, B*S, self.n_nodes, device)
        gat_out = self.gat(node_feats, batched_edges)
        gat_out = gat_out.view(B, S, self.n_nodes, 8).mean(dim=2)

        gru_out, attn_w = self.gru(gat_out)
        scores  = self.attn(gru_out)
        weights = torch.softmax(scores, dim=1)
        pooled  = (gru_out * weights).sum(dim=1)
        out     = self.fc(pooled).view(-1)
        return out, attn_w

def compute_metrics(labels, preds, probs):
    accuracy = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, probs) if len(set(labels))>1 else 0.0
    f1  = f1_score(labels, preds)   if len(set(labels))>1 else 0.0

    TPf = np.sum((preds==1)&(labels==1))
    FNf = np.sum((preds==0)&(labels==1))
    FPf = np.sum((preds==1)&(labels==0))
    recall_fake   = TPf/(TPf+FNf) if (TPf+FNf)>0 else 0.0
    precision_fake= TPf/(TPf+FPf) if (TPf+FPf)>0 else 0.0

    TPr = np.sum((preds==0)&(labels==0))
    FNr = np.sum((preds==1)&(labels==0))
    FRR = FNr/(TPr+FNr)       if (TPr+FNr)>0 else 0.0
    GAR = 1 - FRR

    return accuracy, f1, auc, recall_fake, FRR, GAR, precision_fake

def compute_eer(labels, probs):
    fpr, tpr, _ = roc_curve(labels, probs)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return fpr[idx].item()

def train_epoch(model, loader, crit, opt, device, epoch):
    model.train()
    total_loss=0; all_l=[]; all_p=[]
    max_norm = min(5.0, 0.1*(epoch+1))
    for imgs, labs, lm in tqdm(loader, desc="Train"):
        imgs,labs,lm = imgs.to(device), labs.to(device), lm.to(device)
        opt.zero_grad()
        out,_ = model(imgs,lm)
        out = out.view(-1); labs = labs.float().view(-1)
        loss=crit(out,labs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        opt.step()

        total_loss+=loss.item()
        probs = torch.sigmoid(out).detach().cpu().numpy()
        all_l.extend(labs.cpu().numpy()); all_p.extend(probs)

    preds = (np.array(all_p)>0.5).astype(int)
    acc,f1,auc,rec,fr,gar,prec = compute_metrics(np.array(all_l),preds,np.array(all_p))
    eer = compute_eer(np.array(all_l),np.array(all_p))
    return total_loss/len(loader), acc, f1, auc, rec, fr, gar, prec, eer

def evaluate_model(model, loader, crit, device, epoch=None):
    model.eval()
    total_loss=0; all_l=[]; all_p=[]; all_att=[]
    with torch.no_grad():
        for imgs,labs,lm in tqdm(loader, desc="Eval"):
            imgs,labs,lm = imgs.to(device), labs.to(device), lm.to(device)
            out,att = model(imgs,lm)
            out = out.view(-1); labs=labs.float().view(-1)
            loss=crit(out,labs)
            total_loss+=loss.item()
            probs=torch.sigmoid(out).cpu().numpy()
            all_l.extend(labs.cpu().numpy()); all_p.extend(probs)
            if epoch and epoch%5==0:
                all_att.append(att.cpu().numpy())

    # save preds
    df=pd.DataFrame({'labels':all_l,'probs':all_p})
    os.makedirs("outputs/pred_data",exist_ok=True)
    if epoch: df.to_csv(f"outputs/pred_data/epoch_{epoch}.csv",index=False)

    # save attention
    if all_att:
        os.makedirs("outputs/attn",exist_ok=True)
        np.save(f"outputs/attn/epoch_{epoch}.npy",np.concatenate(all_att))

    preds=(np.array(all_p)>0.5).astype(int)
    acc,f1,auc,rec,fr,gar,prec=compute_metrics(np.array(all_l),preds,np.array(all_p))
    eer=compute_eer(np.array(all_l),np.array(all_p))
    return total_loss/len(loader), acc,f1,auc,rec,fr,gar,prec,eer

def save_model_and_result(model, res, mpath, rpath):
    os.makedirs(os.path.dirname(mpath),exist_ok=True)
    os.makedirs(os.path.dirname(rpath),exist_ok=True)
    torch.save(model.state_dict(), mpath)
    torch.save(model.state_dict(), mpath.replace(".pt",".pth"))
    with open(rpath,"w") as f:
        json.dump(res,f,indent=4)

def main():
    root="data/preprocessed/"; labels="data/List_of_testing_videos.txt"
    seq_len=40; bs=32; epochs=20

    tfm=Compose([Resize((224,224)),ToTensor()])
    ds=DeepfakeDataset(root,labels,transform=tfm,limit=3000,seq_len=seq_len)
    print(f"Dataset size: {len(ds)}")

    groups=[item[2] for item in ds.labels]
    gss=GroupShuffleSplit(n_splits=1,test_size=0.3,random_state=42)
    ti,vi=next(gss.split(ds,groups=groups))
    train_ds, val_ds = Subset(ds,ti), Subset(ds,vi)

    train_labels=[ds.labels[i][0] for i in ti]
    pos_w=torch.tensor([(len(train_labels)-sum(train_labels))/sum(train_labels)]).to(device)
    print(f"Pos weight: {pos_w.item():.4f}")

    train_loader=DataLoader(train_ds,batch_size=bs,shuffle=True,num_workers=2,pin_memory=True,drop_last=True)
    val_loader  =DataLoader(val_ds,  batch_size=bs,shuffle=False,num_workers=2,pin_memory=True,drop_last=True)

    model=DeepfakeModel(seq_len=seq_len).to(device)
    crit=CombinedLoss(bce_weight=0.5,jsd_weight=0.5,pos_weight=pos_w)
    opt=optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-5)
    sched=optim.lr_scheduler.ReduceLROnPlateau(opt,mode="min",factor=0.5,patience=3,verbose=True)

    best_auc=0
    for e in range(1,epochs+1):
        start=time.time()
        tr=train_epoch(model,train_loader,crit,opt,device,e)
        vl=evaluate_model(model,val_loader,crit,device,e)
        sched.step(tr[0])

        print(f"\nEpoch {e}/{epochs} [{time.time()-start:.1f}s]")
        print(f" Train ▶ loss {tr[0]:.4f}, acc {tr[1]:.4f}, f1 {tr[2]:.4f}, auc {tr[3]:.4f}, rec {tr[4]:.4f}, FRR {tr[5]:.4f}, GAR {tr[6]:.4f}, prec {tr[7]:.4f}, EER {tr[8]:.4f}")
        print(f" Val   ▶ loss {vl[0]:.4f}, acc {vl[1]:.4f}, f1 {vl[2]:.4f}, auc {vl[3]:.4f}, rec {vl[4]:.4f}, FRR {vl[5]:.4f}, GAR {vl[6]:.4f}, prec {vl[7]:.4f}, EER {vl[8]:.4f}")

        res={
            "epoch":e,
            "train":dict(zip(["loss","acc","f1","auc","recall","FRR","GAR","prec","EER"],tr)),
            "val":  dict(zip(["loss","acc","f1","auc","recall","FRR","GAR","prec","EER"],vl)),
            "lr":opt.param_groups[0]["lr"],
            "time":time.time()-start
        }
        save_model_and_result(model,res,f"outputs/models/epoch_{e}.pt",f"outputs/results/epoch_{e}.json")

        if vl[3]>best_auc:
            best_auc=vl[3]
            torch.save(model.state_dict(),"outputs/models/best_model.pt")
            print(">> New best model saved.")

    torch.save(model.state_dict(),"outputs/models/final_model.pt")
    print("Training complete, final model saved.")

if __name__=="__main__":
    main()
