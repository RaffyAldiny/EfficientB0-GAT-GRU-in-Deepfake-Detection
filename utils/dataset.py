import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision.transforms import ToTensor
import random

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, labels_file, transform=None, limit=None, seq_len=40):
        self.root_dir = root_dir
        self.transform = transform or ToTensor()
        self.seq_len = seq_len
        self.labels = self._load_labels(labels_file)

        valid = []
        for label, rp, cid in self.labels:
            folder = os.path.join(root_dir, os.path.splitext(rp)[0])
            if os.path.isdir(folder):
                frames = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.png'))])
                if frames:
                    valid.append((label, rp, cid, len(frames)))
        if limit:
            valid = valid[:limit]
        self.labels = valid
        print(f"Dataset initialized with {len(self.labels)} samples")

    def _load_labels(self, fpath):
        out = []
        with open(fpath) as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) != 2:
                    continue
                lab, rp = parts
                try:
                    lab = int(lab)
                except:
                    continue
                if "Celeb-real" in rp:
                    cid = rp.split("/")[-1].split("_")[0]
                elif "Celeb-synthesis" in rp:
                    cid = rp.split("/")[-1].split("_")[0]
                elif "Youtube-real" in rp:
                    cid = rp.split("/")[-1]
                else:
                    continue
                out.append((lab, rp, cid))
        return out

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        lab, rp, cid, total = self.labels[idx]
        folder = os.path.join(self.root_dir, os.path.splitext(rp)[0])
        frames = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.png'))])

        # Select/pad frames
        if total >= self.seq_len:
            start = random.randint(0, total - self.seq_len)
            sel = frames[start:start + self.seq_len]
            idxs = list(range(start, start + self.seq_len))
        else:
            sel = frames
            idxs = list(range(total))

        imgs = []
        for fn in sel:
            try:
                im = Image.open(os.path.join(folder, fn)).convert("RGB")
                imgs.append(self.transform(im))
            except:
                imgs.append(torch.zeros(3, 224, 224))
        if len(imgs) < self.seq_len:
            pad = imgs[-1]
            imgs += [pad.clone() for _ in range(self.seq_len - len(imgs))]
        imgs = torch.stack(imgs)  # [seq_len, 3, 224, 224]

        # Load landmarks
        lm_path = os.path.join(folder, "landmarks.pt")
        if os.path.exists(lm_path):
            all_lm = torch.load(lm_path, map_location='cpu')
            if total >= self.seq_len:
                lm = all_lm[idxs]
            else:
                pad = all_lm[-1].unsqueeze(0).repeat(self.seq_len - total, 1, 1)
                lm = torch.cat([all_lm, pad], 0)
        else:
            lm = torch.zeros(self.seq_len, 68, 2)

        # Sanity checks
        assert lm.shape == (self.seq_len, 68, 2), f"Invalid landmark shape {lm.shape} in {folder}"
        if torch.all(imgs[-1] == 0):
            print(f"Warning: Excessive padding in {folder}")

        return imgs, torch.tensor(lab, dtype=torch.float32), lm
