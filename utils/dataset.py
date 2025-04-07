import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision.transforms import ToTensor
import random

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None, limit=None, seq_len=100):
        """
        Scans the given root_dir for video samples in the subfolders:
          - "Celeb-real" (label = 1)
          - "Celeb-synthesis" (label = 0)
        Each video folder is expected to contain frame images.
        If limit is set, only that many samples are used.
        
        **Note:** All folders are scanned (including those with augmented data),
        so augmented samples will be included in training.
        """
        self.root_dir = root_dir
        self.transform = transform or ToTensor()
        self.seq_len = seq_len

        self.labels = []
        # Process each class folder.
        for folder_name, label in [("Celeb-real", 1), ("Celeb-synthesis", 0)]:
            class_dir = os.path.join(self.root_dir, folder_name)
            if not os.path.isdir(class_dir):
                print(f"Warning: Missing directory {class_dir}")
                continue
            for video_id in sorted(os.listdir(class_dir)):
                video_folder = os.path.join(class_dir, video_id)
                if not os.path.isdir(video_folder):
                    continue
                frame_files = sorted([
                    f for f in os.listdir(video_folder)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ])
                if not frame_files:
                    print(f"Warning: No frames in {video_folder}, skipping.")
                    continue
                self.labels.append((label, video_folder, len(frame_files)))
                if limit is not None and len(self.labels) >= limit:
                    break
            if limit is not None and len(self.labels) >= limit:
                break

        print(f"Dataset initialized with {len(self.labels)} samples.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label, video_folder, total_frames = self.labels[idx]
        frame_files = sorted([
            f for f in os.listdir(video_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        # Select frames: randomly choose a contiguous sequence if possible, otherwise pad.
        if total_frames >= self.seq_len:
            start_idx = random.randint(0, total_frames - self.seq_len)
            selected_frames = frame_files[start_idx:start_idx + self.seq_len]
        else:
            selected_frames = frame_files

        frames = []
        for fname in selected_frames:
            path = os.path.join(video_folder, fname)
            try:
                img = Image.open(path).convert("RGB")
                frames.append(self.transform(img))
            except Exception as e:
                print(f"Warning: Failed to load frame {path}: {e}")
                frames.append(frames[-1] if frames else torch.zeros(3, 224, 224))
        if total_frames < self.seq_len:
            # Pad with the last frame.
            last_frame = frames[-1]
            for _ in range(self.seq_len - total_frames):
                frames.append(last_frame.clone())

        frames_tensor = torch.stack(frames)  # Shape: [seq_len, 3, 224, 224]
        return frames_tensor, torch.tensor(label, dtype=torch.float32)
