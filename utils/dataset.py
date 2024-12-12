# utils/dataset.py
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision.transforms import ToTensor
import random

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, labels_file, transform=None, limit=None, seq_len=100):
        self.root_dir = root_dir
        self.transform = transform if transform else ToTensor()
        self.labels = self._load_labels(labels_file)
        self.seq_len = seq_len

        # Filter out invalid samples
        valid_labels = []
        skipped = 0
        for label, rel_path in self.labels:
            video_folder = os.path.join(self.root_dir, os.path.splitext(rel_path)[0])
            if os.path.exists(video_folder):
                frame_files = sorted([f for f in os.listdir(video_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                if len(frame_files) >= 1:
                    valid_labels.append((label, rel_path, len(frame_files)))
                else:
                    print(f"Warning: No frames found in folder, skipping: {video_folder}")
                    skipped += 1
            else:
                print(f"Warning: Video folder not found, skipping: {video_folder}")
                skipped += 1

        if limit is not None:
            valid_labels = valid_labels[:limit]

        self.labels = valid_labels
        print(f"Dataset initialized with {len(self.labels)} samples. Skipped {skipped} invalid samples.")

    def _load_labels(self, labels_file):
        labels = []
        with open(labels_file, "r") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) != 2:
                    print(f"Warning: Invalid label line format: {line.strip()}")
                    continue
                label, relative_path = parts
                try:
                    label = int(label)
                    labels.append((label, relative_path))
                except ValueError:
                    print(f"Warning: Invalid label value: {label} in line: {line.strip()}")
        return labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label, relative_path, total_frames = self.labels[idx]
        video_folder = os.path.join(self.root_dir, os.path.splitext(relative_path)[0])
        frame_files = sorted([f for f in os.listdir(video_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        frames = []
        if total_frames >= self.seq_len:
            # Randomly select a starting index
            start_idx = random.randint(0, total_frames - self.seq_len)
            selected_frames = frame_files[start_idx:start_idx + self.seq_len]
        else:
            # Use all available frames and pad the rest
            selected_frames = frame_files
            padding_needed = self.seq_len - total_frames

        for frame_file in selected_frames:
            frame_path = os.path.join(video_folder, frame_file)
            try:
                image = Image.open(frame_path).convert("RGB")
                frames.append(self.transform(image))
            except Exception as e:
                print(f"Warning: Failed to load frame {frame_path}: {e}")
                # If a frame fails to load, use the last successfully loaded frame or a black image
                if len(frames) > 0:
                    frames.append(frames[-1])
                else:
                    frames.append(torch.zeros(3, 224, 224))

        if total_frames < self.seq_len:
            # Pad with the last frame
            last_frame = frames[-1]
            for _ in range(self.seq_len - total_frames):
                frames.append(last_frame.clone())

        frames_tensor = torch.stack(frames)  # Shape: [seq_len, 3, 224, 224]
        return frames_tensor, torch.tensor(label, dtype=torch.float32)
