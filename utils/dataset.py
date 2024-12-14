import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision.transforms import ToTensor
import random

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, video_list, labels_dict, transform=None, limit=None, seq_len=100):
        """
        Args:
            root_dir (str): The directory where preprocessed video frames are stored.
            video_list (list): List of video relative paths, e.g. ["Celeb-real/id13_0005", "Celeb-synthesis/id48_id40_0006"].
            labels_dict (dict): Dictionary mapping video relative paths to labels (0 or 1).
            transform (callable, optional): A function/transform to apply to the frames.
            limit (int, optional): Limit the number of samples to use.
            seq_len (int): Number of frames to sample per video.
        """
        self.root_dir = root_dir
        self.transform = transform if transform else ToTensor()
        self.seq_len = seq_len

        # Prepare the dataset entries
        valid_labels = []
        skipped = 0

        for vid in video_list:
            if vid not in labels_dict:
                print(f"Warning: {vid} not in labels_dict, skipping.")
                skipped += 1
                continue

            label = labels_dict[vid]
            video_folder = os.path.join(self.root_dir, os.path.splitext(vid)[0])
            if os.path.exists(video_folder):
                frame_files = sorted([f for f in os.listdir(video_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                if len(frame_files) >= 1:
                    valid_labels.append((label, vid, len(frame_files)))
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

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label, relative_path, total_frames = self.labels[idx]
        video_folder = os.path.join(self.root_dir, os.path.splitext(relative_path)[0])
        frame_files = sorted([f for f in os.listdir(video_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        if total_frames >= self.seq_len:
            # Randomly select a starting index
            start_idx = random.randint(0, total_frames - self.seq_len)
            selected_frames = frame_files[start_idx:start_idx + self.seq_len]
        else:
            # Use all available frames and pad the rest
            selected_frames = frame_files
            # padding will be done later

        frames = []
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
                    # Create a dummy black frame if this is the first frame
                    frames.append(torch.zeros(3, 224, 224))

        if total_frames < self.seq_len:
            # Pad with the last frame
            last_frame = frames[-1]
            for _ in range(self.seq_len - total_frames):
                frames.append(last_frame.clone())

        frames_tensor = torch.stack(frames)  # Shape: [seq_len, 3, H, W]
        return frames_tensor, torch.tensor(label, dtype=torch.float32)
