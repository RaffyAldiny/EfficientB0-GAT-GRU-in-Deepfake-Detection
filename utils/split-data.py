#!/usr/bin/env python3
import os
import shutil
import random
from collections import defaultdict

def get_id_prefix(folder_name: str) -> str:
    """
    Extract the identity prefix (e.g. 'id0') from a folder name like 'id0_xyz...'.
    Adjust the split logic if your naming scheme differs.
    """
    return folder_name.split('_')[0]

def split_identities(class_dir: str, train_ratio=0.8, seed=42):
    """
    Given a directory of class samples (each subfolder is one video/instance),
    group them by identity prefix, then split identities into train/test.
    Returns two lists of folder-names: train_folders, test_folders.
    """
    # 1) List all subfolders
    all_folders = [
        d for d in os.listdir(class_dir)
        if os.path.isdir(os.path.join(class_dir, d))
    ]

    # 2) Group by identity prefix
    id2folders = defaultdict(list)
    for folder in all_folders:
        pid = get_id_prefix(folder)
        id2folders[pid].append(folder)

    # 3) Shuffle and split identity keys
    identities = list(id2folders.keys())
    random.Random(seed).shuffle(identities)
    n_train = int(len(identities) * train_ratio)
    train_ids = set(identities[:n_train])
    test_ids  = set(identities[n_train:])

    # 4) Assign folders by identity set
    train_folders = []
    test_folders  = []
    for pid, folders in id2folders.items():
        if pid in train_ids:
            train_folders.extend(folders)
        else:
            test_folders.extend(folders)

    return train_folders, test_folders

def main():
    # Paths
    source_root = "data/process-merge"
    output_root = "data/Split-data"
    classes     = ["Celeb-real", "Celeb-synthesis"]
    train_ratio = 0.8
    seed        = 42

    # 1) Create target directories
    for split in ["Training", "Testing"]:
        for cls in classes:
            out_dir = os.path.join(output_root, split, cls)
            os.makedirs(out_dir, exist_ok=True)

    # 2) For each class, perform the split & copy
    for cls in classes:
        class_src = os.path.join(source_root, cls)
        if not os.path.isdir(class_src):
            print(f"Warning: class folder not found: {class_src}")
            continue

        print(f"\nProcessing class `{cls}`…")
        train_folders, test_folders = split_identities(class_src, train_ratio, seed)
        print(f" → {len(train_folders)} folders → Training")
        print(f" → {len(test_folders)} folders → Testing")

        # Copy train
        for folder in train_folders:
            src = os.path.join(class_src, folder)
            dst = os.path.join(output_root, "Training", cls, folder)
            shutil.copytree(src, dst)

        # Copy test
        for folder in test_folders:
            src = os.path.join(class_src, folder)
            dst = os.path.join(output_root, "Testing", cls, folder)
            shutil.copytree(src, dst)

    print("\nDone! Your data is now in `data/Split-data/Training/…` and `data/Split-data/Testing/…`")

if __name__ == "__main__":
    main()
