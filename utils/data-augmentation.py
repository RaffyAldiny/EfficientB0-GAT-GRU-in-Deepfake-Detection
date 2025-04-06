#!/usr/bin/env python3
import os
import random
import sys

import albumentations as A
import numpy as np
from PIL import Image

def get_augmentation_pipelines():
    """Return a list of (name, ReplayCompose pipeline) tuples."""
    return [
        ("Noise & Contrast Enhancement", A.ReplayCompose([
            A.RandomBrightnessContrast(p=0.75),
            A.GaussNoise(std_range=(0.02, 0.06), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=0.25),  # ensure odd limits
            A.CLAHE(clip_limit=1.5, tile_grid_size=(8,8), p=1.0),
        ])),
        ("Color & Sharpness Adjustment", A.ReplayCompose([
            A.RandomGamma(gamma_limit=(50, 125), p=0.75),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=1.0
            ),
            A.MedianBlur(blur_limit=(3,7), p=0.3),
            A.Sharpen(alpha=(0.1, 0.4), lightness=(0.75, 1.5), p=1.0),
        ])),
        ("Blur, Partial Desaturation & Scale", A.ReplayCompose([
            A.GaussianBlur(blur_limit=(5, 7), p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=0,
                sat_shift_limit=(-70, -30),
                val_shift_limit=0,
                p=0.6
            ),
            A.ToGray(p=0.4),
            A.RandomScale(scale_limit=0.15, p=0.4),
        ])),
        ("JPEG Compression", A.ReplayCompose([
            A.ImageCompression(quality_lower=20, quality_upper=30, p=0.5),
        ])),
    ]

def augment_folder_frames(in_folder, pipeline, out_folder):
    """
    Apply the SAME random augmentation (identical params) to every image in `in_folder`,
    writing results into `out_folder`.
    """
    frames = sorted(f for f in os.listdir(in_folder)
                    if f.lower().endswith((".png", ".jpg", ".jpeg")))
    if not frames:
        print(f"  [!] No images in {in_folder}, skipping.")
        return

    os.makedirs(out_folder, exist_ok=True)

    # first frame: generate augmentation + record params
    fp0 = os.path.join(in_folder, frames[0])
    img0 = np.array(Image.open(fp0).convert("RGB"))
    res0 = pipeline(image=img0)
    aug0, replay = res0["image"], res0["replay"]
    Image.fromarray(aug0).save(os.path.join(out_folder, frames[0]))

    # replay same params on remaining frames
    for fname in frames[1:]:
        img = np.array(Image.open(os.path.join(in_folder, fname)).convert("RGB"))
        res = pipeline.replay(replay, image=img)
        Image.fromarray(res["image"]).save(os.path.join(out_folder, fname))

def main():
    random.seed(42)

    # --- Locate Training folder (80/20 split must be done already) ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.normpath(
        os.path.join(script_dir, "..", "data", "Final-data", "Training")
    )
    if not os.path.isdir(base_dir):
        print(f"ERROR: Training directory not found:\n  {base_dir}", file=sys.stderr)
        sys.exit(1)

    pipelines  = get_augmentation_pipelines()
    categories = ["Celeb-real", "Celeb-synthesis"]

    # --- Gather all video-folder names per class (ignore existing aug folders) ---
    vids = {}
    for cat in categories:
        cat_path = os.path.join(base_dir, cat)
        vids[cat] = [
            d for d in os.listdir(cat_path)
            if os.path.isdir(os.path.join(cat_path, d))
            and not d.endswith("_data-augmented")
        ]

    augmented_folders = []

    # --- For each class, augment exactly 25% of its folders once ---
    for cat in categories:
        folder_list = vids[cat]
        orig_count  = len(folder_list)
        if orig_count == 0:
            print(f"[!] No folders in `{cat}`, skipping.")
            continue

        # compute how many to augment: at least 1, at most orig_count
        aug_needed = max(1, int(orig_count * 0.25))
        aug_needed = min(aug_needed, orig_count)

        # sample unique folders to augment
        picks = random.sample(folder_list, aug_needed)
        print(f"Augmenting {len(picks)}/{orig_count} folders in `{cat}` (25%)")

        # perform augmentation
        for vid in picks:
            in_folder  = os.path.join(base_dir, cat, vid)
            out_folder = os.path.join(base_dir, cat, vid + "_data-augmented")

            if os.path.exists(out_folder):
                print(f"  [!] Skipping existing: {out_folder}")
                continue

            aug_name, aug_pipe = random.choice(pipelines)
            print(f"  → [{cat}] {vid}_data-augmented  |  {aug_name}")
            augment_folder_frames(in_folder, aug_pipe, out_folder)
            # record the new folder name relative to base_dir
            rel_path = os.path.join(cat, vid + "_data-augmented")
            augmented_folders.append(rel_path)

    # --- Write out list of augmented folders ---
    if augmented_folders:
        txt_path = os.path.join(base_dir, "augmented_folders.txt")
        with open(txt_path, "w") as f:
            for line in augmented_folders:
                f.write(line + "\n")
        print(f"\nWrote list of augmented folders to: {txt_path}")

if __name__ == "__main__":
    main()
