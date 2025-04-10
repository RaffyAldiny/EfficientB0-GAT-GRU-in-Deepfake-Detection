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
            A.GaussNoise(std_range=(0.02, 0.05), p=1.0),
            A.MotionBlur(blur_limit=(3, 6), p=0.25),
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
            A.ImageCompression(quality_lower=5, quality_upper=20, p=0.5),
        ])),
        ("Sensor Noise", A.ReplayCompose([
            A.MotionBlur(blur_limit=(2, 3), p=0.2),
            A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.25), p=1.0),
            A.CLAHE(clip_limit=1, tile_grid_size=(8,8), p=0.4),
        ])),
        ("Edge Analysis", A.ReplayCompose([
            A.Emboss(alpha=(0.08, 0.15), strength=(1, 1), p=1.0),
            A.RandomBrightnessContrast(contrast_limit=(0.1, 0.2), p=0.6),
            A.CLAHE(clip_limit=1, p=0.5)
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

    # First frame: generate augmentation + record params
    fp0 = os.path.join(in_folder, frames[0])
    img0 = np.array(Image.open(fp0).convert("RGB"))
    res0 = pipeline(image=img0)
    aug0, replay = res0["image"], res0["replay"]
    Image.fromarray(aug0).save(os.path.join(out_folder, frames[0]))

    # Replay the same parameters on remaining frames
    for fname in frames[1:]:
        img = np.array(Image.open(os.path.join(in_folder, fname)).convert("RGB"))
        res = pipeline.replay(replay, image=img)
        Image.fromarray(res["image"]).save(os.path.join(out_folder, fname))

def main():
    random.seed(42)

    # Locate Training folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.normpath(
        os.path.join(script_dir, "..", "data", "Final-data", "Training")
    )
    if not os.path.isdir(base_dir):
        print(f"ERROR: Training directory not found:\n  {base_dir}", file=sys.stderr)
        sys.exit(1)

    pipelines  = get_augmentation_pipelines()
    categories = ["Celeb-real", "Celeb-synthesis"]

    # Gather original video-folder names per class
    vids = {}
    for cat in categories:
        cat_path = os.path.join(base_dir, cat)
        vids[cat] = [
            d for d in os.listdir(cat_path)
            if os.path.isdir(os.path.join(cat_path, d))
            and not d.endswith("_data-augmented")
        ]

    # Prepare occurrence counter and record of new folders + augmentation type
    occurrence = { (cat, vid): 0 for cat in categories for vid in vids[cat] }
    augmented_records = []  # will hold "relative_path | augmentation_name"

    # Phase 1: augment 25% of each class
    phase1_counts = {}
    for cat in categories:
        folder_list = vids[cat]
        orig_count  = len(folder_list)
        if orig_count == 0:
            phase1_counts[cat] = 0
            continue

        aug_needed = max(1, int(orig_count * 0.25))
        aug_needed = min(aug_needed, orig_count)
        picks = random.sample(folder_list, aug_needed)
        phase1_counts[cat] = len(picks)

        print(f"[Phase 1] Augmenting {len(picks)}/{orig_count} folders in `{cat}` (50%)")
        for vid in picks:
            occurrence[(cat, vid)] += 1
            cnt = occurrence[(cat, vid)]
            suffix = "_data-augmented" if cnt == 1 else f"_data-augmented_{cnt}"
            in_folder  = os.path.join(base_dir, cat, vid)
            out_folder = os.path.join(base_dir, cat, vid + suffix)

            if os.path.exists(out_folder):
                print(f"  [!] Skipping existing: {out_folder}")
                continue

            aug_name, aug_pipe = random.choice(pipelines)
            print(f"  → [{cat}] {vid}{suffix}  |  {aug_name}")
            augment_folder_frames(in_folder, aug_pipe, out_folder)

            rel_path = os.path.join(cat, vid + suffix)
            augmented_records.append(f"{rel_path} | {aug_name}")

    # Compute new class sizes (for logging only)
    new_counts = {
        cat: len(vids[cat]) + phase1_counts.get(cat, 0)
        for cat in categories
    }
    print(f"\nPost‑Phase 1 counts: {new_counts}")

    # Phase 2: balance minority to majority, but only use folders that were not augmented yet.
    real_count = new_counts["Celeb-real"]
    fake_count = new_counts["Celeb-synthesis"]
    if fake_count < real_count:
        minority = "Celeb-synthesis"
        disparity = real_count - fake_count
    else:
        minority = "Celeb-real"
        disparity = fake_count - real_count

    if disparity <= 0:
        print("Already balanced after Phase 1; no further augmentation needed.")
    else:
        # Only pick folders that have not been augmented (occurrence == 0)
        minority_list = [vid for vid in vids[minority] if occurrence[(minority, vid)] == 0]
        orig_min = len(minority_list)
        if orig_min == 0:
            print(f"All folders in {minority} have been augmented; cannot augment uniquely.")
        else:
            if disparity > orig_min:
                print(f"[Phase 2] Need {disparity} but only {orig_min} unique folders; augmenting each once.")
                picks_min = minority_list[:]
            else:
                picks_min = random.sample(minority_list, disparity)

            print(f"[Phase 2] Augmenting {len(picks_min)}/{orig_min} folders in `{minority}` to balance")
            for vid in picks_min:
                occurrence[(minority, vid)] += 1
                cnt = occurrence[(minority, vid)]
                suffix = "_data-augmented" if cnt == 1 else f"_data-augmented_{cnt}"
                in_folder  = os.path.join(base_dir, minority, vid)
                out_folder = os.path.join(base_dir, minority, vid + suffix)

                if os.path.exists(out_folder):
                    print(f"  [!] Skipping existing: {out_folder}")
                    continue

                aug_name, aug_pipe = random.choice(pipelines)
                print(f"  → [{minority}] {vid}{suffix}  |  {aug_name}")
                augment_folder_frames(in_folder, aug_pipe, out_folder)

                rel_path = os.path.join(minority, vid + suffix)
                augmented_records.append(f"{rel_path} | {aug_name}")

    # Write out list of all augmented folders and their pipeline names
    if augmented_records:
        txt_path = os.path.join(base_dir, "augmented_folders.txt")
        with open(txt_path, "w") as f:
            for record in augmented_records:
                f.write(record + "\n")
        print(f"\nWrote list of augmented folders and pipelines to: {txt_path}")

if __name__ == "__main__":
    main()
