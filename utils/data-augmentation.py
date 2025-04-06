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
            A.MotionBlur(blur_limit=(3, 6), p=0.25),
            A.CLAHE(clip_limit=1.5, tile_grid_size=(8,8), p=1.0),
        ])),
        ("Color & Sharpness Adjustment", A.ReplayCompose([
            A.RandomGamma(gamma_limit=(50, 125), p=0.75),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=1.0
            ),
            A.MedianBlur(blur_limit=(3,6), p=0.3),
            A.Sharpen(alpha=(0.1, 0.4), lightness=(0.75, 1.5), p=1.0),
        ])),
        ("Blur, Partial Desaturation & Scale", A.ReplayCompose([
            A.GaussianBlur(blur_limit=(5, 7), p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=0,            # no hue shift
                sat_shift_limit=(-70, -30),   # partial desaturation
                val_shift_limit=0,
                p=0.6
            ),
            A.ToGray(p=0.4),               # 40% chance full grayscale
            A.RandomScale(scale_limit=0.15, p=0.4),
        ])),
        ("JPEG Compression", A.ReplayCompose([
            A.JpegCompression(quality_lower=40, quality_upper=80, p=0.5),
        ])),
    ]

def augment_folder_frames(in_folder, pipeline, out_folder):
    """
    Apply the SAME random augmentation (identical params) to every image in `in_folder`,
    writing results into `out_folder`.
    """
    # gather all image files
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

    pipelines = get_augmentation_pipelines()
    categories = ["Celeb-real", "Celeb-synthesis"]

    # --- 1) Gather all video-folder names per class ---
    vids = {}
    for cat in categories:
        cat_path = os.path.join(base_dir, cat)
        vids[cat] = [
            d for d in os.listdir(cat_path)
            if os.path.isdir(os.path.join(cat_path, d))
            and not d.endswith("_data-augmented")
        ]

    n_real = len(vids["Celeb-real"])
    n_fake = len(vids["Celeb-synthesis"])
    print(f"Found {n_real} real vs {n_fake} fake videos in Training/")

    # --- 2) Determine minority & majority, compute disparity ---
    if n_fake < n_real:
        minority, majority = "Celeb-synthesis", "Celeb-real"
        disparity = n_real - n_fake
    else:
        minority, majority = "Celeb-real", "Celeb-synthesis"
        disparity = n_fake - n_real

    print(f"Minority class: {minority} (need +{disparity} aug samples)")

    # --- 3) Prepare a list of picks for minority augmentation ---
    minority_list = vids[minority]
    if not minority_list:
        print(f"ERROR: No videos found in minority class `{minority}`", file=sys.stderr)
        sys.exit(1)

    # Round-robin sample until we have exactly `disparity` picks
    picks_min = []
    while len(picks_min) < disparity:
        picks_min.extend(random.sample(minority_list, len(minority_list)))
    picks_min = picks_min[:disparity]

    # --- 4) Optionally, do a small amount of majority augmentation for variety ---
    maj_rate = 0.02
    majority_list = vids[majority]
    n_maj = max(1, int(len(majority_list) * maj_rate))
    picks_maj = random.sample(majority_list, n_maj)
    print(f"Augmenting {len(picks_min)} from `{minority}` and {len(picks_maj)} from `{majority}`")

    # --- 5) Run the augmentations ---
    # We'll keep a counter so we can suffix duplicates uniquely
    occurrence = { (minority, v): 0 for v in minority_list }
    occurrence.update({ (majority, v): 0 for v in majority_list })

    for cat in categories:
        cat_path = os.path.join(base_dir, cat)
        picks = picks_min if cat == minority else picks_maj

        for vid in picks:
            occurrence[(cat, vid)] += 1
            count = occurrence[(cat, vid)]
            suffix = "_data-augmented" if count == 1 else f"_data-augmented_{count}"
            out_folder = os.path.join(cat_path, vid + suffix)
            in_folder  = os.path.join(cat_path, vid)

            if os.path.exists(out_folder):
                print(f"  [!] Skipping existing: {out_folder}")
                continue

            aug_name, aug_pipe = random.choice(pipelines)
            print(f"  → [{cat}] {vid}{suffix}  |  {aug_name}")
            augment_folder_frames(in_folder, aug_pipe, out_folder)

if __name__ == "__main__":
    main()
