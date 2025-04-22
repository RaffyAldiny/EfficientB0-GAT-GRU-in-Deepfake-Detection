#!/usr/bin/env python3
"""
Plot accuracy curves for EfficientGatGRU runs (0 % → 100 % augmentation).

• Place this script anywhere in your project.
• Make sure the folder ‘Augmented 0-100 % Result/’ sits next to it (or
  change `FOLDER_NAME` below).
• Run with:  python plot_accuracy.py
"""

import os
import glob
import json
from datetime import datetime
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────────────
# Settings – adjust if your directory structure changes
# ──────────────────────────────────────────────────────────────────────────────
FOLDER_NAME = "Augmented 0-100 % Result"      # folder that holds the JSON files
FILE_GLOB   = "EfficientGatGRU-*%_augmented.json"   # file‑name pattern
# ──────────────────────────────────────────────────────────────────────────────


def discover_json_files(folder_path: str, pattern: str):
    """Return a sorted list ‑‑ (pct_int, pct_label, full_path)."""
    files_info = []
    for fp in glob.glob(os.path.join(folder_path, pattern)):
        base = os.path.basename(fp)
        try:
            pct_int = int(base.split("-")[1].split("%")[0])
            files_info.append((pct_int, f"{pct_int}%", fp))
        except (IndexError, ValueError):
            continue
    return sorted(files_info, key=lambda x: x[0])


def load_histories(files_info):
    """Parse JSON content → dicts: {pct_label: (epochs, acc_list)}."""
    train_hist, val_hist = {}, {}

    for _, pct_label, fp in files_info:
        with open(fp, "r") as f:
            epochs_data = json.load(f)

        epochs    = [d["Epoch"]                          for d in epochs_data]
        train_acc = [d["Training"]["Training Accuracy"]  for d in epochs_data]
        val_acc   = [d["Testing"]["Val Accuracy"]        for d in epochs_data]

        train_hist[pct_label] = (epochs, train_acc)
        val_hist[pct_label]   = (epochs, val_acc)

    return train_hist, val_hist


def plot_and_save(train_hist, val_hist, save_dir):
    """Draw two subplots and save with YYYYMMDD_HHMMSS timestamp suffix."""
    fig, (ax_t, ax_v) = plt.subplots(1, 2, figsize=(14, 6))

    # ── Training ────────────────────────────────────────────────────────────
    for pct_label, (epochs, acc) in train_hist.items():
        best_idx = max(range(len(acc)), key=lambda i: acc[i])
        ax_t.plot(
            epochs,
            acc,
            marker="o",
            label=f"{pct_label} Dataset Augmentation "
                  f"(Train: Epoch {epochs[best_idx]} – {acc[best_idx]:.4f})",
        )
    ax_t.set(title="Training Accuracy", xlabel="Epoch", ylabel="Accuracy")
    ax_t.grid(True, ls="--", alpha=0.5)
    ax_t.legend(fontsize="small")

    # ── Validation ──────────────────────────────────────────────────────────
    for pct_label, (epochs, acc) in val_hist.items():
        best_idx = max(range(len(acc)), key=lambda i: acc[i])
        ax_v.plot(
            epochs,
            acc,
            marker="o",
            label=f"{pct_label} Dataset Augmentation "
                  f"(Val: Epoch {epochs[best_idx]} – {acc[best_idx]:.4f})",
        )
    ax_v.set(title="Validation Accuracy", xlabel="Epoch", ylabel="Accuracy")
    ax_v.grid(True, ls="--", alpha=0.5)
    ax_v.legend(fontsize="small")

    plt.tight_layout()

    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"accuracy_comparison_{ts}.png"
    out  = os.path.join(save_dir, name)
    fig.savefig(out, dpi=300)
    print(f"✔️  Plot saved to:  {out}")


def main():
    folder_path = os.path.join(os.getcwd(), FOLDER_NAME)
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    files_info = discover_json_files(folder_path, FILE_GLOB)
    if not files_info:
        raise FileNotFoundError(
            f"No JSON files matching '{FILE_GLOB}' inside '{folder_path}'."
        )

    train_hist, val_hist = load_histories(files_info)
    plot_and_save(train_hist, val_hist, folder_path)


if __name__ == "__main__":
    main()
