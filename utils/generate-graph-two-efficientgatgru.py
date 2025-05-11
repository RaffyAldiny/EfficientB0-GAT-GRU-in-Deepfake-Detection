#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare two training histories (first 20 epochs) and save an accuracy‑curve figure.

Features
--------
• Accepts any accuracy‑key spelling ("Training Accuracy", "accuracy", "acc", …).
• Optional --json-dir to avoid any search latency.
• Shallow fallback search (≤4 levels) if a file isn’t found directly.
• Uses orjson automatically for faster loading if installed.
• Date‑stamped output:  Result Figures/Accuracy <A> vs <B>_<YYYY-MM-DD>.png
"""

from __future__ import annotations
import sys, argparse, re
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 🔧 EDIT HERE (or pass --file-a / --file-b on CLI) ────────────────────────────
FILE_A = "FaceForens-EfficientB0GatGru.json"   # model A JSON filename
FILE_B = "FACEFORENSICS-EfficientNetB0Alone.json"  # model B JSON filename
LABELS = {FILE_A: "EfficientNetB0 + GAT‑GRU",
          FILE_B: "EfficientNetB0 Alone"}

MAX_EPOCHS = 20
MAX_DEPTH  = 4          # search depth if direct path fails
COLORS     = ["#0066ff", "#ff9900"]
MARKERS    = ["o", "X"]

# ──────────────────────────────────────────────────────────────────────────────
# Fast JSON loader (orjson if available)
try:
    import orjson as fastjson  # type: ignore
    def load_json(path: Path): return fastjson.loads(path.read_bytes())
except ModuleNotFoundError:
    import json as fastjson
    def load_json(path: Path):
        with path.open(encoding="utf-8") as f:
            return fastjson.load(f)

# Regex to find accuracy-like key
ACC_RGX = re.compile(r"acc|accuracy", re.I)

def find_accuracy_key(sample_dict: dict) -> str:
    """Return the first key that looks like an accuracy metric."""
    for k in sample_dict:
        if ACC_RGX.search(k):
            return k
    raise KeyError("No accuracy‑like key found in JSON epoch record!")


def shallow_search(root: Path, filename: str, depth: int = MAX_DEPTH) -> Path | None:
    queue = [root]
    for _ in range(depth + 1):
        next_q: list[Path] = []
        for folder in queue:
            cand = folder / filename
            if cand.is_file():
                return cand
            next_q += [p for p in folder.iterdir() if p.is_dir()]
        queue = next_q
    return None


def locate(path_hint: Path | None, fname: str) -> Path:
    # direct hint
    if path_hint and (path_hint / fname).is_file():
        return path_hint / fname
    # script root
    script_root = Path(__file__).resolve().parents[1]
    # direct under root
    if (script_root / fname).is_file():
        return script_root / fname
    # shallow search
    hit = shallow_search(script_root, fname)
    if hit:
        return hit
    raise FileNotFoundError(f"{fname} not found under {script_root}")


def extract_metric(js: list[dict], split: str, key: str) -> np.ndarray:
    return np.array([ep[split][key] for ep in js][:MAX_EPOCHS])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Plot Training and Validation Accuracy curves for two models."
    )
    ap.add_argument("--json-dir", type=Path,
                    help="Folder containing both JSON logs")
    ap.add_argument("--file-a", type=str, default=FILE_A,
                    help="Filename for model A")
    ap.add_argument("--file-b", type=str, default=FILE_B,
                    help="Filename for model B")
    args = ap.parse_args(argv)

    # Locate files
    paths = [locate(args.json_dir, fname) for fname in (args.file_a, args.file_b)]
    labels = [LABELS.get(p.name, p.stem) for p in paths]

    # Load data
    data = [load_json(p) for p in paths]
    # Detect accuracy keys
    tr_keys = [find_accuracy_key(d[0]["Training"]) for d in data]
    val_keys = [find_accuracy_key(d[0]["Testing"]) for d in data]

    # Prepare output path
    out_dir = paths[0].parent / "Result Figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    date_tag = datetime.now().strftime("%Y-%m-%d")
    out_png = out_dir / f"Accuracy {labels[0]} vs {labels[1]}_{date_tag}.png"

    # Plot settings
    rcParams.update({"axes.grid": True, "grid.linestyle": ":", "font.size": 13})
    fig, (ax_tr, ax_val) = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

    for i, lbl in enumerate(labels):
        tr = extract_metric(data[i], "Training", tr_keys[i])
        vl = extract_metric(data[i], "Testing", val_keys[i])
        epochs = np.arange(1, len(tr) + 1)

        ax_tr.plot(
            epochs, tr,
            label=f"{lbl} (best E{tr.argmax()+1}:{tr.max():.4f})",
            color=COLORS[i], marker=MARKERS[i]
        )
        ax_val.plot(
            epochs, vl,
            label=f"{lbl} (best E{vl.argmax()+1}:{vl.max():.4f})",
            color=COLORS[i], marker=MARKERS[i]
        )

        ax_tr.axvline(tr.argmax()+1, color=COLORS[i], ls="--", lw=1)
        ax_val.axvline(vl.argmax()+1, color=COLORS[i], ls="--", lw=1)

    ax_tr.set(title="Training Accuracy (first 20 epochs)", xlabel="Epoch", ylabel="Accuracy")
    ax_val.set(title="Validation Accuracy (first 20 epochs)", xlabel="Epoch")
    ax_tr.legend(frameon=True)
    ax_val.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    print(f"✅  Figure saved to: {out_png}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.exit(f"❌  {e}")
