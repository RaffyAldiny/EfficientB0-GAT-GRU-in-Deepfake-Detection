#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare two training histories (first 20 epochs) and save a loss‑curve figure.

Features
--------
• Accepts any loss‑key spelling ("Training Loss", "Loss", "train_loss", …).  
• Optional --json-dir to avoid any search latency.  
• Shallow fallback search (≤3 levels) if a file isn’t found directly.  
• Uses orjson automatically for faster loading if installed.  
• Date‑stamped output:  Result Figures/Loss <A> vs <B>_<YYYY-MM-DD>.png
"""

from __future__ import annotations
import sys, argparse, re
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 🔧 EDIT HERE (or pass --file-a / --file-b on CLI) ────────────────────────────
FILE_A = "Best-EfficientNetB0GATGRU.json"   # model A
FILE_B = "EfficientGatGRU-Alone.json"       # model B
LABELS = {FILE_A: "Eff‑Net‑B0 + GAT‑GRU",
          FILE_B: "GAT‑GRU Alone"}

MAX_EPOCHS = 20
MAX_DEPTH  = 3          # search depth if direct path fails
COLORS     = ["#0066ff", "#ff9900"]
MARKERS    = ["o", "X"]

# ──────────────────────────────────────────────────────────────────────────────
#  Fast JSON loader (orjson if available)
try:
    import orjson as fastjson  # type: ignore
    def load_json(path: Path): return fastjson.loads(path.read_bytes())
except ModuleNotFoundError:
    import json as fastjson
    def load_json(path: Path):
        with path.open(encoding="utf‑8") as f: return fastjson.load(f)

LOSS_RGX = re.compile(r"loss", re.I)    # matches any key containing “loss”

def find_loss_key(sample_dict: dict) -> str:
    """Return the first key that looks like a loss metric."""
    for k in sample_dict:
        if LOSS_RGX.search(k):
            return k
    raise KeyError("No loss‑like key found in JSON epoch record!")

def shallow_search(root: Path, filename: str, depth: int = MAX_DEPTH) -> Path | None:
    queue = [root]
    for _ in range(depth + 1):
        next_q = []
        for folder in queue:
            cand = folder / filename
            if cand.is_file():
                return cand
            next_q += [p for p in folder.iterdir() if p.is_dir()]
        queue = next_q
    return None

def locate(path_hint: Path | None, fname: str) -> Path:
    if path_hint and (path_hint / fname).is_file():
        return path_hint / fname
    script_root = Path(__file__).resolve().parents[1]
    if (script_root / fname).is_file():
        return script_root / fname
    hit = shallow_search(script_root, fname)
    if hit:
        return hit
    raise FileNotFoundError(f"{fname} not found under {script_root}")

def extract_losses(js: list[dict], split: str, key: str) -> np.ndarray:
    return np.array([ep[split][key] for ep in js][:MAX_EPOCHS])

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-dir", type=Path, help="Folder containing both logs")
    ap.add_argument("--file-a", type=str, default=FILE_A, help="Filename for model A")
    ap.add_argument("--file-b", type=str, default=FILE_B, help="Filename for model B")
    args = ap.parse_args(argv)

    paths = [locate(args.json_dir, fname) for fname in (args.file_a, args.file_b)]
    labels = [LABELS.get(p.name, p.stem) for p in paths]

    # Load once, detect loss‑key once
    data  = [load_json(p) for p in paths]
    keys  = [find_loss_key(d[0]["Training"]) for d in data]   # Training key per file
    vkeys = [find_loss_key(d[0]["Testing" ]) for d in data]   # Validation key

    # Prepare output
    out_dir  = paths[0].parent / "Result Figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    date_tag = datetime.now().strftime("%Y-%m-%d")
    out_png  = out_dir / f"Loss {labels[0]} vs {labels[1]}_{date_tag}.png"

    rcParams.update({"axes.grid": True, "grid.linestyle": ":", "font.size": 13})
    fig, (ax_tr, ax_val) = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

    for i, lbl in enumerate(labels):
        tr = extract_losses(data[i], "Training", keys[i])
        vl = extract_losses(data[i], "Testing",  vkeys[i])
        epochs = np.arange(1, len(tr) + 1)

        ax_tr.plot(epochs, tr, label=f"{lbl} (best E{tr.argmin()+1}:{tr.min():.4f})",
                   color=COLORS[i], marker=MARKERS[i])
        ax_val.plot(epochs, vl, label=f"{lbl} (best E{vl.argmin()+1}:{vl.min():.4f})",
                    color=COLORS[i], marker=MARKERS[i])

        ax_tr.axvline(tr.argmin()+1, color=COLORS[i], ls="--", lw=1)
        ax_val.axvline(vl.argmin()+1, color=COLORS[i], ls="--", lw=1)

    ax_tr.set(title="Training Loss (first 20 epochs)", xlabel="Epoch", ylabel="Loss")
    ax_val.set(title="Validation Loss (first 20 epochs)", xlabel="Epoch")
    ax_tr.legend(frameon=True); ax_val.legend(frameon=True)
    fig.tight_layout(); fig.savefig(out_png, dpi=200)
    print(f"✅  Figure saved to: {out_png}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.exit(f"❌  {e}")
