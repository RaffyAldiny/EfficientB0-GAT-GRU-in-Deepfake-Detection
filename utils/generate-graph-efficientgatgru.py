#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare up to four training histories (first 20 epochs) and save an accuracy-curve figure.

Features
--------
• Accepts any accuracy-key spelling ("Training Accuracy", "accuracy", "acc", …).
• Optional --json-dir to avoid any search latency.
• Shallow fallback search (≤4 levels) if a file isn’t found directly.
• Uses orjson automatically for faster loading if installed.
• Date-stamped output:  Result Figures/Accuracy Comparison 4 Models_<YYYY-MM-DD>.png
"""

from __future__ import annotations
import sys
import argparse
import re
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ─── CONFIG ────────────────────────────────────────────────────────────────
FILE_NAMES = {
    "CelebDF-EfficientNetB0":               "CelebDF-EfficientB0Alone.json",
    "CelebDF-EfficientNetB0+GAT+GRU":       "CelebDF-EfficientB0GatGru.json",
    "FaceForensics-EfficientNetB0":         "FaceForensics-EfficientB0Alone.json",
    "FaceForensics-EfficientNetB0+GAT+GRU": "FaceForensics-EfficientB0GatGru.json",
}

MAX_EPOCHS = 20
MAX_DEPTH  = 4  # how deep to search if direct path fails

COLORS = {
    "CelebDF-EfficientNetB0":               "#0066ff",
    "CelebDF-EfficientNetB0+GAT+GRU":       "#ff9900",
    "FaceForensics-EfficientNetB0":         "#00cc66",
    "FaceForensics-EfficientNetB0+GAT+GRU": "#cc00cc",
}
MARKERS = {
    "CelebDF-EfficientNetB0":               "o",
    "CelebDF-EfficientNetB0+GAT+GRU":       "X",
    "FaceForensics-EfficientNetB0":         "s",
    "FaceForensics-EfficientNetB0+GAT+GRU": "^",
}

ACC_RGX = re.compile(r"acc|accuracy", re.I)


# ─── FAST JSON LOADER ───────────────────────────────────────────────────────
try:
    import orjson as fastjson  # type: ignore
    def load_json(path: Path):
        return fastjson.loads(path.read_bytes())
except ModuleNotFoundError:
    import json as fastjson
    def load_json(path: Path):
        with path.open(encoding="utf-8") as f:
            return fastjson.load(f)


# ─── FILE LOCATING ─────────────────────────────────────────────────────────
def shallow_search(root: Path, filename: str, max_depth: int = MAX_DEPTH) -> Path | None:
    queue = [root]
    for _ in range(max_depth + 1):
        next_q: list[Path] = []
        for folder in queue:
            candidate = folder / filename
            if candidate.is_file():
                return candidate
            next_q.extend(p for p in folder.iterdir() if p.is_dir())
        queue = next_q
    return None


def locate_jsons(json_dir: Path | None) -> dict[str, Path]:
    project_root = Path(__file__).resolve().parents[1]
    search_roots = [json_dir] if json_dir else [project_root]

    found: dict[str, Path] = {}
    missing = FILE_NAMES.copy()

    for label, fname in FILE_NAMES.items():
        for root in search_roots:
            if root and (root / fname).is_file():
                found[label] = root / fname
                missing.pop(label, None)
                break

    for label, fname in list(missing.items()):
        hit = shallow_search(project_root, fname)
        if hit:
            found[label] = hit
            missing.pop(label)

    if missing:
        msg = "❌  Could not locate:\n" + "\n".join(f"   • {v}" for v in missing.values())
        raise FileNotFoundError(msg)

    return found


# ─── KEY DETECTION & EXTRACTION ─────────────────────────────────────────────
def find_accuracy_key(sample: dict) -> str:
    for k in sample:
        if ACC_RGX.search(k):
            return k
    raise KeyError("No accuracy-like key found in JSON epoch record!")


def extract_metric(js: list[dict], split: str, key: str) -> np.ndarray:
    return np.array([ep[split][key] for ep in js][:MAX_EPOCHS])


# ─── MAIN ───────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Plot Training and Validation Accuracy curves for up to four models."
    )
    parser.add_argument("--json-dir", type=Path,
                        help="Directory containing the JSON logs (skips any search).")
    args = parser.parse_args(argv)

    json_paths = locate_jsons(args.json_dir)
    labels     = list(json_paths.keys())
    data       = {lbl: load_json(path) for lbl, path in json_paths.items()}

    tr_keys  = {lbl: find_accuracy_key(data[lbl][0]["Training"]) for lbl in labels}
    val_keys = {lbl: find_accuracy_key(data[lbl][0]["Testing"])  for lbl in labels}

    out_dir = next(iter(json_paths.values())).parent / "Result Figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    date_tag = datetime.now().strftime("%Y-%m-%d")
    out_png  = out_dir / f"Accuracy Comparison 4 Models_{date_tag}.png"

    rcParams.update({"axes.grid": True, "grid.linestyle": ":", "font.size": 13})
    fig, (ax_tr, ax_val) = plt.subplots(1, 2, figsize=(18, 6))

    for lbl in labels:
        tr      = extract_metric(data[lbl], "Training", tr_keys[lbl])
        vl      = extract_metric(data[lbl], "Testing",  val_keys[lbl])
        epochs  = np.arange(1, len(tr) + 1)

        ax_tr.plot(
            epochs, tr,
            label=f"{lbl} (best E{tr.argmax()+1}:{tr.max():.4f})",
            color=COLORS[lbl], marker=MARKERS[lbl]
        )
        ax_val.plot(
            epochs, vl,
            label=f"{lbl} (best E{vl.argmax()+1}:{vl.max():.4f})",
            color=COLORS[lbl], marker=MARKERS[lbl]
        )

        ax_tr.axvline(tr.argmax()+1, color=COLORS[lbl], ls="--", lw=1)
        ax_val.axvline(vl.argmax()+1, color=COLORS[lbl], ls="--", lw=1)

    ax_tr.set(title="Training Accuracy (first 20 epochs)", xlabel="Epoch", ylabel="Accuracy")
    ax_val.set(title="Validation Accuracy (first 20 epochs)", xlabel="Epoch")

    # smaller legend text; both legends in lower right
    ax_tr.legend(frameon=True, fontsize=9, loc="lower right")
    ax_val.legend(frameon=True, fontsize=9, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    print(f"✅  Figure saved to: {out_png}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.exit(f"❌  {e}")
