#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loss‑curve comparison ‒ BCE only  vs  BCE + JSD  (first 20 epochs)

Fast version:
  • Tries direct paths first.
  • Falls back to a *shallow* search (depth ≤ 3) only if needed.
  • Uses orjson if available for quicker JSON parsing.
"""

from __future__ import annotations

import sys
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ‑‑‑ CONFIG ‑‑‑
FILE_NAMES = {
    "BCE Only":  "FaceForens-BCEONLYNOJSD.json",
    "BCE + JSD": "FaceForens-JSBCELoss.json",
}
MAX_EPOCHS   = 20
COLORS       = {"BCE Only": "#0066ff", "BCE + JSD": "#ff9900"}
MARKERS      = {"BCE Only": "o",        "BCE + JSD": "X"}
MAX_DEPTH    = 4            # how deep to search if direct path fails
# ‑‑‑ try to import orjson for speed ‑‑‑
try:
    import orjson as fastjson  # type: ignore
    def load_json(p: Path):
        return fastjson.loads(p.read_bytes())
except ModuleNotFoundError:
    import json as fastjson    # falls back silently
    def load_json(p: Path):
        with p.open(encoding="utf‑8") as f:
            return fastjson.load(f)

# ────────────────────────────────────────────────────────────────
def shallow_search(root: Path, filename: str, max_depth: int = MAX_DEPTH) -> Path | None:
    """Breadth‑first search up to *max_depth* levels; returns first hit or None."""
    queue = [root]
    for depth in range(max_depth + 1):
        next_queue = []
        for folder in queue:
            candidate = folder / filename
            if candidate.is_file():
                return candidate
            next_queue.extend(p for p in folder.iterdir() if p.is_dir())
        queue = next_queue
    return None


def locate_jsons(json_dir: Path | None) -> dict[str, Path]:
    """Locate the two required JSON logs as quickly as possible."""
    project_root = Path(__file__).resolve().parents[1]
    search_roots = [json_dir] if json_dir else [project_root]

    found: dict[str, Path] = {}
    missing: dict[str, str] = {}

    # 1) Direct checks first
    for label, fname in FILE_NAMES.items():
        for root in search_roots:
            if root:
                candidate = root / fname
                if candidate.is_file():
                    found[label] = candidate
                    break
        else:
            missing[label] = fname

    # 2) Shallow search only for the still‑missing ones
    for label, fname in list(missing.items()):
        hit = shallow_search(project_root, fname)
        if hit:
            found[label] = hit
            del missing[label]

    if missing:
        msg = "❌  Could not locate:\n" + "\n".join(f"   • {v}" for v in missing.values())
        raise FileNotFoundError(msg)

    return found


def load_losses(json_path: Path, split: str, key: str) -> np.ndarray:
    data = load_json(json_path)
    return np.array([ep[split][key] for ep in data][:MAX_EPOCHS])


def plot_and_save(json_paths: dict[str, Path]) -> Path:
    out_dir = json_paths["BCE Only"].parent / "Result Figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    date_tag = datetime.now().strftime("%Y-%m-%d")
    out_path = out_dir / f"Loss BCE vs BCE+JSD_{date_tag}.png"

    rcParams.update({"axes.grid": True, "grid.linestyle": ":", "font.size": 13})
    fig, (ax_tr, ax_val) = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

    for label, jpath in json_paths.items():
        tr = load_losses(jpath, "Training", "Training Loss")
        vl = load_losses(jpath, "Testing",  "Val Loss")
        epochs = np.arange(1, len(tr) + 1)

        ax_tr.plot(epochs, tr,
                   label=f"{label} (best≈E{tr.argmin()+1}:{tr.min():.4f})",
                   color=COLORS[label], marker=MARKERS[label])
        ax_val.plot(epochs, vl,
                    label=f"{label} (best≈E{vl.argmin()+1}:{vl.min():.4f})",
                    color=COLORS[label], marker=MARKERS[label])

        ax_tr.axvline(tr.argmin()+1, color=COLORS[label], ls="--", lw=1)
        ax_val.axvline(vl.argmin()+1, color=COLORS[label], ls="--", lw=1)

    ax_tr.set(title="Training Loss (first 20 epochs)", xlabel="Epoch", ylabel="Loss")
    ax_val.set(title="Validation Loss (first 20 epochs)", xlabel="Epoch")
    ax_tr.legend(frameon=True); ax_val.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    return out_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot BCE vs BCE+JSD loss curves.")
    parser.add_argument("--json-dir", type=Path,
                        help="Directory that directly contains the two JSON logs "
                             "(skips any search).")
    args = parser.parse_args(argv)

    json_paths = locate_jsons(args.json_dir)
    out_path = plot_and_save(json_paths)
    print(f"✅  Figure saved to: {out_path}")


# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as err:
        sys.exit(str(err))
