# utils/visualize.py
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import det_curve, RocCurveDisplay

def plot_metrics(results_dir, output_dir):
    """Plot accuracy/loss curves from JSON results"""
    epochs, train_acc, val_acc = [], [], []
    for fname in os.listdir(results_dir):
        if not fname.startswith("epoch_"): continue
        with open(os.path.join(results_dir, fname)) as f:
            data = json.load(f)
        epochs.append(data["epoch"])
        train_acc.append(data["train"]["acc"])
        val_acc.append(data["val"]["acc"])
    
    plt.figure()
    plt.plot(epochs, train_acc, label="Train")
    plt.plot(epochs, val_acc, label="Validation")
    plt.title("Accuracy vs Epoch")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "accuracy_curve.png"))

def plot_det_curves(pred_dir, output_dir):
    """Plot Detection Error Tradeoff (DET) curves"""
    plt.figure()
    for fname in os.listdir(pred_dir):
        epoch = fname.split("_")[1].split(".")[0]
        df = pd.read_csv(os.path.join(pred_dir, fname))
        fpr, fnr, _ = det_curve(df['labels'], df['probs'])
        plt.plot(fpr, fnr, label=f"Epoch {epoch}")
    
    plt.title("DET Curves")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "det_curves.png"))

def plot_attention_heatmaps(attn_dir, output_dir):
    """Visualize temporal attention weights"""
    for fname in os.listdir(attn_dir):
        epoch = fname.split("_")[1].split(".")[0]
        attn = np.load(os.path.join(attn_dir, fname))
        avg_attn = attn.mean(axis=0)  # Average across batches
        
        plt.figure(figsize=(10,3))
        sns.heatmap(avg_attn.T, cmap="viridis")
        plt.title(f"Attention Weights - Epoch {epoch}")
        plt.xlabel("Frames")
        plt.ylabel("Sequence Position")
        plt.savefig(os.path.join(output_dir, f"attn_epoch_{epoch}.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", default="outputs/pred_data")
    parser.add_argument("--attn_dir", default="outputs/attn")
    parser.add_argument("--results_dir", default="outputs/results")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    plot_metrics(args.results_dir, args.output_dir)
    plot_det_curves(args.pred_dir, args.output_dir)
    plot_attention_heatmaps(args.attn_dir, args.output_dir)