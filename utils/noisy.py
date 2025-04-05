import copy
import random
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils.dataset import DeepfakeDataset
from main import evaluate_model  # Ensure your PYTHONPATH includes project root
from utils.losses import CombinedLoss
import torch

def test_noise_robustness(clean_model, dataset, device, noise_levels=[0.1, 0.3, 0.5]):
    criterion = CombinedLoss(bce_weight=0.5, jsd_weight=0.5,
                             pos_weight=torch.tensor([(len(dataset)//2)/ (len(dataset)//2)]).to(device))
    results = []
    for noise in noise_levels:
        noisy_ds = copy.deepcopy(dataset)
        # Corrupt labels
        for i in range(len(noisy_ds.labels)):
            label, rp, cid, total = noisy_ds.labels[i]
            if random.random() < noise:
                noisy_ds.labels[i] = (1 - label, rp, cid, total)

        loader = DataLoader(noisy_ds, batch_size=32, shuffle=False,
                            num_workers=2, pin_memory=True, drop_last=True)
        metrics = evaluate_model(clean_model, loader, criterion, device)
        # metrics: (loss, acc, f1, auc, rec, FRR, GAR, prec, eer)
        results.append({'noise': noise, 'acc': metrics[1]})

    # Plot
    noise_vals = [r['noise'] for r in results]
    accs = [r['acc'] for r in results]
    plt.plot(noise_vals, accs, marker='o')
    plt.xlabel("Noise Level")
    plt.ylabel("Accuracy")
    plt.title("Noise Robustness")
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/noise_robustness.png")
    plt.close()
    return results
