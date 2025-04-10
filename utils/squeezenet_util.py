import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from datetime import datetime
import os
import time
import json
from tqdm import tqdm
import numpy as np
import random
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from torchvision.models import squeezenet1_1

# Import your dataset class.
from utils.dataset import DeepfakeDataset

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_metrics(labels, preds, probs):
    """
    Computes the following metrics:
      - Accuracy
      - F1-score
      - AUC
      - Recall (for fake samples)
      - False Rejection Rate (FRR)
      - Genuine Acceptance Rate (GAR)
      - Precision (for fake samples)
    """
    accuracy = accuracy_score(labels, preds)
    if len(set(labels)) > 1:
        auc = roc_auc_score(labels, probs)
        f1 = f1_score(labels, preds)
    else:
        auc = 0.0
        f1 = 0.0

    TP_fake = np.sum((preds == 1) & (labels == 1))
    FN_fake = np.sum((preds == 0) & (labels == 1))
    FP_fake = np.sum((preds == 1) & (labels == 0))
    recall_fake = TP_fake / (TP_fake + FN_fake) if (TP_fake + FN_fake) > 0 else 0.0
    precision_fake = TP_fake / (TP_fake + FP_fake) if (TP_fake + FP_fake) > 0 else 0.0

    TP_real = np.sum((preds == 0) & (labels == 0))
    FN_real = np.sum((preds == 1) & (labels == 0))
    FRR = FN_real / (TP_real + FN_real) if (TP_real + FN_real) > 0 else 0.0
    GAR = 1 - FRR

    return accuracy, f1, auc, recall_fake, FRR, GAR, precision_fake

# ---------------------------
# Model Definition
# ---------------------------
class SqueezeNetModel(nn.Module):
    def __init__(self):
        """
        SqueezeNet1_1-based model for binary classification.
        The pretrained SqueezeNet1_1 is modified so that the classifier outputs a single logit.
        Only the first frame from each sample is used.
        """
        super(SqueezeNetModel, self).__init__()
        self.squeezenet = squeezenet1_1(pretrained=True)
        # Replace the Conv2d in the classifier to output 1 channel.
        self.squeezenet.classifier[1] = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
        # Optionally remove the ReLU activation by using Identity.
        self.squeezenet.classifier[2] = nn.Identity()

    def forward(self, x):
        # x is expected to be of shape: [B, seq_len, C, H, W].
        # We use only the first frame.
        first_frame = x[:, 0, :, :, :]
        out = self.squeezenet(first_frame)  # Expected shape: [B, 1, 1, 1]
        logits = out.view(-1)  # Flatten to [B]
        return logits

# ---------------------------
# Training and Evaluation Functions (with AMP)
# ---------------------------
def train_epoch(model, dataloader, criterion, optimizer, device, grad_clip=5.0):
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    epoch_loss = 0.0
    all_labels = []
    all_probs = []
    with tqdm(total=len(dataloader), desc="Training", unit="batch") as pbar:
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                outputs = outputs.view(-1)
                labels = labels.float().view(-1)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
            pbar.update(1)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = (all_probs > 0.5).astype(int)
    epoch_acc, epoch_f1, epoch_auc, epoch_recall, epoch_frr, epoch_gar, epoch_precision = compute_metrics(all_labels, all_preds, all_probs)
    average_loss = epoch_loss / len(dataloader)
    torch.cuda.empty_cache()
    return average_loss, epoch_acc, epoch_auc, epoch_f1, epoch_recall, epoch_frr, epoch_gar, epoch_precision

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0.0
    all_labels = []
    all_probs = []
    with torch.no_grad():
        with tqdm(total=len(dataloader), desc="Evaluating", unit="batch") as pbar:
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    outputs = outputs.view(-1)
                    labels = labels.float().view(-1)
                    loss = criterion(outputs, labels)
                epoch_loss += loss.item()
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs)
                pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
                pbar.update(1)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = (all_probs > 0.5).astype(int)
    val_acc, val_f1, val_auc, val_recall, val_frr, val_gar, val_precision = compute_metrics(all_labels, all_preds, all_probs)
    average_loss = epoch_loss / len(dataloader)
    torch.cuda.empty_cache()
    return average_loss, val_acc, val_auc, val_f1, val_recall, val_frr, val_gar, val_precision

def save_model_and_result(model, results, model_filename, results_filename):
    base_model_name = os.path.splitext(os.path.basename(model_filename))[0]
    base_results_name = os.path.splitext(os.path.basename(results_filename))[0]
    model_filename = f"{base_model_name}_{datetime.now().strftime('%Y%m%d')}.pt"
    results_filename = f"{base_results_name}_{datetime.now().strftime('%Y%m%d')}.json"
    model_path = os.path.join(drive_output_dir, "models", model_filename)
    results_path = os.path.join(drive_output_dir, "results", results_filename)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    try:
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        pth_model_path = model_path.replace(".pt", ".pth")
        torch.save(model.state_dict(), pth_model_path)
        print(f"Model also saved to {pth_model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")
    try:
        with open(results_path, 'w') as file:
            json.dump(results, file, indent=4)
        print(f"Results saved to {results_path}")
    except Exception as e:
        print(f"Error saving results: {e}")
