import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, RandomHorizontalFlip
from datetime import datetime
import os
import time
import json
from tqdm import tqdm
import numpy as np
import random
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from torchvision.models import mobilenet_v3_small

# Import your custom dataset module.
from utils.dataset import DeepfakeDataset  # Your dataset class.

# ---------------------------
# Utility Functions and Setup
# ---------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Base output directory for saving models and results.
drive_output_dir = "/content/drive/MyDrive/Deepfake-Thesis/MobileNetV3-Small"
os.makedirs(os.path.join(drive_output_dir, "models"), exist_ok=True)
os.makedirs(os.path.join(drive_output_dir, "results"), exist_ok=True)

# Get current date in YYYYMMDD format for filenames.
current_date = datetime.now().strftime("%Y%m%d")

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
# Model Definition (MobileNetV3 Small)
# ---------------------------
class MobileNetModel(nn.Module):
    def __init__(self):
        """
        MobileNetV3 Small based model for binary classification.
        The pretrained MobileNetV3 Small is loaded from torchvision.
        Regardless of how many frames are provided per sample, only the first frame is used.
        
        Modifications:
         - Replaces the default classifier with one that includes a dropout layer.
         - Adjusts the final linear layer to accept the correct number of input features (576) as inferred from the model.
        """
        super(MobileNetModel, self).__init__()
        # Load MobileNetV3 Small pretrained model.
        self.mobilenet = mobilenet_v3_small(pretrained=True)
        # The default classifier expects features of size 1024,
        # but your network appears to output features of size 576.
        # To check, you can run a dummy input through the mobilenet.
        # Here we set the in_features accordingly.
        in_features = 576  
        
        # Replace the classifier with our binary classification head.
        # We insert a dropout layer to reduce overfitting.
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(0.5),  # Dropout to reduce overfitting.
            nn.Linear(in_features, 1)
        )

    def forward(self, x):
        # x is expected to be of shape: [B, seq_len, C, H, W].
        # We use only the first frame from each sample.
        first_frame = x[:, 0, :, :, :]  # [B, C, H, W]
        logits = self.mobilenet(first_frame).view(-1)  # [B]
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
    epoch_acc, epoch_f1, epoch_auc, epoch_recall, epoch_frr, epoch_gar, epoch_precision = compute_metrics(
        all_labels, all_preds, all_probs
    )
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
    val_acc, val_f1, val_auc, val_recall, val_frr, val_gar, val_precision = compute_metrics(
        all_labels, all_preds, all_probs
    )
    average_loss = epoch_loss / len(dataloader)
    torch.cuda.empty_cache()
    return average_loss, val_acc, val_auc, val_f1, val_recall, val_frr, val_gar, val_precision

def save_model_and_result(model, results, model_filename, results_filename):
    base_model_name = os.path.splitext(os.path.basename(model_filename))[0]
    base_results_name = os.path.splitext(os.path.basename(results_filename))[0]
    model_filename = f"{base_model_name}_{current_date}.pt"
    results_filename = f"{base_results_name}_{current_date}.json"
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

# ---------------------------
# Main Training Loop
# ---------------------------
def main():
    model = MobileNetModel().to(device)
    
    # Define image transformations.
    # Additional RandomHorizontalFlip is added to help mitigate overfitting by slightly increasing input variance.
    transform = Compose([
        Resize((224, 224)),
        RandomHorizontalFlip(p=0.5),
        ToTensor()
    ])
    
    # Data directory.
    data_root_dir = "/content/Deepfake-Thesis/data/Final-data"
    train_dataset = DeepfakeDataset(
        root_dir=os.path.join(data_root_dir, "Training"),
        transform=transform,
        seq_len=40  # Even if multiple frames are provided, only the first frame is used.
    )
    test_dataset = DeepfakeDataset(
        root_dir=os.path.join(data_root_dir, "Testing"),
        transform=transform,
        seq_len=40
    )
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    # Compute positive weight for handling class imbalance.
    train_labels = [sample[0] for sample in train_dataset.labels]
    num_pos = sum(train_labels)
    num_neg = len(train_labels) - num_pos
    ratio = num_neg / num_pos if num_pos > 0 else 1.0
    pos_weight = torch.tensor([ratio], dtype=torch.float32).to(device)
    print(f"Computed pos_weight (from training set): {pos_weight.item():.4f}")
    
    # Use plain BCEWithLogitsLoss.
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    batch_size = 16
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    # Using a slightly higher weight decay and a lower learning rate to slow convergence.
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    num_epochs = 20
    best_auc = 0.0
    all_epoch_results = []  # To store results for each epoch.
    
    print("Starting training...")
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss, train_acc, train_auc, train_f1, train_recall, train_frr, train_gar, train_precision = train_epoch(
            model, train_dataloader, criterion, optimizer, device, grad_clip=5.0
        )
        val_loss, val_acc, val_auc, val_f1, val_recall, val_frr, val_gar, val_precision = evaluate_model(
            model, test_dataloader, criterion, device
        )
        scheduler.step(val_loss)
        epoch_time = time.time() - start_time
        
        epoch_result = {
            'Epoch': epoch + 1,
            'Training': {
                'Loss': train_loss,
                'Accuracy': train_acc,
                'AUC': train_auc,
                'F1-Score': train_f1,
                'Recall': train_recall,
                'FRR': train_frr,
                'GAR': train_gar,
                'Precision': train_precision
            },
            'Testing': {
                'Loss': val_loss,
                'Accuracy': val_acc,
                'AUC': val_auc,
                'F1-Score': val_f1,
                'Recall': val_recall,
                'FRR': val_frr,
                'GAR': val_gar,
                'Precision': val_precision
            },
            'Epoch Time': epoch_time
        }
        all_epoch_results.append(epoch_result)
        
        save_model_and_result(
            model, 
            epoch_result, 
            model_filename=f"epoch-{epoch+1}-mobilenetv3small-alone.pt", 
            results_filename=f"epoch-{epoch+1}-mobilenetv3small-alone.json"
        )
        
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train AUC: {train_auc:.4f} | "
              f"Train F1: {train_f1:.4f} | Train Recall: {train_recall:.4f} | Train FRR: {train_frr:.4f} | "
              f"Train GAR: {train_gar:.4f} | Train Precision: {train_precision:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f} | "
              f"Val Recall: {val_recall:.4f} | Val FRR: {val_frr:.4f} | Val GAR: {val_gar:.4f} | Val Precision: {val_precision:.4f}")
        print(f"Epoch Time: {epoch_time / 60:.2f} minutes")
        
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_path = os.path.join(drive_output_dir, "models", f"best_mobilenetv3small_model_epoch_{epoch+1}_{current_date}.pt")
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model updated and saved to {best_model_path}")
    
    final_model_path = os.path.join(drive_output_dir, "models", f"deepfake_model_final_{current_date}.pt")
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved at {final_model_path}")
    
    results_filename = f"all_epoch_results_{current_date}.json"
    results_path = os.path.join(drive_output_dir, "results", results_filename)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    try:
        with open(results_path, "w") as f:
            json.dump(all_epoch_results, f, indent=4)
        print(f"All epoch results saved to {results_path}")
    except Exception as e:
        print(f"Error saving all epoch results: {e}")
    
    print("Training completed.")

if __name__ == "__main__":
    main()
