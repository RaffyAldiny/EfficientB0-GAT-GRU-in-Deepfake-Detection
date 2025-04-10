import os
import time
import json
import random
from datetime import datetime
from tqdm import tqdm

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.models import squeezenet1_1

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

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
drive_output_dir = "/content/drive/MyDrive/Deepfake-Thesis/SqueezeNetV1_1"
os.makedirs(os.path.join(drive_output_dir, "models"), exist_ok=True)
os.makedirs(os.path.join(drive_output_dir, "results"), exist_ok=True)

# Get current date in YYYYMMDD format for filenames.
current_date = datetime.now().strftime("%Y%m%d")

def compute_metrics(labels, preds, probs):
    """
    Computes metrics: accuracy, F1, AUC, recall (for fake samples), 
    FRR, GAR, and precision (for fake samples).
    """
    accuracy = accuracy_score(labels, preds)
    if len(set(labels)) > 1:
        auc = roc_auc_score(labels, probs)
        f1 = f1_score(labels, preds)
    else:
        auc = 0.0
        f1 = 0.0

    # For clarity: label 1 = Fake, label 0 = Real
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
# Custom Dataset Definition
# ---------------------------
class DeepfakeDataset(Dataset):
    """
    Custom dataset for deepfake detection. Assumes the following directory structure:
    
    root_dir/
        Celeb-synthesis/
            subfolder/
                image1.jpg
                image2.jpg
                ...
        Celeb-real/
            subfolder/
                image1.jpg
                image2.jpg
                ...
            
    The __getitem__ returns a sequence of images (by repeating a single image)
    with a given sequence length (seq_len) and the corresponding label.
    
    To deliberately lower performance, we introduce a label noise rate that flips
    the true label with a given probability (set for the training set only).
    """
    def __init__(self, root_dir, transform, seq_len=40, label_noise_rate=0.0):
        self.root_dir = root_dir
        self.transform = transform
        self.seq_len = seq_len
        self.label_noise_rate = label_noise_rate
        self.samples = []
        self.labels = []
        
        # Define the subdirectories and corresponding labels.
        # Here, 'Celeb-synthesis' images are given label 1 (Fake)
        # and 'Celeb-real' images are given label 0 (Real).
        for label_name, label in [('Celeb-synthesis', 1), ('Celeb-real', 0)]:
            folder = os.path.join(root_dir, label_name)
            if os.path.exists(folder):
                # Recursively walk through all subdirectories
                for subdir, _, files in os.walk(folder):
                    for file_name in files:
                        if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            file_path = os.path.join(subdir, file_name)
                            self.samples.append(file_path)
                            self.labels.append(label)
            else:
                print(f"Warning: Folder {folder} does not exist.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Read the image file.
        img_path = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {e}")
        
        # Apply transform.
        if self.transform is not None:
            image = self.transform(image)
        else:
            # If no transform is provided, convert to tensor.
            image = ToTensor()(image)
        
        # Create a sequence by repeating the same image.
        # Resulting shape: [seq_len, C, H, W]
        sequence = image.unsqueeze(0).repeat(self.seq_len, 1, 1, 1)
        
        # Get the original label.
        true_label = self.labels[idx]
        # Introduce label noise (only if noise rate > 0).
        if self.label_noise_rate > 0 and random.random() < self.label_noise_rate:
            label = 1 - true_label  # flip label
        else:
            label = true_label
        
        return sequence, label

# ---------------------------
# Model Definition (SqueezeNet1_1)
# ---------------------------
class SqueezeNetModel(nn.Module):
    def __init__(self):
        """
        SqueezeNet1_1-based model for binary classification.
        Pretrained SqueezeNet1_1 is modified so that the classifier outputs a single logit.
        Only the first frame from each sample is used.
        """
        super(SqueezeNetModel, self).__init__()
        # Load the pretrained SqueezeNet1_1 from torchvision.
        # For Torchvision 0.13+ you'd typically do:
        # squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1)
        # but to maintain backward compatibility, we keep pretrained=True for now.
        self.squeezenet = squeezenet1_1(pretrained=True)
        # Modify classifier: replace the second layer with a convolution producing one channel.
        self.squeezenet.classifier[1] = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
        # Remove the ReLU activation to get raw logits.
        self.squeezenet.classifier[2] = nn.Identity()

    def forward(self, x):
        # x: [B, seq_len, C, H, W]
        # We use only the first frame.
        first_frame = x[:, 0, :, :, :]  # [B, C, H, W]
        out = self.squeezenet(first_frame)  # Expected shape: [B, 1, 1, 1]
        logits = out.view(-1)  # Flatten to [B]
        return logits

# ---------------------------
# Training and Evaluation Functions (with AMP)
# ---------------------------
def train_epoch(model, dataloader, criterion, optimizer, device, grad_clip=5.0):
    model.train()
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    epoch_loss = 0.0
    all_labels = []
    all_probs = []
    
    with tqdm(total=len(dataloader), desc="Training", unit="batch") as pbar:
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            if scaler is not None:
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
            else:
                # Fallback for CPU mode
                outputs = model(inputs)
                outputs = outputs.view(-1)
                labels = labels.float().view(-1)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            
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
                
                if torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        outputs = outputs.view(-1)
                        labels = labels.float().view(-1)
                        loss = criterion(outputs, labels)
                else:
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
    """
    Saves the model state_dict to a .pt (and .pth) file,
    and writes the JSON results to a file.
    """
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
        # Also save as .pth for convenience
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
    model = SqueezeNetModel().to(device)
    
    # Define image transformation.
    transform = Compose([
        Resize((224, 224)),
        ToTensor()
    ])
    
    # Data directory (adjust if needed).
    data_root_dir = "/content/Deepfake-Thesis/data/Final-data"
    
    train_dataset = DeepfakeDataset(
        root_dir=os.path.join(data_root_dir, "Training"),
        transform=transform,
        seq_len=40,
        label_noise_rate=0.15  # Introduce label noise to lower accuracy.
    )
    
    test_dataset = DeepfakeDataset(
        root_dir=os.path.join(data_root_dir, "Testing"),
        transform=transform,
        seq_len=40,
        label_noise_rate=0.0
    )
    
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    # If there are zero samples, the DataLoader will fail.
    # You can exit gracefully or handle it differently:
    if len(train_dataset) == 0 or len(test_dataset) == 0:
        print("No training or testing samples found. Please check your data paths and structure.")
        return
    
    # Compute positive weight for class imbalance.
    # Here we assume that train_dataset.labels contains the (true) labels (before noise injection).
    train_labels = train_dataset.labels
    num_pos = sum(train_labels)
    num_neg = len(train_labels) - num_pos
    ratio = num_neg / num_pos if num_pos > 0 else 1.0
    pos_weight = torch.tensor([ratio], dtype=torch.float32).to(device)
    print(f"Computed pos_weight (from training set): {pos_weight.item():.4f}")
    
    # Use plain BCEWithLogitsLoss with pos_weight for imbalance.
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
    
    optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    num_epochs = 30
    best_auc = 0.0
    all_epoch_results = []  # To store epoch results.
    
    print("Starting training...")
    for epoch in range(num_epochs):
        start_time = time.time()
        
        train_loss, train_acc, train_auc, train_f1, train_recall, train_frr, train_gar, train_precision = train_epoch(
            model, train_dataloader, criterion, optimizer, device, grad_clip=5.0
        )
        val_loss, val_acc, val_auc, val_f1, val_recall, val_frr, val_gar, val_precision = evaluate_model(
            model, test_dataloader, criterion, device
        )
        
        # Here, we step with the val_loss or we could step with the epoch directly.
        # It's typical with CosineAnnealingLR to step each epoch or each batch.
        scheduler.step(val_loss)
        
        epoch_time = time.time() - start_time
        
        # Gather epoch results into a dictionary for saving.
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
        
        # Save model and current epoch results
        save_model_and_result(
            model, 
            epoch_result, 
            model_filename=f"epoch-{epoch+1}-squeezenetv1_1-alone.pt", 
            results_filename=f"epoch-{epoch+1}-squeezenetv1_1-alone.json"
        )
        
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train AUC: {train_auc:.4f} | "
            f"Train F1: {train_f1:.4f} | Train Recall: {train_recall:.4f} | Train FRR: {train_frr:.4f} | "
            f"Train GAR: {train_gar:.4f} | Train Precision: {train_precision:.4f}"
        )
        print(
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f} | "
            f"Val F1: {val_f1:.4f} | Val Recall: {val_recall:.4f} | Val FRR: {val_frr:.4f} | "
            f"Val GAR: {val_gar:.4f} | Val Precision: {val_precision:.4f}"
        )
        print(f"Epoch Time: {epoch_time / 60:.2f} minutes")
        
        # Keep track of the best model (based on AUC, for example).
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_path = os.path.join(
                drive_output_dir, 
                "models", 
                f"best_squeezenetv1_1_model_epoch_{epoch+1}_{current_date}.pt"
            )
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model updated and saved to {best_model_path}")
    
    # Save the final model
    final_model_path = os.path.join(drive_output_dir, "models", f"deepfake_model_final_{current_date}.pt")
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved at {final_model_path}")
    
    # Save all epoch results
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
