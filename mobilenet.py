# main.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision import models
from utils.dataset import DeepfakeDataset
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.utils import class_weight
import os
import time
from tqdm import tqdm
import random
import numpy as np
import json
from datetime import datetime

def set_seed(seed=42):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class DeepfakeModel(nn.Module):
    """Deepfake Detection Model using MobileNetV3-Small."""

    def __init__(self, dropout_rate=0.5):
        super(DeepfakeModel, self).__init__()
        self.mobilenet = models.mobilenet_v3_small(pretrained=True)
        # Replace the classifier with a new one for binary classification
        # MobileNetV3-Small classifier: Sequential(Linear, Hardswish, Dropout, Linear)
        # We'll replace it with Dropout and Linear to output a single logit
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.mobilenet.classifier[0].in_features, 1)  # Output single logit for binary classification
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, C, H, W)

        Returns:
            torch.Tensor: Output logits of shape (batch_size)
        """
        batch_size, seq_len, c, h, w = x.shape
        # Merge batch and sequence dimensions
        x = x.view(batch_size * seq_len, c, h, w)
        logits = self.mobilenet(x)
        # Aggregate logits over the sequence, e.g., by averaging
        logits = logits.view(batch_size, seq_len, -1)
        logits = logits.mean(dim=1)  # Simple aggregation
        return logits.squeeze(1)  # Shape: (batch_size)

def compute_metrics(labels, preds, probs):
    """
    Compute various metrics:
    Accuracy, F1 (for fake class), AUC, Recall (for fake),
    FRR (for real), GAR, and Precision (for fake).
    """
    accuracy = accuracy_score(labels, preds)

    # Check if we have both classes for AUC and F1
    if len(set(labels)) > 1:
        auc = roc_auc_score(labels, probs)
        f1 = f1_score(labels, preds)
    else:
        auc = 0.0
        f1 = 0.0

    # Confusion matrix values
    # Consider fake=1 as positive for these calculations
    TP_fake = np.sum((preds == 1) & (labels == 1))
    FN_fake = np.sum((preds == 0) & (labels == 1))
    FP_fake = np.sum((preds == 1) & (labels == 0))
    # Recall for fake class
    recall_fake = TP_fake / (TP_fake + FN_fake) if (TP_fake + FN_fake) > 0 else 0.0
    # Precision for fake class
    precision_fake = TP_fake / (TP_fake + FP_fake) if (TP_fake + FP_fake) > 0 else 0.0

    # For FRR and GAR, consider real=0 as "genuine"
    TP_real = np.sum((preds == 0) & (labels == 0))
    FN_real = np.sum((preds == 1) & (labels == 0))  # falsely rejecting real
    FRR = FN_real / (TP_real + FN_real) if (TP_real + FN_real) > 0 else 0.0
    GAR = 1 - FRR

    return accuracy, f1, auc, recall_fake, FRR, GAR, precision_fake

def train_epoch(model, dataloader, criterion, optimizer, device, grad_clip=5.0):
    """Train the model for one epoch."""
    model.train()
    epoch_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    with tqdm(total=len(dataloader), desc="Training", unit="batch") as pbar:
        for inputs, labels in dataloader:
            # Assuming inputs shape: (batch_size, seq_len, C, H, W)
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)  # Shape: (batch_size)
            labels = labels.float()

            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            preds = (probs > 0.5).astype(int)
            batch_labels_np = labels.cpu().numpy()

            # Compute batch metrics
            batch_acc, batch_f1, batch_auc, batch_recall, batch_frr, batch_gar, batch_precision = compute_metrics(
                batch_labels_np, preds, probs
            )

            all_labels.extend(batch_labels_np)
            all_preds.extend(preds)
            all_probs.extend(probs)

            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Acc': f"{batch_acc:.4f}",
                'F1': f"{batch_f1:.4f}",
                'AUC': f"{batch_auc:.4f}",
                'Recall': f"{batch_recall:.4f}",
                'FRR': f"{batch_frr:.4f}",
                'GAR': f"{batch_gar:.4f}",
                'Precision': f"{batch_precision:.4f}"
            })
            pbar.update(1)

    # Epoch-level metrics
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    epoch_acc, epoch_f1, epoch_auc, epoch_recall, epoch_frr, epoch_gar, epoch_precision = compute_metrics(
        all_labels, all_preds, all_probs
    )
    average_loss = epoch_loss / len(dataloader)
    return average_loss, epoch_acc, epoch_auc, epoch_f1, epoch_recall, epoch_frr, epoch_gar, epoch_precision

def evaluate_model(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    epoch_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        with tqdm(total=len(dataloader), desc="Evaluating", unit="batch") as pbar:
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)  # Shape: (batch_size)
                labels = labels.float()

                loss = criterion(outputs, labels)
                epoch_loss += loss.item()

                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                batch_labels_np = labels.cpu().numpy()

                # Compute batch metrics
                batch_acc, batch_f1, batch_auc, batch_recall, batch_frr, batch_gar, batch_precision = compute_metrics(
                    batch_labels_np, preds, probs
                )

                all_labels.extend(batch_labels_np)
                all_preds.extend(preds)
                all_probs.extend(probs)

                pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Acc': f"{batch_acc:.4f}",
                    'F1': f"{batch_f1:.4f}",
                    'AUC': f"{batch_auc:.4f}",
                    'Recall': f"{batch_recall:.4f}",
                    'FRR': f"{batch_frr:.4f}",
                    'GAR': f"{batch_gar:.4f}",
                    'Precision': f"{batch_precision:.4f}"
                })
                pbar.update(1)

    # Epoch-level metrics
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    val_acc, val_f1, val_auc, val_recall, val_frr, val_gar, val_precision = compute_metrics(
        all_labels, all_preds, all_probs
    )
    average_loss = epoch_loss / len(dataloader)
    return average_loss, val_acc, val_auc, val_f1, val_recall, val_frr, val_gar, val_precision

def save_model_and_result(model, results, model_path, results_path):
    """
    Save model state and results to disk.

    Args:
        model (torch.nn.Module): Trained model.
        results (dict): Evaluation results.
        model_path (str): Path to save the model state.
        results_path (str): Path to save the results JSON.
    """
    # Ensure directories exist
    model_dir = os.path.dirname(model_path)
    results_dir = os.path.dirname(results_path)

    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if results_dir and not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Save Model
    try:
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

    # Save Results in a JSON file
    try:
        with open(results_path, 'w') as file:
            json.dump(results, file, indent=4)
        print(f"Results saved to {results_path}")
    except Exception as e:
        print(f"Error saving results: {e}")

def get_unique_filename(base_dir, prefix, extension, model_name):
    """Generate a unique filename using the current timestamp and model name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{model_name}_{timestamp}.{extension}"
    return os.path.join(base_dir, filename)

def main():
    """Main function to train and evaluate the deepfake detection model."""
    # Check if preprocessed data exists
    preprocessed_dir = "data/preprocessed/"
    required_subdirs = ["Celeb-real", "Celeb-synthesis", "Youtube-real"]
    missing_subdirs = [subdir for subdir in required_subdirs if not os.path.exists(os.path.join(preprocessed_dir, subdir))]

    if missing_subdirs:
        print(f"Error: Preprocessed directories missing: {missing_subdirs}. Please run the preprocessing script first.")
        print("Proceeding to process with available data.")
    else:
        print("Preprocessed data found. Proceeding to training.")

    dropout_rate = 0.75
    model = DeepfakeModel(dropout_rate=dropout_rate).to(device)

    transform = Compose([
        Resize((224, 224)),
        ToTensor()
    ])

    labels_file = "data/List_of_testing_videos.txt"
    if not os.path.exists(labels_file):
        print(f"Error: Labels file not found at {labels_file}")
        return

    dataset = DeepfakeDataset(
        root_dir="data/preprocessed/",
        labels_file=labels_file,
        transform=transform,
        limit=3000,  # Adjust limit as needed
        seq_len=1  # Since we're no longer using sequences
    )

    if len(dataset) == 0:
        print("Error: No valid samples found in the dataset.")
        return
    else:
        print(f"Number of samples in the dataset: {len(dataset)}")

    labels = [label for _, label in dataset]
    unique, counts = torch.unique(torch.tensor(labels), return_counts=True)
    class_counts = dict(zip(unique.tolist(), counts.tolist()))
    print(f"Class distribution: {class_counts}")

    pos_weight = None
    model_name = "mobilenet_v3_small"  # Updated model name
    if len(class_counts) == 2:
        classes = np.array(list(class_counts.keys()))
        labels_np = np.array(labels)
        class_weights = class_weight.compute_class_weight('balanced', classes=classes, y=labels_np)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        pos_weight = class_weights[1] / (class_weights[0] if class_weights[0] != 0 else 1.0)
        print(f"Class Weights: {class_weights.tolist()} | Pos Weight: {pos_weight:.4f}")
    else:
        print("Not a binary classification scenario or class count issue. Using default weights.")

    criterion = nn.BCEWithLogitsLoss(pos_weight=(torch.tensor(pos_weight).to(device) if pos_weight is not None else None))

    test_split = 0.3
    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Data Loaders
    batch_size = 32  # Batch Size [Default is 32]

    # Load Train Data (70%)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=3,
        pin_memory=True,
        drop_last=True
    )

    # Load Test Data (30%)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3,
        verbose=True
    )

    num_epochs = 20  # Number of Epochs [Default is 20]
    best_auc = 0.0

    print("Starting training...")
    for epoch in range(num_epochs):
        start_time = time.time()

        # Epoch Training
        train_loss, train_acc, train_auc, train_f1, train_recall, train_frr, train_gar, train_precision = train_epoch(
            model, train_dataloader, criterion, optimizer, device, grad_clip=5.0
        )

        # Epoch Evaluation 
        val_loss, val_acc, val_auc, val_f1, val_recall, val_frr, val_gar, val_precision = evaluate_model(
            model, test_dataloader, criterion, device
        )

        scheduler.step(val_auc)

        epoch_time = time.time() - start_time
        
        # Generate unique filenames using timestamp and model name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"epoch-{epoch+1}_{model_name}_{timestamp}.pt"
        results_filename = f"epoch-{epoch+1}_{model_name}_results_{timestamp}.json"
        model_path = os.path.join("outputs", "models", model_filename)
        results_path = os.path.join("outputs", "results", results_filename)

        results = {
            'Epoch': epoch + 1,  # Current epoch number
            'Training': {
                'Training Loss': train_loss,  
                'Training Accuracy': train_acc, 
                'Training AUC': train_auc, 
                'Training F1-Score': train_f1,  
                'Training Recall': train_recall,  
                'Training FRR': train_frr,  
                'Training GAR': train_gar,  
                'Training Precision': train_precision  
            },
            'Testing': {
                'Val Loss': val_loss,  
                'Val Accuracy': val_acc,  
                'Val AUC': val_auc,  
                'Val F1-Score': val_f1, 
                'Val Recall': val_recall, 
                'Val FRR': val_frr,  
                'Val GAR': val_gar, 
                'Val Precision': val_precision  
            },
            'Epoch Time': epoch_time  # Duration of the epoch in seconds
        }

        save_model_and_result(
            model, 
            results, 
            model_path=model_path, 
            results_path=results_path
        )

        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train AUC: {train_auc:.4f} | Train F1: {train_f1:.4f} | Train Recall: {train_recall:.4f} | Train FRR: {train_frr:.4f} | Train GAR: {train_gar:.4f} | Train Precision: {train_precision:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f} | Val Recall: {val_recall:.4f} | Val FRR: {val_frr:.4f} | Val GAR: {val_gar:.4f} | Val Precision: {val_precision:.4f}")
        print(f"Epoch Time: {epoch_time/60:.2f} minutes")

        if val_auc > best_auc:
            best_auc = val_auc
            best_model_filename = f"best_{model_name}_model_{timestamp}.pt"
            best_model_path = os.path.join("outputs", "models", best_model_filename)
            os.makedirs("outputs/models", exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
            print("Best model updated and saved.")

    # Save the final model with a unique name
    final_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_filename = f"deepfake_model_final_{model_name}_{final_timestamp}.pt"
    final_model_path = os.path.join("outputs", "models", final_model_filename)
    os.makedirs("outputs/models", exist_ok=True)
    torch.save(model.state_dict(), final_model_path)
    print(f"Training completed. Final model saved at {final_model_path}")

if __name__ == "__main__":
    main()
