import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, ToTensor, Resize
from models.efficientnet import get_efficientnet
from models.gat import GAT
from models.gru import GRU
from utils.dataset import DeepfakeDataset
from utils.losses import CombinedLoss  # Use the revised losses code
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import GroupShuffleSplit  # For group-aware splitting
import os
import time
from tqdm import tqdm
import random
import numpy as np
import json

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

def create_batched_edge_index(base_edge_index, batch_size, num_nodes, device):
    """
    Create a batched edge index for the graph attention network dynamically
    based on the current batch size.
    
    Args:
        base_edge_index (torch.Tensor): Edge index for a single sequence (chain graph).
        batch_size (int): Actual batch size.
        num_nodes (int): Number of nodes per sequence (seq_len).
        device (torch.device): Device to place the tensor.
        
    Returns:
        torch.Tensor: Batched edge index.
    """
    edge_index = base_edge_index.clone()
    edge_index = edge_index.repeat(1, batch_size)
    offsets = torch.arange(batch_size, device=device) * num_nodes
    num_edges_per_sample = base_edge_index.size(1)
    offsets = offsets.unsqueeze(0).repeat(2, num_edges_per_sample)
    edge_index += offsets
    return edge_index

def create_chain_graph(seq_len: int) -> torch.Tensor:
    """
    Dynamically generate a chain graph edge index for a given sequence length.
    Each node is connected bidirectionally to its immediate neighbor.
    
    Returns:
        torch.Tensor: Edge index of shape (2, num_edges).
    """
    edge_list = []
    for i in range(seq_len - 1):
        edge_list.extend([[i, i+1], [i+1, i]])
    return torch.tensor(edge_list, dtype=torch.long).t().contiguous()

class DeepfakeModel(nn.Module):
    """
    Deepfake Detection Model combining EfficientNet, GAT, GRU, and Temporal Attention.
    
    Improvements:
      - A temporal attention layer is added to weight all GRU timesteps.
      - Dropout rate reduced to 0.3 to mitigate underfitting.
    """
    def __init__(self, seq_len=40, dropout_rate=0.3):
        super(DeepfakeModel, self).__init__()
        self.seq_len = seq_len
        self.efficientnet = get_efficientnet()
        # Projection: from 1280 to 256.
        self.projection = nn.Linear(1280, 256)
        self.gat = GAT(in_channels=256, out_channels=8, heads=1)
        self.gru = GRU(input_size=8, hidden_size=32, num_layers=1, dropout=dropout_rate)
        # Temporal attention: maps GRU hidden states (32) to a scalar score.
        self.attention = nn.Linear(32, 1)
        self.fc = nn.Linear(32, 1)

    def forward(self, x, batched_edge_index):
        batch_size, seq_len, c, h, w = x.shape
        # Extract spatial features.
        x = x.view(batch_size * seq_len, c, h, w)
        spatial_features = self.efficientnet(x).squeeze(-1).squeeze(-1)
        projected_features = self.projection(spatial_features)
        # Process through GAT.
        gat_output = self.gat(projected_features, batched_edge_index)
        gat_output = gat_output.view(batch_size, seq_len, -1)
        # Process temporal sequence with GRU.
        gru_output = self.gru(gat_output)  # Shape: [batch_size, seq_len, 32]
        # Apply temporal attention.
        attn_scores = self.attention(gru_output)          # [batch_size, seq_len, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)     # Normalize over timesteps.
        weighted_output = torch.sum(gru_output * attn_weights, dim=1)  # [batch_size, 32]
        output = self.fc(weighted_output)
        return output

def compute_metrics(labels, preds, probs):
    """
    Compute metrics: Accuracy, F1 (for fake class), AUC, Recall (for fake),
    FRR (for real), GAR, and Precision (for fake).
    
    Metrics are computed globally over the epoch.
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

def train_epoch(model, dataloader, criterion, optimizer, device, base_edge_index, seq_len, grad_clip=5.0):
    """
    Train the model for one epoch.
    
    Improvement:
      - Collect predictions and labels across the entire epoch and compute metrics globally.
    """
    model.train()
    epoch_loss = 0.0
    all_labels = []
    all_probs = []  # Collect raw labels and probabilities.

    with tqdm(total=len(dataloader), desc="Training", unit="batch") as pbar:
        for inputs, labels in dataloader:
            current_batch_size = inputs.size(0)
            batched_edge_index = create_batched_edge_index(base_edge_index, current_batch_size, seq_len, device)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, batched_edge_index)
            outputs = outputs.view(-1)
            labels = labels.float().view(-1)

            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            batch_labels_np = labels.cpu().numpy()

            all_labels.extend(batch_labels_np)
            all_probs.extend(probs)

            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
            pbar.update(1)

    # Compute epoch-level metrics.
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = (all_probs > 0.5).astype(int)
    epoch_acc, epoch_f1, epoch_auc, epoch_recall, epoch_frr, epoch_gar, epoch_precision = compute_metrics(
        all_labels, all_preds, all_probs
    )
    average_loss = epoch_loss / len(dataloader)
    return average_loss, epoch_acc, epoch_auc, epoch_f1, epoch_recall, epoch_frr, epoch_gar, epoch_precision

def evaluate_model(model, dataloader, criterion, device, base_edge_index, seq_len):
    """
    Evaluate the model on the validation set.
    
    Improvement:
      - Collect predictions and labels globally and compute metrics once.
    """
    model.eval()
    epoch_loss = 0.0
    all_labels = []
    all_probs = []  # Collect raw labels and probabilities.

    with torch.no_grad():
        with tqdm(total=len(dataloader), desc="Evaluating", unit="batch") as pbar:
            for inputs, labels in dataloader:
                current_batch_size = inputs.size(0)
                batched_edge_index = create_batched_edge_index(base_edge_index, current_batch_size, seq_len, device)
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs, batched_edge_index)
                outputs = outputs.view(-1)
                labels = labels.float().view(-1)

                loss = criterion(outputs, labels)
                epoch_loss += loss.item()

                probs = torch.sigmoid(outputs).cpu().numpy()
                batch_labels_np = labels.cpu().numpy()

                all_labels.extend(batch_labels_np)
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
    return average_loss, val_acc, val_auc, val_f1, val_recall, val_frr, val_gar, val_precision

def save_model_and_result(model, results, model_path, results_path):
    """
    Save model state and training results to disk.
    Additionally, save the model as both .pt and .pth files.
    """
    model_dir = os.path.dirname(model_path)
    results_dir = os.path.dirname(results_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if results_dir and not os.path.exists(results_dir):
        os.makedirs(results_dir)

    try:
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        # Also save as a .pth file.
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

def main():
    """Main function to train and evaluate the deepfake detection model."""
    preprocessed_dir = "data/preprocessed/"
    required_subdirs = ["Celeb-real", "Celeb-synthesis", "Youtube-real"]
    missing_subdirs = [sub for sub in required_subdirs if not os.path.exists(os.path.join(preprocessed_dir, sub))]
    if missing_subdirs:
        print(f"Error: Preprocessed directories missing: {missing_subdirs}. Please run the preprocessing script first.")
        print("Missing directory but proceeding to process")
    else:
        print("Preprocessed data found. Proceeding to training.")

    seq_len = 40
    dropout_rate = 0.3  # Reduced dropout for less aggressive regularization.
    model = DeepfakeModel(seq_len=seq_len, dropout_rate=dropout_rate).to(device)

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
        limit=3000,
        seq_len=seq_len
    )
    if len(dataset) == 0:
        print("Error: No valid samples found in the dataset.")
        return
    else:
        print(f"Number of samples in the dataset: {len(dataset)}")

    # Use GroupShuffleSplit for group-aware splitting based on celebrity IDs.
    from sklearn.model_selection import GroupShuffleSplit
    groups = [item[2] for item in dataset.labels]  # celeb_id is the third element.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(gss.split(dataset, groups=groups))
    
    # Verify no overlapping groups between train and test.
    train_celebs = {dataset.labels[i][2] for i in train_idx}
    test_celebs = {dataset.labels[i][2] for i in test_idx}
    assert len(train_celebs & test_celebs) == 0, "Data leakage detected: overlapping celebrity IDs!"
    print(f"Train groups: {len(train_celebs)}, Test groups: {len(test_celebs)}")

    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)

    # Recompute pos_weight based on the training subset.
    train_labels = [dataset.labels[i][0] for i in train_idx]  # label is the first element.
    num_pos = sum(train_labels)
    num_neg = len(train_labels) - num_pos
    ratio = num_neg / num_pos if num_pos > 0 else 1.0
    pos_weight = torch.tensor([ratio], dtype=torch.float32).to(device)
    print(f"Computed pos_weight (from train subset): {pos_weight.item():.4f}")

    criterion = CombinedLoss(
        bce_weight=0.5,
        jsd_weight=0.5,
        pos_weight=pos_weight
    )

    batch_size = 32
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,  # pin_memory=True is safe on CPU-only systems.
        drop_last=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )

    # Create dynamic chain graph edge index.
    base_edge_index = create_chain_graph(seq_len).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',  # Monitor validation loss.
        factor=0.5,
        patience=3,
        verbose=True
    )

    num_epochs = 20
    best_auc = 0.0

    print("Starting training...")
    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_acc, train_auc, train_f1, train_recall, train_frr, train_gar, train_precision = train_epoch(
            model, train_dataloader, criterion, optimizer, device, base_edge_index, seq_len, grad_clip=5.0
        )
        val_loss, val_acc, val_auc, val_f1, val_recall, val_frr, val_gar, val_precision = evaluate_model(
            model, test_dataloader, criterion, device, base_edge_index, seq_len
        )
        scheduler.step(val_loss)
        epoch_time = time.time() - start_time
        
        results = {
            'Epoch': epoch + 1,
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
            'Epoch Time': epoch_time
        }

        save_model_and_result(
            model, 
            results, 
            model_path=f"outputs/models/epoch-{epoch+1}-model.pt", 
            results_path=f"outputs/results/epoch-{epoch+1}-model.json"
        )

        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train AUC: {train_auc:.4f} | Train F1: {train_f1:.4f} | Train Recall: {train_recall:.4f} | Train FRR: {train_frr:.4f} | Train GAR: {train_gar:.4f} | Train Precision: {train_precision:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f} | Val Recall: {val_recall:.4f} | Val FRR: {val_frr:.4f} | Val GAR: {val_gar:.4f} | Val Precision: {val_precision:.4f}")
        print(f"Epoch Time: {epoch_time / 60:.2f} minutes")

        if val_auc > best_auc:
            best_auc = val_auc
            os.makedirs("outputs", exist_ok=True)
            torch.save(model.state_dict(), f"outputs/best_deepfake_model_epoch_{epoch+1}.pt")
            print("Best model updated and saved.")

    print("Training completed. Saving final model...")
    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), "outputs/deepfake_model_final.pt")
    print("Model saved at outputs/deepfake_model_final.pt")

if __name__ == "__main__":
    main()
