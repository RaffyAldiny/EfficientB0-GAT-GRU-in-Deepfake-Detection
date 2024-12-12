import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, ToTensor, Resize
from models.efficientnet import get_efficientnet
from models.gat import GAT
from models.gru import GRU
from utils.preprocess import preprocess_dataset
from utils.dataset import DeepfakeDataset
from utils.losses import CombinedLoss
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.utils import class_weight
import os
import time
from tqdm import tqdm
import random
import numpy as np

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

def create_batched_edge_index(edge_index, batch_size, num_nodes, device):
    """
    Create a batched edge index for the graph attention network.

    Args:
        edge_index (torch.Tensor): Original edge index.
        batch_size (int): Number of samples in a batch.
        num_nodes (int): Number of nodes per graph.
        device (torch.device): Device to place the tensor on.

    Returns:
        torch.Tensor: Batched edge index.
    """
    edge_index = edge_index.clone()
    edge_index = edge_index.repeat(1, batch_size)
    offsets = torch.arange(batch_size, device=device) * num_nodes
    offsets = offsets.unsqueeze(0).repeat(2, edge_index.size(1) // batch_size)
    edge_index += offsets
    return edge_index

class DeepfakeModel(nn.Module):
    """Deepfake Detection Model combining EfficientNet, GAT, and GRU."""

    def __init__(self, seq_len=100, dropout_rate=0.5):
        super(DeepfakeModel, self).__init__()
        self.seq_len = seq_len
        self.efficientnet = get_efficientnet()
        self.gat = GAT(in_channels=1280, out_channels=8, heads=1)  
        self.gru = GRU(input_size=8, hidden_size=32, num_layers=1, dropout=dropout_rate)
        self.fc = nn.Linear(32, 1)

    def forward(self, x, batched_edge_index):
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w)
        spatial_features = self.efficientnet(x).squeeze(-1).squeeze(-1)
        node_features = spatial_features

        gat_output = self.gat(node_features, batched_edge_index)  
        gat_output = gat_output.view(batch_size, seq_len, -1)

        gru_output = self.gru(gat_output)
        output = self.fc(gru_output[:, -1, :])
        return output

def train_epoch(model, dataloader, criterion, optimizer, device, batched_edge_index, grad_clip=5.0):
    """Train the model for one epoch."""
    model.train()
    epoch_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    with tqdm(total=len(dataloader), desc="Training", unit="batch") as pbar:
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, batched_edge_index)
            outputs = outputs.view(-1)
            labels = labels.float().view(-1)

            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

            batch_accuracy = accuracy_score(labels.cpu().numpy(), preds)
            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Acc': f"{batch_accuracy:.4f}"})
            pbar.update(1)

    average_loss = epoch_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    f1 = f1_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.0
    return average_loss, accuracy, auc, f1

def evaluate_model(model, dataloader, criterion, device, batched_edge_index):
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

                outputs = model(inputs, batched_edge_index)
                outputs = outputs.view(-1)
                labels = labels.float().view(-1)

                loss = criterion(outputs, labels)
                epoch_loss += loss.item()

                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(int)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds)
                all_probs.extend(probs)

                batch_accuracy = accuracy_score(labels.cpu().numpy(), preds)
                pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Acc': f"{batch_accuracy:.4f}"})
                pbar.update(1)

    average_loss = epoch_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    f1 = f1_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.0
    return average_loss, accuracy, auc, f1

def main():
    """Main function to train and evaluate the deepfake detection model."""
    # If data isn't preprocessed, run preprocessing
    preprocessed_dir = "data/preprocessed/"
    if not os.path.exists(preprocessed_dir) or len(os.listdir(preprocessed_dir)) == 0:
        print("Starting preprocessing...")
        os.makedirs("data/preprocessed/real", exist_ok=True)
        os.makedirs("data/preprocessed/synthesis", exist_ok=True)
        os.makedirs("data/preprocessed/YouTube-real", exist_ok=True)

        print("Preprocessing Celeb-real...")
        preprocess_dataset("data/Celeb-real", "data/preprocessed/real")
        print("Preprocessing Celeb-synthesis...")
        preprocess_dataset("data/Celeb-synthesis", "data/preprocessed/synthesis")
        print("Preprocessing YouTube-real...")
        preprocess_dataset("data/YouTube-real", "data/preprocessed/YouTube-real")
    else:
        print("Preprocessed data found. Skipping preprocessing.")

    seq_len = 100
    dropout_rate = 0.5
    model = DeepfakeModel(seq_len=seq_len, dropout_rate=dropout_rate).to(device)

    transform = Compose([
        Resize((224, 224)),
        ToTensor()
    ])

    labels_file = "data/List_of_testing_videos.txt"
    if not os.path.exists(labels_file):
        print(f"Error: Labels file not found at {labels_file}")
        return

    # The DeepfakeDataset class handles padding shorter sequences
    dataset = DeepfakeDataset(
        root_dir="data/preprocessed/",
        labels_file=labels_file,
        transform=transform,
        limit=2000,  # Adjust limit as needed
        seq_len=seq_len
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
    if len(class_counts) == 2:
        classes = np.array(list(class_counts.keys()))  # Convert to numpy array
        labels_np = np.array(labels)  # Convert labels to numpy array
        class_weights = class_weight.compute_class_weight('balanced', classes=classes, y=labels_np)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        pos_weight = class_weights[1] / (class_weights[0] if class_weights[0] != 0 else 1.0)
        print(f"Class Weights: {class_weights.tolist()} | Pos Weight: {pos_weight:.4f}")
    else:
        print("Not a binary classification scenario or class count issue. Using default weights.")

    criterion = CombinedLoss(
        bce_weight=0.5, 
        jsd_weight=0.5, 
        pos_weight=(torch.tensor(pos_weight).to(device) if pos_weight is not None else None)
    )

    test_split = 0.3
    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    batch_size = 9
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=False, 
        drop_last=True
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=False, 
        drop_last=True
    )

    # Create chain graph for seq_len nodes
    edge_list = []
    for i in range(seq_len - 1):
        edge_list.append([i, i + 1])
        edge_list.append([i + 1, i])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(device)

    # Precompute batched_edge_index once
    batched_edge_index = create_batched_edge_index(edge_index, batch_size, seq_len, device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=3, 
        verbose=True
    )

    num_epochs = 3
    best_auc = 0.0

    print("Starting training...")
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss, train_acc, train_auc, train_f1 = train_epoch(
            model, train_dataloader, criterion, optimizer, device, batched_edge_index, grad_clip=5.0
        )
        val_loss, val_acc, val_auc, val_f1 = evaluate_model(
            model, test_dataloader, criterion, device, batched_edge_index
        )

        scheduler.step(val_auc)

        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train AUC: {train_auc:.4f} | Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f}")
        print(f"Epoch Time: {epoch_time/60:.2f} minutes")

        if val_auc > best_auc:
            best_auc = val_auc
            os.makedirs("outputs", exist_ok=True)
            torch.save(model.state_dict(), "outputs/best_deepfake_model.pth")
            print("Best model updated and saved.")

    print("Training completed. Saving final model...")
    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), "outputs/deepfake_model_final.pth")
    print("Model saved at outputs/deepfake_model_final.pth")

if __name__ == "__main__":
    main()
