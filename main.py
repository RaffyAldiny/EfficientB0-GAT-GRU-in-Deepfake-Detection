# main.py
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
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.utils import class_weight
import os
import time
from tqdm import tqdm
import random
import numpy as np
import math

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Jensen-Shannon Divergence for binary classification
def jensen_shannon_divergence(pred_probs, true_labels, epsilon=1e-8):
    """
    Computes the Jensen-Shannon Divergence between predicted probabilities and true labels.
    pred_probs: Tensor of shape (N,) with predicted probability of class=1
    true_labels: Tensor of shape (N,) with ground truth labels in {0,1}

    We consider distributions:
    P = [1 - y, y]
    Q = [1 - p, p]
    M = (P + Q) / 2

    JS(P||Q) = 0.5 * (KL(P||M) + KL(Q||M))

    KL(A||B) = sum over i: A[i]*log(A[i]/B[i])
    """
    # Convert scalar labels to distributions: P and Q
    # P for true distribution: if y=1, P=[0,1], if y=0, P=[1,0]
    P = torch.stack([1 - true_labels, true_labels], dim=1)
    Q = torch.stack([1 - pred_probs, pred_probs], dim=1)

    M = 0.5 * (P + Q)

    # Compute KL divergences
    # Add epsilon to avoid log(0)
    P = P + epsilon
    Q = Q + epsilon
    M = M + epsilon

    KL_PM = torch.sum(P * torch.log(P / M), dim=1)
    KL_QM = torch.sum(Q * torch.log(Q / M), dim=1)

    JS = 0.5 * (KL_PM + KL_QM)
    return JS.mean()

def create_batched_edge_index(edge_index, batch_size, num_nodes, device):
    edge_index = edge_index.clone()
    edge_index = edge_index.repeat(1, batch_size)
    offsets = torch.arange(batch_size, device=device) * num_nodes
    offsets = offsets.unsqueeze(0).repeat(2, edge_index.size(1) // batch_size)
    edge_index += offsets
    return edge_index

class DeepfakeModel(nn.Module):
    def __init__(self, seq_len=100, dropout_rate=0.5):
        super(DeepfakeModel, self).__init__()
        self.seq_len = seq_len
        self.efficientnet = get_efficientnet()
        self.gat = GAT(in_channels=1280, out_channels=8, heads=1)  
        self.gru = GRU(input_size=8, hidden_size=32, num_layers=1, dropout=dropout_rate)# Output: 8
        self.fc = nn.Linear(32, 1)

    def forward(self, x, edge_index, batch_size, num_nodes):
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w)
        spatial_features = self.efficientnet(x).squeeze(-1).squeeze(-1)
        node_features = spatial_features

        batched_edge_index = create_batched_edge_index(edge_index, batch_size, num_nodes, x.device)
        gat_output = self.gat(node_features, batched_edge_index)  
        gat_output = gat_output.view(batch_size, seq_len, -1)

        gru_output = self.gru(gat_output)
        output = self.fc(gru_output[:, -1, :])
        return output

class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, jsd_weight=0.5, pos_weight=None):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.jsd_weight = jsd_weight
        if pos_weight is not None:
            self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, labels):
        # BCE Loss
        bce = self.bce_loss(logits, labels)

        # Convert logits to probabilities for JSD
        probs = torch.sigmoid(logits)
        jsd = jensen_shannon_divergence(probs, labels)

        # Weighted sum
        loss = self.bce_weight * bce + self.jsd_weight * jsd
        return loss

def train_epoch(model, dataloader, criterion, optimizer, device, edge_index, num_nodes, grad_clip=5.0):
    model.train()
    epoch_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    with tqdm(total=len(dataloader), desc="Training", unit="batch") as pbar:
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, edge_index, inputs.size(0), num_nodes)
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

def evaluate_model(model, dataloader, criterion, device, edge_index, num_nodes):
    model.eval()
    epoch_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        with tqdm(total=len(dataloader), desc="Evaluating", unit="batch") as pbar:
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs, edge_index, inputs.size(0), num_nodes)
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
    # Check if preprocessed data exists to avoid reprocessing
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

    dataset = DeepfakeDataset(
        root_dir="data/preprocessed/",
        labels_file=labels_file,
        transform=transform,
        limit=50,     # Adjust limit as needed
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

    # Compute class weights for binary classification
    pos_weight = None
    if len(class_counts) == 2:
        classes = list(class_counts.keys())
        counts_list = list(class_counts.values())
        class_weights = class_weight.compute_class_weight('balanced', classes=classes, y=labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        pos_weight = class_weights[1] / (class_weights[0] if class_weights[0] != 0 else 1.0)
        print(f"Class Weights: {class_weights.tolist()} | Pos Weight: {pos_weight:.4f}")
    else:
        print("Not a binary classification scenario or class count issue. Using default weights.")

    # Create combined loss with BCE and JSD
    # Weighted 0.5 each as per your requirement
    criterion = CombinedLoss(bce_weight=0.5, jsd_weight=0.5, pos_weight=(torch.tensor(pos_weight).to(device) if pos_weight is not None else None))

    # Train/test split
    test_split = 0.2
    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Adjust batch size
    batch_size = 2  # If resources allow, you can increase this
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)

    # Create chain graph for seq_len nodes
    num_nodes = seq_len
    edge_list = []
    for i in range(num_nodes - 1):
        edge_list.append([i, i + 1])
        edge_list.append([i + 1, i])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    num_epochs = 3
    best_auc = 0.0

    print("Starting training...")
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss, train_acc, train_auc, train_f1 = train_epoch(
            model, train_dataloader, criterion, optimizer, device, edge_index, num_nodes, grad_clip=5.0
        )
        val_loss, val_acc, val_auc, val_f1 = evaluate_model(
            model, test_dataloader, criterion, device, edge_index, num_nodes
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
