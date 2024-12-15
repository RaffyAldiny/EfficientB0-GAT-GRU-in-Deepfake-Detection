import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
from models.efficientnet import get_efficientnet
from models.gat import GAT
from models.gru import GRU
from utils.dataset import DeepfakeDataset
from utils.losses import CombinedLoss
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
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
    Dynamically create for each batch to match the current batch size.
    """
    edge_index = edge_index.clone()
    edge_index = edge_index.repeat(1, batch_size)
    offsets = torch.arange(batch_size, device=device) * num_nodes
    offsets = offsets.unsqueeze(0).repeat(2, edge_index.size(1) // batch_size)
    edge_index += offsets
    return edge_index

class DeepfakeModel(nn.Module):
    """Deepfake Detection Model combining EfficientNet, GAT, and GRU with dropout."""
    def __init__(self, seq_len=40, dropout_rate=0.5):
        super(DeepfakeModel, self).__init__()
        self.seq_len = seq_len
        self.efficientnet = get_efficientnet()
        # Projection layer: from 1280 (EfficientNet output) to 256
        self.projection = nn.Linear(1280, 256)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.gat = GAT(in_channels=256, out_channels=8, heads=1)
        self.gru = GRU(input_size=8, hidden_size=32, num_layers=1, dropout=dropout_rate)
        self.fc = nn.Linear(32, 1)

    def forward(self, x, batched_edge_index):
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w)
        spatial_features = self.efficientnet(x).squeeze(-1).squeeze(-1)

        # Apply projection and dropout
        projected_features = self.projection(spatial_features)
        projected_features = self.dropout(projected_features)

        gat_output = self.gat(projected_features, batched_edge_index)
        gat_output = gat_output.view(batch_size, seq_len, -1)

        gru_output = self.gru(gat_output)
        output = self.fc(gru_output[:, -1, :])
        return output

def compute_metrics(labels, preds, probs):
    """
    Compute various metrics:
    Accuracy, F1 (for fake class), AUC, Recall (for fake),
    FRR (for real), GAR, and Precision (for fake).
    """
    if len(labels) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

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

def train_epoch(model, dataloader, criterion, optimizer, device, edge_index, seq_len, grad_clip=5.0):
    """Train the model for one epoch."""
    if len(dataloader) == 0:
        # No training data
        return 0, 0, 0, 0, 0, 0, 0, 0

    model.train()
    epoch_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    with tqdm(total=len(dataloader), desc="Training", unit="batch") as pbar:
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            current_batch_size = inputs.size(0)
            # Dynamically create batched_edge_index
            batched_edge_index = create_batched_edge_index(edge_index, current_batch_size, seq_len, device)

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
            batch_labels_np = labels.cpu().numpy()

            # Compute batch metrics
            batch_acc, batch_f1, batch_auc, batch_recall, batch_frr, batch_gar, batch_precision = compute_metrics(batch_labels_np, preds, probs)

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

    average_loss = epoch_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    epoch_acc, epoch_f1, epoch_auc, epoch_recall, epoch_frr, epoch_gar, epoch_precision = compute_metrics(all_labels, all_preds, all_probs)
    return average_loss, epoch_acc, epoch_auc, epoch_f1, epoch_recall, epoch_frr, epoch_gar, epoch_precision

def evaluate_model(model, dataloader, criterion, device, edge_index, seq_len):
    """Evaluate the model."""
    if len(dataloader) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    model.eval()
    epoch_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        with tqdm(total=len(dataloader), desc="Evaluating", unit="batch") as pbar:
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)

                current_batch_size = inputs.size(0)
                # Dynamically create batched_edge_index for current batch
                batched_edge_index = create_batched_edge_index(edge_index, current_batch_size, seq_len, device)

                outputs = model(inputs, batched_edge_index)
                outputs = outputs.view(-1)
                labels = labels.float().view(-1)

                loss = criterion(outputs, labels)
                epoch_loss += loss.item()

                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                batch_labels_np = labels.cpu().numpy()

                # Compute batch metrics
                batch_acc, batch_f1, batch_auc, batch_recall, batch_frr, batch_gar, batch_precision = compute_metrics(batch_labels_np, preds, probs)

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

    average_loss = epoch_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    val_acc, val_f1, val_auc, val_recall, val_frr, val_gar, val_precision = compute_metrics(all_labels, all_preds, all_probs)
    return average_loss, val_acc, val_auc, val_f1, val_recall, val_frr, val_gar, val_precision

def main():
    """Main function to train and evaluate the deepfake detection model."""
    # Check if preprocessed data exists
    preprocessed_dir = "data/preprocessed/"
    required_subdirs = ["Celeb-real", "Celeb-synthesis"]
    missing_subdirs = [subdir for subdir in required_subdirs if not os.path.exists(os.path.join(preprocessed_dir, subdir))]

    if missing_subdirs:
        print(f"Error: Preprocessed directories missing: {missing_subdirs}. Please run the preprocessing script first.")
        print("Proceeding with directories only found in the preprocessed folder.")
    else:
        print("Preprocessed data found. Proceeding to training.")

    seq_len = 40
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

    video_paths = []
    labels = []
    with open(labels_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                lbl, vid = line.split()
                lbl = int(lbl)
                video_paths.append(vid)
                labels.append(lbl)

    if len(video_paths) == 0:
        print("No videos found in the labels file.")
        return

    # Split into train+val and test
    train_val_videos, test_videos, train_val_labels, test_labels = train_test_split(
        video_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    if len(train_val_videos) == 0 or len(test_videos) == 0:
        print("Train/test split resulted in no data. Check dataset and splitting parameters.")
        return

    # Split train+val into train and val
    train_videos, val_videos, train_labels, val_labels = train_test_split(
        train_val_videos, train_val_labels, test_size=0.25, random_state=42, stratify=train_val_labels
    )

    print(f"Training videos: {len(train_videos)} | Validation videos: {len(val_videos)} | Test videos: {len(test_videos)}")

    labels_dict = dict(zip(video_paths, labels))

    train_dataset = DeepfakeDataset(
        root_dir=preprocessed_dir,
        video_list=train_videos,
        labels_dict=labels_dict,
        transform=transform,
        seq_len=seq_len
    )

    val_dataset = DeepfakeDataset(
        root_dir=preprocessed_dir,
        video_list=val_videos,
        labels_dict=labels_dict,
        transform=transform,
        seq_len=seq_len
    )

    test_dataset = DeepfakeDataset(
        root_dir=preprocessed_dir,
        video_list=test_videos,
        labels_dict=labels_dict,
        transform=transform,
        seq_len=seq_len
    )

    if len(train_dataset) == 0:
        print("Error: No valid samples found in the training dataset.")
        return

    # Compute class weights from the training set
    train_labels_all = [label for _, label in train_dataset]
    unique, counts = torch.unique(torch.tensor(train_labels_all), return_counts=True)
    class_counts = dict(zip(unique.tolist(), counts.tolist()))
    print(f"Class distribution in train: {class_counts}")

    pos_weight = None
    if len(class_counts) == 2:
        classes = np.array(list(class_counts.keys()))
        labels_np = np.array(train_labels_all)
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

    batch_size = 32
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
        drop_last=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        drop_last=False
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        drop_last=False
    )

    # Create chain graph for seq_len nodes
    edge_list = []
    for i in range(seq_len - 1):
        edge_list.append([i, i + 1])
        edge_list.append([i + 1, i])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3,
        verbose=True
    )

    num_epochs = 10
    best_auc = 0.0

    print("Starting training...")
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss, train_acc, train_auc, train_f1, train_recall, train_frr, train_gar, train_precision = train_epoch(
            model, train_dataloader, criterion, optimizer, device, edge_index, seq_len, grad_clip=5.0
        )
        val_loss, val_acc, val_auc, val_f1, val_recall, val_frr, val_gar, val_precision = evaluate_model(
            model, val_dataloader, criterion, device, edge_index, seq_len
        )

        scheduler.step(val_auc)

        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train AUC: {train_auc:.4f} | Train F1: {train_f1:.4f} | Train Recall: {train_recall:.4f} | Train FRR: {train_frr:.4f} | Train GAR: {train_gar:.4f} | Train Precision: {train_precision:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f} | Val Recall: {val_recall:.4f} | Val FRR: {val_frr:.4f} | Val GAR: {val_gar:.4f} | Val Precision: {val_precision:.4f}")
        print(f"Epoch Time: {epoch_time/60:.2f} minutes")

        if val_auc > best_auc:
            best_auc = val_auc
            os.makedirs("outputs", exist_ok=True)
            torch.save(model.state_dict(), "outputs/best_deepfake_model.pth")
            print("Best model updated and saved.")

    # Evaluate on the test set
    test_loss, test_acc, test_auc, test_f1, test_recall, test_frr, test_gar, test_precision = evaluate_model(
        model, test_dataloader, criterion, device, edge_index, seq_len
    )
    print("\nTest Set Performance:")
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test AUC: {test_auc:.4f} | Test F1: {test_f1:.4f} | Test Recall: {test_recall:.4f} | Test FRR: {test_frr:.4f} | Test GAR: {test_gar:.4f} | Test Precision: {test_precision:.4f}")

    print("Training completed. Saving final model...")
    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), "outputs/deepfake_model_final.pth")
    print("Model saved at outputs/deepfake_model_final.pth")

if __name__ == "__main__":
    main()
