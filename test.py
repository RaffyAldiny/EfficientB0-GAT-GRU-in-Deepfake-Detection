import os
import random
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import accuracy_score
from models.efficientnet import get_efficientnet  # Original EfficientNet for Eff-GAT-GRU model
from models.gat import GAT  # GAT Model
from models.gru import GRU  # GRU Model
from torchvision import models

# Try importing tqdm for progress bars; if not installed, prompt the user
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("⚠️ tqdm library not found. Progress bars will be disabled.")
    print("You can install it using 'pip install tqdm' for better progress visualization.\n")

# Import your DeepfakeDataset
from utils.dataset import DeepfakeDataset  # Ensure this is correctly implemented

#############################################
# EfficientB0-GAT-GRU Model (Original)
#############################################
class DeepfakeModel_Effgatgru(torch.nn.Module):
    """Deepfake Detection Model combining EfficientNet, GAT, and GRU."""
    def __init__(self, seq_len=40, dropout_rate=0.5):
        super(DeepfakeModel_Effgatgru, self).__init__()
        self.seq_len = seq_len
        self.efficientnet = get_efficientnet()
        # Projection layer: from 1280 (EfficientNet output) to 256
        self.projection = torch.nn.Linear(1280, 256)
        self.gat = GAT(in_channels=256, out_channels=8, heads=1)
        self.gru = GRU(input_size=8, hidden_size=32, num_layers=1, dropout=dropout_rate)
        self.fc = torch.nn.Linear(32, 1)

    def forward(self, x, batched_edge_index):
        batch_size, seq_len, c, h, w = x.shape  # Expecting 5D input
        x = x.view(batch_size * seq_len, c, h, w)
        spatial_features = self.efficientnet(x).squeeze(-1).squeeze(-1)

        # Apply projection
        projected_features = self.projection(spatial_features)

        gat_output = self.gat(projected_features, batched_edge_index)
        gat_output = gat_output.view(batch_size, seq_len, -1)

        gru_output = self.gru(gat_output)
        output = self.fc(gru_output[:, -1, :])
        return output


#############################################
# EfficientNet-B0 Only Model (New)
#############################################
class DeepfakeModel_Efficientnet(torch.nn.Module):
    """Deepfake Detection Model using only EfficientNet-B0."""
    def __init__(self, dropout_rate=0.5):
        super(DeepfakeModel_Efficientnet, self).__init__()
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        num_features = self.efficientnet.classifier[1].in_features
        # Replace the classifier with a new one for binary classification
        self.efficientnet.classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(num_features, 1)  # Output single logit for binary classification
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
        logits = self.efficientnet(x)
        # Aggregate logits over the sequence, e.g., by averaging
        logits = logits.view(batch_size, seq_len, -1)
        logits = logits.mean(dim=1)  # Simple aggregation
        return logits.squeeze(1)  # Shape: (batch_size)


# Define transformations for input preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model_effgatgru(model_path, seq_len=40, dropout_rate=0.5, device="cpu"):
    """Load the EfficientB0-GAT-GRU DeepfakeModel and its weights."""
    model = DeepfakeModel_Effgatgru(seq_len=seq_len, dropout_rate=dropout_rate)
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()  # Set model to evaluation mode
        print(f"✅ EfficientB0-GAT-GRU model loaded successfully from '{model_path}'.")
    except Exception as e:
        print(f"❌ Error loading the EfficientB0-GAT-GRU model: {e}")
        raise e
    return model

def load_model_efficientnet(model_path, dropout_rate=0.5, device="cpu"):
    """Load the EfficientNet-B0 DeepfakeModel and its weights."""
    model = DeepfakeModel_Efficientnet(dropout_rate=dropout_rate)
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()  # Set model to evaluation mode
        print(f"✅ EfficientNet-B0 model loaded successfully from '{model_path}'.")
    except Exception as e:
        print(f"❌ Error loading the EfficientNet-B0 model: {e}")
        raise e
    return model

def create_edge_index(seq_len):
    """Create edge index for GAT."""
    edge_list = []
    for i in range(seq_len - 1):
        edge_list.append([i, i + 1])
        edge_list.append([i + 1, i])
    return torch.tensor(edge_list, dtype=torch.long).t().contiguous()

def load_frames(video_path, seq_len=40):
    """Load video frames from a folder."""
    frames = []
    frame_files = sorted(os.listdir(video_path))
    total_frames = len(frame_files)
    print(f"📁 Loading frames from '{video_path}' ({total_frames} frames found).")
    
    # Determine the number of frames to load
    frames_to_load = min(seq_len, total_frames)
    frames_loaded = 0

    for frame_file in frame_files[:frames_to_load]:
        frame_path = os.path.join(video_path, frame_file)
        try:
            frame = Image.open(frame_path).convert('RGB')  # Ensure 3 channels
            frames.append(transform(frame))
            frames_loaded += 1
        except Exception as e:
            print(f"⚠️ Error loading frame '{frame_path}': {e}")
            frames.append(torch.zeros(3, 224, 224))  # Placeholder for failed frames
    
    # Handle padding if less than seq_len
    if frames_loaded < seq_len:
        padding_needed = seq_len - frames_loaded
        print(f"🔧 Padding with {padding_needed} zero frames to reach {seq_len} frames.")
        for _ in range(padding_needed):
            frames.append(torch.zeros(3, 224, 224))  # Assuming 3 channels and 224x224
    
    frames = torch.stack(frames[:seq_len])  # Ensure exactly seq_len frames
    return frames

def evaluate_model(model, dataset_path, edge_index, max_samples=10, seq_len=40, device="cpu", model_name="Model"):
    """Evaluate the model on the dataset."""
    y_true = []
    y_pred = []
    # If video belongs to "Celeb-real" or "Youtube-real" -> label=1 (Real)
    # If video belongs to "Celeb-synthesis" -> label=0 (Fake)
    labels = {"Celeb-real": 1, "Celeb-synthesis": 0, "Youtube-real": 1}
    total_videos = sum([
        min(len(os.listdir(os.path.join(dataset_path, label))), max_samples) 
        for label in labels 
        if os.path.isdir(os.path.join(dataset_path, label))
    ])
    
    print(f"\n🔍 Starting evaluation for {model_name} on dataset '{dataset_path}' with {total_videos} total videos.\n")

    for label, is_real in labels.items():
        label_path = os.path.join(dataset_path, label)
        print(f"📂 Processing label '{label}' (Real: {is_real}).")
        
        if not os.path.isdir(label_path):
            print(f"⚠️ Warning: Directory '{label_path}' does not exist. Skipping.")
            continue
        
        video_folders = os.listdir(label_path)
        if not video_folders:
            print(f"⚠️ Warning: No videos found in '{label_path}'. Skipping.")
            continue
        
        sampled_folders = random.sample(video_folders, min(len(video_folders), max_samples))
        print(f"📋 {len(sampled_folders)} videos selected for evaluation under label '{label}'.\n")
        
        # Initialize progress bar if tqdm is available
        if TQDM_AVAILABLE:
            video_iterator = tqdm(sampled_folders, desc=f"Evaluating '{label}'", unit="video")
        else:
            video_iterator = sampled_folders
        
        for video_folder in video_iterator:
            video_path_full = os.path.join(label_path, video_folder)
            print(f"➡️ Processing video '{video_folder}'...")
            frames = load_frames(video_path_full, seq_len=seq_len)
            frames = frames.unsqueeze(0)  # [1, seq_len, C, H, W]
            frames = frames.to(device)

            # Create batched edge index for batch_size=1
            batched_edge_index = edge_index.clone().to(device)

            # Model prediction
            with torch.no_grad():
                if model_name == "EfficientNet-B0":
                    # EfficientNet model does not use GAT-GRU or edge index
                    outputs = model(frames)  # Shape: (batch_size)
                else:
                    # Eff-GAT-GRU model uses batched_edge_index
                    outputs = model(frames, batched_edge_index)  # Shape: (batch_size)
                
                probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
                print("probs ", probs)
                pred = int(probs > 0.5)
            
            y_true.append(is_real)
            y_pred.append(pred)
            print(f"✅ Video '{video_folder}' processed. Prediction: {'Real' if pred == 1 else 'Fake'}\n")
    
    if not y_true or not y_pred:
        print(f"❌ Error: No predictions were made for {model_name}. Please check your dataset and model.")
        return 0.0, 0, 0.0
    
    # Compute metrics
    total_samples = len(y_true)
    correct_predictions = sum([1 for true, pred in zip(y_true, y_pred) if true == pred])
    accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0.0
    
    print(f"📊 {model_name} Evaluation Metrics:")
    print(f"📝 Total Samples: {total_samples}")
    print(f"✅ Correct Predictions: {correct_predictions}")
    print(f"📈 Accuracy: {accuracy:.2f}%\n")
    
    return accuracy, correct_predictions, total_samples

def main():
    """Main function to evaluate both models and compare their performance."""
    # Paths and parameters
    model_path_effgatgru = "outputs/models/epoch-20-efficientgatgru-model.pt"  # Path to EfficientB0-GAT-GRU model
    model_path_efficientnet = "outputs/models/epoch-20-efficientnet-model.pt"  # Path to EfficientNet-B0 model
    dataset_path = "data/preprocessed"
    seq_len = 40
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("🛠️ Starting Deepfake Detection Evaluation Script\n")
    
    # Check if model files exist
    if not os.path.exists(model_path_effgatgru):
        print(f"❌ Error: EfficientB0-GAT-GRU model file not found at '{model_path_effgatgru}'.")
        exit(1)
    else:
        print(f"📂 EfficientB0-GAT-GRU model file found at '{model_path_effgatgru}'.")
    
    if not os.path.exists(model_path_efficientnet):
        print(f"❌ Error: EfficientNet-B0 model file not found at '{model_path_efficientnet}'.")
        exit(1)
    else:
        print(f"📂 EfficientNet-B0 model file found at '{model_path_efficientnet}'.")
    
    # Check if dataset directory exists
    if not os.path.isdir(dataset_path):
        print(f"❌ Error: Dataset directory not found at '{dataset_path}'.")
        exit(1)
    else:
        print(f"📂 Dataset directory found at '{dataset_path}'.\n")
    
    # Load the EfficientB0-GAT-GRU model
    print("🔄 Loading the EfficientB0-GAT-GRU model...")
    model_effgatgru = load_model_effgatgru(model_path_effgatgru, seq_len=seq_len, device=device)
    model_effgatgru.to(device)
    print("🔄 EfficientB0-GAT-GRU model loading complete.\n")
    
    # Load the EfficientNet-B0 model
    print("🔄 Loading the EfficientNet-B0 model...")
    model_efficientnet = load_model_efficientnet(model_path_efficientnet, device=device)
    model_efficientnet.to(device)
    print("🔄 EfficientNet-B0 model loading complete.\n")
    
    # Create edge index for GAT (used only by EfficientB0-GAT-GRU)
    print("🔧 Creating edge index for GAT...")
    edge_index = create_edge_index(seq_len=seq_len)
    edge_index = edge_index.to(device)
    print("🔧 Edge index creation complete.\n")
    
    # Evaluate EfficientB0-GAT-GRU Model
    print("🔍 Evaluating EfficientB0-GAT-GRU Model...\n")
    accuracy_effgatgru, correct_effgatgru, total_effgatgru = evaluate_model(
        model=model_effgatgru,
        dataset_path=dataset_path,
        edge_index=edge_index,
        max_samples=8,  # Adjust as needed
        seq_len=seq_len,
        device=device,
        model_name="EfficientB0-GAT-GRU"
    )
    
    # Evaluate EfficientNet-B0 Model
    print("🔍 Evaluating EfficientNet-B0 Model...\n")
    accuracy_efficientnet, correct_efficientnet, total_efficientnet = evaluate_model(
        model=model_efficientnet,
        dataset_path=dataset_path,
        edge_index=edge_index,  # Not used by EfficientNet, but we pass for compatibility
        max_samples=8,  # Adjust as needed
        seq_len=seq_len,
        device=device,
        model_name="EfficientNet-B0"
    )
    
    # Compare the results
    print("📊 Comparison of Model Performance:")
    print(f"🔸 EfficientB0-GAT-GRU:")
    print(f"   📝 Total Samples: {total_effgatgru}")
    print(f"   ✅ Correct Predictions: {correct_effgatgru}")
    print(f"   📈 Accuracy: {accuracy_effgatgru:.2f}%\n")
    
    print(f"🔸 EfficientNet-B0:")
    print(f"   📝 Total Samples: {total_efficientnet}")
    print(f"   ✅ Correct Predictions: {correct_efficientnet}")
    print(f"   📈 Accuracy: {accuracy_efficientnet:.2f}%\n")
    
    print("🎉 Evaluation Completed Successfully.\n")

if __name__ == "__main__":
    main()
