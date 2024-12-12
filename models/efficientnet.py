# models/efficientnet.py
from torchvision.models import efficientnet_b0
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights

def get_efficientnet():
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    feature_extractor = nn.Sequential(*list(model.children())[:-1])  # Remove last classification layer
    for param in feature_extractor.parameters():
        param.requires_grad = False  # Freeze EfficientNet parameters
    return feature_extractor
