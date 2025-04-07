from torchvision.models import efficientnet_b0
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights

def get_efficientnet(freeze=True, adaptive_pool=True):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    # Remove the classifier (last layer)
    feature_layers = list(model.children())[:-1]
    if adaptive_pool:
        feature_layers.append(nn.AdaptiveAvgPool2d((1, 1)))
    feature_extractor = nn.Sequential(*feature_layers)
    if freeze:
        for param in feature_extractor.parameters():
            param.requires_grad = False
    return feature_extractor
