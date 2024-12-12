import torch
import torch.nn as nn
import torch.nn.functional as F

def jensen_shannon_divergence(pred_probs: torch.Tensor, true_labels: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Computes the Jensen-Shannon Divergence between predicted probabilities and true labels.
    
    Args:
        pred_probs (torch.Tensor): Predicted probabilities of shape (N,).
        true_labels (torch.Tensor): Ground truth labels of shape (N,).
        epsilon (float): Small value to ensure numerical stability.
        
    Returns:
        torch.Tensor: The mean Jensen-Shannon Divergence over the batch.
    """
    # Ensure pred_probs are between epsilon and 1 - epsilon
    pred_probs = pred_probs.clamp(min=epsilon, max=1.0 - epsilon)
    true_labels = true_labels.clamp(min=epsilon, max=1.0 - epsilon)
    
    # Convert scalar labels to distributions: P and Q
    P = torch.stack([1 - true_labels, true_labels], dim=1)  # Shape: (N, 2)
    Q = torch.stack([1 - pred_probs, pred_probs], dim=1)    # Shape: (N, 2)
    
    # Compute the average distribution M
    M = 0.5 * (P + Q)
    
    # Compute KL divergences
    KL_PM = F.kl_div(torch.log(M), P, reduction='batchmean')  # KL(P || M)
    KL_QM = F.kl_div(torch.log(M), Q, reduction='batchmean')  # KL(Q || M)
    
    # Compute JSD
    JSD = 0.5 * (KL_PM + KL_QM)
    
    return JSD

class CombinedLoss(nn.Module):
    """
    Combines Binary Cross-Entropy Loss with Jensen-Shannon Divergence.
    
    Args:
        bce_weight (float): Weight for the BCE loss component.
        jsd_weight (float): Weight for the JSD loss component.
        pos_weight (torch.Tensor, optional): Weight for the positive class in BCE loss.
    """
    def __init__(self, bce_weight: float = 0.5, jsd_weight: float = 0.5, pos_weight: torch.Tensor = None):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.jsd_weight = jsd_weight
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the combined loss.
        
        Args:
            logits (torch.Tensor): Raw model outputs before activation, shape (N,).
            labels (torch.Tensor): Ground truth labels, shape (N,).
            
        Returns:
            torch.Tensor: Combined loss scalar.
        """
        # Compute BCE loss
        bce = self.bce_loss(logits, labels)
        
        # Compute predicted probabilities
        probs = torch.sigmoid(logits)
        
        # Compute JSD
        jsd = jensen_shannon_divergence(probs, labels)
        
        # Combine losses
        loss = self.bce_weight * bce + self.jsd_weight * jsd
        return loss