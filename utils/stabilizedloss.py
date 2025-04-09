import torch
import torch.nn as nn
import torch.nn.functional as F

def jensen_shannon_divergence(pred_probs: torch.Tensor, true_labels: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Computes a numerically stable Jensen-Shannon Divergence (JSD) between the predicted probabilities and true labels.
    
    Fixes applied:
      - Clamps pred_probs to [epsilon, 1 - epsilon] to avoid log(0).
      - Clamps the averaged distribution M before normalization.
    
    Args:
        pred_probs (torch.Tensor): Predicted probabilities (after sigmoid), shape (N,).
        true_labels (torch.Tensor): True binary labels, shape (N,).
        epsilon (float): Small constant for numerical stability.
        
    Returns:
        torch.Tensor: The mean JSD over the batch.
    """
    # Clamp predicted probabilities.
    pred_probs = pred_probs.clamp(min=epsilon, max=1.0 - epsilon)
    
    # Convert true labels and predictions into two-element distributions.
    P = torch.stack([1 - true_labels, true_labels], dim=1)
    Q = torch.stack([1 - pred_probs, pred_probs], dim=1)
    
    # Compute the average distribution.
    M = 0.5 * (P + Q)
    
    # Clamp M before normalization to avoid extreme values.
    M = M.clamp(min=epsilon)
    M = M / M.sum(dim=1, keepdim=True)
    log_M = torch.log(M)
    
    # Compute symmetric KL divergences.
    KL_PM = F.kl_div(log_M, P, reduction='batchmean')
    KL_QM = F.kl_div(log_M, Q, reduction='batchmean')
    
    return 0.5 * (KL_PM + KL_QM)

class CombinedLoss(nn.Module):
    """
    Combines a smoothed Binary Cross-Entropy (BCE) loss with a detached Jensen-Shannon Divergence (JSD) loss,
    plus an entropy regularization term.

    Key Improvements:
      - Label Smoothing: Uses smoothed labels (e.g. 0.05 and 0.95) instead of hard 0/1 targets.
      - Temperature Scaling: Softens the probability distribution before computing JSD.
      - Gradient Isolation: Computes the JSD term inside a torch.no_grad() block to prevent its gradients 
        from propagating to earlier layers.
      - Entropy Regularization: Penalizes overconfident (i.e. low entropy) predictions.

    Recommended Usage:
        criterion = CombinedLoss(
            pos_weight=pos_weight,
            temp=0.7,        # Start with 0.7, adjust within 0.5 - 1.0 based on experiments.
            jsd_weight=0.2   # Start low and potentially increase if necessary.
        )
    """
    def __init__(self, pos_weight=None, temp=0.7, jsd_weight=0.2):
        super().__init__()
        self.temp = temp
        self.jsd_weight = jsd_weight
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Apply label smoothing: convert hard labels to soft targets.
        smooth_labels = labels * 0.9 + 0.05  # For example, 1 becomes 0.95 and 0 becomes 0.05.
        bce = self.bce_loss(logits, smooth_labels)
        
        # JSD Calculation with gradient isolation.
        with torch.no_grad():
            # Apply temperature scaling before the sigmoid.
            scaled_logits = logits / self.temp
            probs = torch.sigmoid(scaled_logits)
            jsd = jensen_shannon_divergence(probs, labels)
        
        # Entropy regularization: penalize overconfident predictions.
        main_probs = torch.sigmoid(logits)
        epsilon = 1e-8
        main_probs = main_probs.clamp(min=epsilon, max=1.0 - epsilon)
        entropy = -main_probs * torch.log(main_probs) - (1 - main_probs) * torch.log(1 - main_probs)
        
        # Combine the losses.
        return bce + self.jsd_weight * jsd + 0.1 * entropy.mean()
