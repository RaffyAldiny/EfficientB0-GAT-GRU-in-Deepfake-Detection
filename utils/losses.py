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
    # Clamp M before normalization.
    M = M.clamp(min=epsilon)
    # Normalize M to form a valid probability distribution.
    M = M / M.sum(dim=1, keepdim=True)
    log_M = torch.log(M)
    KL_PM = F.kl_div(log_M, P, reduction='batchmean')
    KL_QM = F.kl_div(log_M, Q, reduction='batchmean')
    return 0.5 * (KL_PM + KL_QM)

class CombinedLoss(nn.Module):
    """
    Combines Binary Cross-Entropy (BCE) loss with Jensen-Shannon Divergence (JSD) loss.
    
    Allows partial gradient propagation for the JSD branch:
      - jsd_grad_factor = 0   -> No gradient flows (JSD branch is detached).
      - jsd_grad_factor = 1   -> Full gradient flows.
      - 0 < jsd_grad_factor < 1 -> Partial gradient scaling.
    """
    def __init__(self, bce_weight: float = 0.7, 
                 jsd_weight: float = 0.3, 
                 pos_weight: torch.Tensor = None, 
                 jsd_grad_factor: float = 0.3):
        super().__init__()
        self.bce_weight = bce_weight
        self.jsd_weight = jsd_weight
        self.jsd_grad_factor = jsd_grad_factor
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # BCE loss.
        bce = self.bce_loss(logits, labels)
        
        # JSD loss with partial gradient flow.
        full_probs = torch.sigmoid(logits)
        if self.jsd_grad_factor == 1.0:
            probs = full_probs
        elif self.jsd_grad_factor == 0.0:
            probs = full_probs.detach()
        else:
            # This formulation allows exactly jsd_grad_factor portion of the gradient to flow.
            probs = self.jsd_grad_factor * full_probs + (1 - self.jsd_grad_factor) * full_probs.detach()
        
        jsd = jensen_shannon_divergence(probs, labels)
        return self.bce_weight * bce + self.jsd_weight * jsd
