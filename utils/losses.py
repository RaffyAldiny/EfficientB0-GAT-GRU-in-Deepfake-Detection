# utils/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F

def jensen_shannon_divergence(pred_probs: torch.Tensor, true_labels: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Computes a numerically stable Jensen-Shannon Divergence (JSD) between the predicted probabilities and true labels.
    
    Fixes applied:
      - Clamps pred_probs to [epsilon, 1 - epsilon] to avoid log(0).
      - Clamps the averaged distribution M and then re-normalizes it to maintain a valid probability distribution.
    
    Args:
        pred_probs (torch.Tensor): Predicted probabilities (after sigmoid), shape (N,).
        true_labels (torch.Tensor): True binary labels, shape (N,).
        epsilon (float): Small constant for numerical stability.
        
    Returns:
        torch.Tensor: The mean JSD over the batch.
    """
    # Clamp predicted probabilities to avoid numerical issues.
    pred_probs = pred_probs.clamp(min=epsilon, max=1.0 - epsilon)
    
    # Convert true labels and predictions into two-element distributions:
    #   e.g., label=1 -> [0, 1], label=0 -> [1, 0].
    P = torch.stack([1 - true_labels, true_labels], dim=1)
    Q = torch.stack([1 - pred_probs, pred_probs], dim=1)
    
    # Compute the average distribution M = 0.5*(P + Q).
    M = 0.5 * (P + Q)
    
    # Clamp M to ensure each element >= epsilon and re-normalize so rows sum to 1.
    M = M.clamp(min=epsilon)
    M = M / M.sum(dim=1, keepdim=True)
    
    # Take the log of M for KL divergences.
    log_M = torch.log(M)
    
    # KL(P || M) and KL(Q || M).
    KL_PM = F.kl_div(log_M, P, reduction='batchmean')  # KL(P || M).
    KL_QM = F.kl_div(log_M, Q, reduction='batchmean')  # KL(Q || M).
    
    # Jensen-Shannon Divergence = 0.5 * [KL(P || M) + KL(Q || M)].
    return 0.5 * (KL_PM + KL_QM)


class CombinedLoss(nn.Module):
    """
    Combines Binary Cross-Entropy (BCE) loss with Jensen-Shannon Divergence (JSD) loss.
    
    Includes partial gradient propagation logic for the JSD branch:
      - jsd_grad_factor=0   -> JSD branch is fully detached (no gradient).
      - jsd_grad_factor=1   -> JSD branch has full gradient.
      - 0 < jsd_grad_factor < 1 -> partial gradient scaling.
    """
    def __init__(self, bce_weight: float = 0.7, 
                 jsd_weight: float = 0.3, 
                 pos_weight: torch.Tensor = None, 
                 jsd_grad_factor: float = 0.5):
        """
        Args:
            bce_weight (float): Weight for the BCE loss.
            jsd_weight (float): Weight for the JSD loss.
            pos_weight (torch.Tensor, optional): Positive class weight for BCE.
            jsd_grad_factor (float): Controls gradient flow in JSD branch
                                     (0 = no gradient, 1 = full gradient).
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.jsd_weight = jsd_weight
        self.jsd_grad_factor = jsd_grad_factor
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Computes the combined loss:
          1) BCE loss with raw logits.
          2) JSD between predicted probabilities and labels, 
             with partial gradient flow determined by jsd_grad_factor.
        
        Args:
            logits (torch.Tensor): Raw model outputs (before sigmoid), shape (N,).
            labels (torch.Tensor): Ground truth labels, shape (N,).
            
        Returns:
            torch.Tensor: The combined loss scalar.
        """
        # 1) BCE loss.
        bce = self.bce_loss(logits, labels)
        
        # 2) JSD loss with partial gradient flow.
        full_probs = torch.sigmoid(logits)
        
        if self.jsd_grad_factor == 1.0:
            # Full gradient from JSD branch.
            probs = full_probs
        elif self.jsd_grad_factor == 0.0:
            # Fully detached from the JSD branch, no gradient.
            probs = full_probs.detach()
        else:
            # Partial gradient blend.
            probs = (self.jsd_grad_factor * full_probs +
                     (1 - self.jsd_grad_factor) * full_probs.detach())
        
        jsd = jensen_shannon_divergence(probs, labels)
        
        # Weighted sum.
        return self.bce_weight * bce + self.jsd_weight * jsd
