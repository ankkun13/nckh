"""
Focal Loss for Class Imbalance
===============================
Cancer driver genes are rare (~2-5% of all genes), creating severe
class imbalance. Focal Loss down-weights easy negatives and focuses
on hard-to-classify examples.

Reference: Lin et al., 2017, "Focal Loss for Dense Object Detection"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Weighting factor for the rare class (default: 0.25).
        gamma: Focusing parameter (default: 2.0).
               gamma=0 recovers standard cross-entropy.
        reduction: 'mean', 'sum', or 'none'.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            logits: Raw model output (n_nodes x 2).
            targets: Ground truth labels (n_nodes,) with values 0 or 1.
            mask: Optional boolean mask (n_nodes,) for labeled nodes only.
        
        Returns:
            Scalar loss value.
        """
        targets = targets.long()
        
        # Compute softmax probabilities
        probs = F.softmax(logits, dim=1)
        
        # Gather probabilities of true classes
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute alpha weight (higher for positive/driver class)
        alpha_t = torch.where(
            targets == 1,
            torch.tensor(self.alpha, device=logits.device),
            torch.tensor(1 - self.alpha, device=logits.device)
        )
        
        # Focal loss
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        focal_loss = alpha_t * focal_weight * ce_loss
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.float()
            focal_loss = focal_loss * mask
            if self.reduction == 'mean':
                return focal_loss.sum() / mask.sum().clamp(min=1)
            elif self.reduction == 'sum':
                return focal_loss.sum()
        else:
            if self.reduction == 'mean':
                return focal_loss.mean()
            elif self.reduction == 'sum':
                return focal_loss.sum()
        
        return focal_loss


class MaskedCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss with masks and positive class weighting.
    Compatible with MODCAN's original loss function.
    """
    
    def __init__(self, pos_weight: float = 2.0):
        super(MaskedCrossEntropyLoss, self).__init__()
        self.pos_weight = pos_weight
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw output (n_nodes x 2).
            targets: Labels (n_nodes,).
            mask: Boolean mask (n_nodes,).
        """
        targets = targets.long()
        
        # Weight positive examples higher
        pos_weight = torch.ones(logits.shape[0], device=logits.device) + \
                     (self.pos_weight - 1) * targets.float()
        
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        weighted_loss = pos_weight * ce_loss
        
        mask_float = mask.float()
        mask_float = mask_float / mask_float.mean().clamp(min=1e-8)
        
        loss = (weighted_loss * mask_float).sum() / logits.shape[0]
        return loss
