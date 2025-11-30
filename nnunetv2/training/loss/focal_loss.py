import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):

    def __init__(self, alpha=.25, gamma=2., reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits, target):
        """
        Args:
            logits: (B, C, H, W, D) - raw model outputs
            target: (B, H, W, D) - integer class labels
        """
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=1)  # (B, C, H, W, D)
        
        # Get log probabilities (more numerically stable)
        log_probs = F.log_softmax(logits, dim=1)  # (B, C, H, W, D)
        
        # Gather the probabilities for the correct classes
        # This is equivalent to indexing with target
        target_expanded = target.unsqueeze(1).to(torch.int)  # (B, 1, H, W, D)
        pt = probs.gather(1, target_expanded).squeeze(1)  # (B, H, W, D)
        log_pt = log_probs.gather(1, target_expanded).squeeze(1)  # (B, H, W, D)
        
        # Compute focal loss
        focal_weight = (1 - pt) ** self.gamma
        loss = -self.alpha * focal_weight * log_pt
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss