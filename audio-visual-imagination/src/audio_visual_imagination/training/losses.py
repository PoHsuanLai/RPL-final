"""Loss functions for imagination task"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class ImaginationLoss(nn.Module):
    """
    Multi-task loss for imagination prediction.

    Combines:
    1. Classification loss (CrossEntropy)
    2. Localization loss (MSE for angles + distance)
    3. Confidence calibration loss
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 2.0,
        gamma: float = 0.5,
        class_weights: torch.Tensor = None
    ):
        """
        Args:
            alpha: Weight for classification loss
            beta: Weight for localization loss
            gamma: Weight for confidence loss
            class_weights: Optional class weights for imbalanced data
        """
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.class_weights = class_weights
        self.classification_loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.

        Args:
            outputs: Dictionary from model with keys:
                - 'class_logits': [batch, num_classes]
                - 'location': [batch, 3] (azimuth, elevation, distance)
                - 'confidence': [batch, 1]
            targets: Dictionary with ground truth:
                - 'class': [batch] class indices
                - 'location': [batch, 3] true location
                - 'is_present': [batch] binary (1 if event present, 0 if background)

        Returns:
            Dictionary with individual and total losses
        """
        # ========== Classification Loss ==========
        loss_cls = self.classification_loss(
            outputs['class_logits'],
            targets['class']
        )

        # ========== Localization Loss ==========
        # Only compute for present events (not background)
        mask = targets['is_present'].bool()

        if mask.sum() > 0:
            pred_loc = outputs['location'][mask]  # [num_present, 3]
            true_loc = targets['location'][mask]  # [num_present, 3]

            # Separate losses for angles and distance
            # Azimuth and elevation: MSE on normalized angles
            angle_loss = F.mse_loss(pred_loc[:, :2], true_loc[:, :2])

            # Distance: log-scale MSE (better for varying distances)
            dist_pred = torch.nn.functional.softplus(pred_loc[:, 2]) + 1e-6
            dist_true = true_loc[:, 2] + 1e-6
            distance_loss = F.mse_loss(torch.log(dist_pred), torch.log(dist_true))

            loss_loc = angle_loss + distance_loss
        else:
            loss_loc = torch.tensor(0.0, device=outputs['location'].device)

        # ========== Confidence Calibration Loss ==========
        # Confidence should match actual correctness
        pred_conf = outputs['confidence'].squeeze(-1)  # [batch]

        # Compute actual correctness
        pred_class = outputs['class_logits'].argmax(dim=-1)  # [batch]
        is_correct = (pred_class == targets['class']).float()  # [batch]

        # MSE between predicted confidence and actual correctness
        loss_conf = F.mse_loss(pred_conf, is_correct)

        # ========== Total Loss ==========
        total_loss = self.alpha * loss_cls + self.beta * loss_loc + self.gamma * loss_conf

        return {
            'total': total_loss,
            'classification': loss_cls,
            'localization': loss_loc,
            'confidence': loss_conf
        }


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Alternative to standard cross-entropy.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [batch, num_classes] logits
            targets: [batch] class indices

        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
