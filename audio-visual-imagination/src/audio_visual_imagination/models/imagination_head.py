"""Multi-task imagination head for event prediction"""

import torch
import torch.nn as nn
from typing import Dict


class ImaginationHead(nn.Module):
    """
    Multi-task head for predicting hidden hazard properties.

    Predicts:
    1. Event class (what is the hazard?)
    2. 3D location (where is it?)
    3. Confidence (how certain are we?)
    """

    def __init__(
        self,
        input_dim: int = 1024,
        num_classes: int = 10,
        hidden_dim: int = 512,
        dropout: float = 0.3
    ):
        """
        Args:
            input_dim: Dimension of fused features from adapter
            num_classes: Number of hazard event classes
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.num_classes = num_classes

        # Shared feature extraction
        self.shared_backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Task 1: Event Classification
        # Predict which type of hazard (vehicle, alarm, crash, etc.)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        # Task 2: 3D Localization
        # Predict [azimuth, elevation, distance]
        # Azimuth: [-180, 180] degrees
        # Elevation: [-90, 90] degrees
        # Distance: [0, inf) meters
        self.localizer = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

        # Task 3: Uncertainty/Confidence Estimation
        # How confident is the model about this prediction?
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def forward(self, fused_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            fused_features: [batch, input_dim] from adapter

        Returns:
            Dictionary containing:
                - 'class_logits': [batch, num_classes] unnormalized class scores
                - 'location': [batch, 3] predicted [azimuth, elevation, distance]
                - 'confidence': [batch, 1] prediction confidence in [0, 1]
        """
        # Extract shared features
        shared_feat = self.shared_backbone(fused_features)  # [batch, hidden_dim]

        # Multi-task predictions
        class_logits = self.classifier(shared_feat)  # [batch, num_classes]
        location = self.localizer(shared_feat)       # [batch, 3]
        confidence = self.confidence_head(shared_feat)  # [batch, 1]

        return {
            'class_logits': class_logits,
            'location': location,
            'confidence': confidence
        }

    def predict(
        self,
        fused_features: torch.Tensor,
        temperature: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions with post-processing.

        Args:
            fused_features: [batch, input_dim]
            temperature: Temperature for softmax (higher = more uncertain)

        Returns:
            Dictionary with processed outputs:
                - 'class_probs': [batch, num_classes] class probabilities
                - 'class_pred': [batch] predicted class indices
                - 'location': [batch, 3] location prediction
                - 'confidence': [batch, 1] confidence score
        """
        outputs = self.forward(fused_features)

        # Apply softmax with temperature
        class_probs = torch.softmax(outputs['class_logits'] / temperature, dim=-1)
        class_pred = torch.argmax(class_probs, dim=-1)

        # Normalize location predictions
        location = outputs['location'].clone()
        # Azimuth: tanh * 180 -> [-180, 180]
        location[:, 0] = torch.tanh(location[:, 0]) * 180.0
        # Elevation: tanh * 90 -> [-90, 90]
        location[:, 1] = torch.tanh(location[:, 1]) * 90.0
        # Distance: softplus -> [0, inf)
        location[:, 2] = torch.nn.functional.softplus(location[:, 2])

        return {
            'class_probs': class_probs,
            'class_pred': class_pred,
            'location': location,
            'confidence': outputs['confidence']
        }
