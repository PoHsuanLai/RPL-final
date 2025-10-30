"""Cross-modal adapter from SELD features to imagination task"""

import torch
import torch.nn as nn


class SELDToImaginationAdapter(nn.Module):
    """
    Adapts pretrained SELD audio features with visual features for imagination task.

    Key features:
    - Cross-modal attention fusion
    - Occlusion-aware reliability weighting
    - Adaptive modality balancing
    """

    def __init__(
        self,
        audio_dim: int = 512,
        visual_dim: int = 2048,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.3
    ):
        """
        Args:
            audio_dim: Dimension of SELD audio features
            visual_dim: Dimension of visual encoder features (ResNet-50: 2048)
            embed_dim: Common embedding dimension for fusion
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.audio_dim = audio_dim
        self.visual_dim = visual_dim
        self.embed_dim = embed_dim

        # Project audio and visual to common dimension
        self.audio_projection = nn.Sequential(
            nn.Linear(audio_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.visual_projection = nn.Sequential(
            nn.Linear(visual_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Cross-modal attention
        # Audio queries attend to visual keys/values
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Reliability weighting network
        # Learns to weight audio vs visual based on occlusion level
        self.reliability_net = nn.Sequential(
            nn.Linear(1, 64),  # Input: occlusion percentage [0, 1]
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # Output: [w_audio, w_visual]
            nn.Softmax(dim=-1)
        )

        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
        occlusion_level: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            audio_features: [batch, audio_dim] from SELD encoder (temporally pooled)
            visual_features: [batch, visual_dim] from visual encoder
            occlusion_level: [batch, 1] occlusion percentage in [0, 1]

        Returns:
            fused_features: [batch, embed_dim * 2] fused representation
        """
        batch_size = audio_features.size(0)

        # Project to common embedding space
        audio_proj = self.audio_projection(audio_features)  # [batch, embed_dim]
        visual_proj = self.visual_projection(visual_features)  # [batch, embed_dim]

        # Cross-modal attention: audio queries attend to visual
        # Add sequence dimension for attention
        audio_query = audio_proj.unsqueeze(1)  # [batch, 1, embed_dim]
        visual_kv = visual_proj.unsqueeze(1)   # [batch, 1, embed_dim]

        attn_out, attn_weights = self.cross_attention(
            query=audio_query,
            key=visual_kv,
            value=visual_kv
        )
        attn_out = attn_out.squeeze(1)  # [batch, embed_dim]

        # Compute reliability weights based on occlusion
        weights = self.reliability_net(occlusion_level)  # [batch, 2]
        w_audio = weights[:, 0:1]  # [batch, 1]
        w_visual = weights[:, 1:2]  # [batch, 1]

        # Weighted combination
        # When occlusion is high, w_audio should be larger
        weighted_audio = w_audio * audio_proj  # [batch, embed_dim]
        weighted_visual = w_visual * attn_out  # [batch, embed_dim]

        # Concatenate weighted features
        fused = torch.cat([weighted_audio, weighted_visual], dim=-1)  # [batch, embed_dim * 2]

        # Final fusion
        fused = self.fusion(fused)  # [batch, embed_dim * 2]

        return fused

    def get_attention_weights(
        self,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Get attention weights for visualization.

        Args:
            audio_features: [batch, audio_dim]
            visual_features: [batch, visual_dim]

        Returns:
            attention_weights: [batch, num_heads, 1, 1] attention map
        """
        with torch.no_grad():
            audio_proj = self.audio_projection(audio_features)
            visual_proj = self.visual_projection(visual_features)

            audio_query = audio_proj.unsqueeze(1)
            visual_kv = visual_proj.unsqueeze(1)

            _, attn_weights = self.cross_attention(
                query=audio_query,
                key=visual_kv,
                value=visual_kv
            )

        return attn_weights
