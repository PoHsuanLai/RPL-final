"""Full audio-visual imagination model with pretrained SELD encoder"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from typing import Dict, Optional

from .adapter import SELDToImaginationAdapter
from .imagination_head import ImaginationHead


class AudioVisualImagination(nn.Module):
    """
    Full model combining:
    1. Pretrained SELD encoder (frozen initially)
    2. Visual encoder (ResNet-50)
    3. Cross-modal adapter
    4. Imagination head
    """

    def __init__(
        self,
        pretrained_seld_path: Optional[str] = None,
        num_classes: int = 10,
        audio_dim: int = 512,
        freeze_seld: bool = True,
        freeze_visual: bool = False
    ):
        """
        Args:
            pretrained_seld_path: Path to pretrained SELD checkpoint (None = placeholder)
            num_classes: Number of hazard event classes
            audio_dim: Dimension of SELD audio features
            freeze_seld: Whether to freeze SELD encoder initially
            freeze_visual: Whether to freeze visual encoder
        """
        super().__init__()

        self.audio_dim = audio_dim
        self.num_classes = num_classes

        # ========== AUDIO ENCODER (SELD) ==========
        # TODO: Load actual pretrained SELD encoder
        # For now, create placeholder that matches SELD output
        if pretrained_seld_path is not None:
            self.seld_encoder = self._load_seld_encoder(pretrained_seld_path)
        else:
            # Placeholder: simple CNN + GRU that mimics SELD architecture
            self.seld_encoder = self._create_placeholder_seld()

        if freeze_seld:
            self.freeze_seld()

        # ========== VISUAL ENCODER (ResNet-50) ==========
        # Input: RGB-D (4 channels) + Occlusion Mask (1 channel) = 5 channels
        self.visual_encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Modify first conv layer to accept 5 channels instead of 3
        original_conv = self.visual_encoder.conv1
        self.visual_encoder.conv1 = nn.Conv2d(
            in_channels=5,  # RGB + Depth + Occlusion
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )

        # Initialize new channels with pretrained RGB weights
        with torch.no_grad():
            # Copy RGB weights
            self.visual_encoder.conv1.weight[:, :3] = original_conv.weight
            # Initialize depth and occlusion channels with mean of RGB
            self.visual_encoder.conv1.weight[:, 3:] = original_conv.weight.mean(dim=1, keepdim=True)

        # Remove final classification layer
        visual_dim = self.visual_encoder.fc.in_features
        self.visual_encoder.fc = nn.Identity()

        if freeze_visual:
            self.freeze_visual()

        # ========== ADAPTER ==========
        self.adapter = SELDToImaginationAdapter(
            audio_dim=audio_dim,
            visual_dim=visual_dim,
            embed_dim=512,
            num_heads=8
        )

        # ========== IMAGINATION HEAD ==========
        self.imagination_head = ImaginationHead(
            input_dim=1024,  # 512 * 2 from adapter
            num_classes=num_classes,
            hidden_dim=512
        )

    def _create_placeholder_seld(self) -> nn.Module:
        """
        Creates a placeholder SELD encoder.
        Replace this with actual SELD model loading.

        Expected input: 4-channel FOA audio spectrogram [batch, 4, freq, time]
        Expected output: [batch, time, audio_dim] features
        """
        class PlaceholderSELD(nn.Module):
            def __init__(self, audio_dim):
                super().__init__()
                # Simple CNN + GRU placeholder
                self.conv = nn.Sequential(
                    nn.Conv2d(4, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, None))  # Pool frequency dimension
                )
                self.gru = nn.GRU(256, audio_dim // 2, num_layers=2, bidirectional=True, batch_first=True)

            def forward(self, x):
                # x: [batch, 4, freq, time]
                x = self.conv(x)  # [batch, 256, 1, time]
                x = x.squeeze(2).permute(0, 2, 1)  # [batch, time, 256]
                x, _ = self.gru(x)  # [batch, time, audio_dim]
                return x

        return PlaceholderSELD(self.audio_dim)

    def _load_seld_encoder(self, checkpoint_path: str) -> nn.Module:
        """
        Load pretrained SELD encoder from checkpoint.
        TODO: Implement based on specific SELD model architecture
        """
        # This will be implemented once we download actual SELD checkpoint
        raise NotImplementedError("SELD loading will be implemented after downloading checkpoint")

    def freeze_seld(self):
        """Freeze SELD encoder parameters"""
        for param in self.seld_encoder.parameters():
            param.requires_grad = False

    def unfreeze_seld(self):
        """Unfreeze all SELD encoder parameters"""
        for param in self.seld_encoder.parameters():
            param.requires_grad = True

    def unfreeze_seld_top(self, ratio: float = 0.5):
        """
        Unfreeze top portion of SELD encoder for fine-tuning.

        Args:
            ratio: Fraction of parameters to unfreeze (from the end)
        """
        params = list(self.seld_encoder.parameters())
        split_idx = int(len(params) * (1 - ratio))

        # Freeze bottom
        for param in params[:split_idx]:
            param.requires_grad = False

        # Unfreeze top
        for param in params[split_idx:]:
            param.requires_grad = True

    def freeze_visual(self):
        """Freeze visual encoder parameters"""
        for param in self.visual_encoder.parameters():
            param.requires_grad = False

    def unfreeze_visual(self):
        """Unfreeze visual encoder parameters"""
        for param in self.visual_encoder.parameters():
            param.requires_grad = True

    def temporal_pool(self, audio_features: torch.Tensor, method: str = 'mean') -> torch.Tensor:
        """
        Pool temporal dimension of SELD output.

        Args:
            audio_features: [batch, time, audio_dim]
            method: 'mean', 'max', or 'attention'

        Returns:
            pooled: [batch, audio_dim]
        """
        if method == 'mean':
            return audio_features.mean(dim=1)
        elif method == 'max':
            return audio_features.max(dim=1)[0]
        elif method == 'attention':
            # Simple attention pooling
            weights = torch.softmax(audio_features.mean(dim=-1), dim=1)  # [batch, time]
            return (audio_features * weights.unsqueeze(-1)).sum(dim=1)
        else:
            raise ValueError(f"Unknown pooling method: {method}")

    def forward(
        self,
        audio: torch.Tensor,
        visual: torch.Tensor,
        occlusion_level: torch.Tensor,
        pool_method: str = 'mean'
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through full model.

        Args:
            audio: [batch, 4, freq, time] FOA spectrogram
            visual: [batch, 5, H, W] RGB-D + occlusion mask
            occlusion_level: [batch, 1] occlusion percentage [0, 1]
            pool_method: Temporal pooling method for audio

        Returns:
            Dictionary with imagination outputs
        """
        # Extract audio features
        audio_features = self.seld_encoder(audio)  # [batch, time, audio_dim]
        audio_pooled = self.temporal_pool(audio_features, method=pool_method)  # [batch, audio_dim]

        # Extract visual features
        visual_features = self.visual_encoder(visual)  # [batch, visual_dim]

        # Cross-modal fusion
        fused = self.adapter(audio_pooled, visual_features, occlusion_level)  # [batch, 1024]

        # Predict imagination outputs
        outputs = self.imagination_head(fused)

        return outputs

    def predict(
        self,
        audio: torch.Tensor,
        visual: torch.Tensor,
        occlusion_level: torch.Tensor,
        temperature: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions with post-processing.

        Args:
            audio: [batch, 4, freq, time]
            visual: [batch, 5, H, W]
            occlusion_level: [batch, 1]
            temperature: Softmax temperature

        Returns:
            Processed predictions
        """
        # Get raw outputs
        outputs = self.forward(audio, visual, occlusion_level)

        # Post-process
        audio_features = self.seld_encoder(audio)
        audio_pooled = self.temporal_pool(audio_features)
        visual_features = self.visual_encoder(visual)
        fused = self.adapter(audio_pooled, visual_features, occlusion_level)

        predictions = self.imagination_head.predict(fused, temperature=temperature)

        return predictions

    def get_num_params(self, trainable_only: bool = True) -> int:
        """Count model parameters"""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
