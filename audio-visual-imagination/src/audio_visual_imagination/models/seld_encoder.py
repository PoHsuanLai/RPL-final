"""
SELD Audio Encoder ported from DCASE2023 baseline.

Original implementation:
https://github.com/sony/audio-visual-seld-dcase2023

This is the audio-only portion of the AudioVisualCRNN model,
which we will use as a pretrained encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN3(nn.Module):
    """
    3-layer CNN encoder for audio features.

    This is the audio encoder from the DCASE2023 SELD baseline.
    It processes audio spectrograms (amplitude + phase difference features).
    """

    def __init__(
        self,
        in_channels: int = 7,  # amp_phasediff format: 7 channels
        out_channels: int = 64,
        kernel_size: tuple = (3, 3),
        stride: tuple = (1, 1),
        padding: tuple = (1, 1)
    ):
        """
        Args:
            in_channels: Input channels (7 for FOA amp_phasediff)
            out_channels: Output embedding dimension
            kernel_size: Conv kernel size
            stride: Conv stride
            padding: Conv padding
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.conv3 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """
        Args:
            x: [batch, in_channels, freq, time]

        Returns:
            features: [batch, out_channels, freq', time']
        """
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=(4, 4))

        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=(2, 4))

        x = F.relu_(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, kernel_size=(2, 2))

        return x


class SELDAudioEncoder(nn.Module):
    """
    Complete SELD audio encoder: CNN + GRU

    This matches the architecture from DCASE2023 baseline,
    extracting the audio-only portion for our transfer learning.
    """

    def __init__(
        self,
        in_channels: int = 7,  # FOA amp_phasediff
        cnn_out_channels: int = 64,
        gru_hidden_size: int = 256,
        gru_num_layers: int = 1,
        bidirectional: bool = True
    ):
        """
        Args:
            in_channels: Audio feature channels (7 for FOA amp_phasediff)
            cnn_out_channels: CNN output channels
            gru_hidden_size: GRU hidden dimension
            gru_num_layers: Number of GRU layers
            bidirectional: Use bidirectional GRU
        """
        super().__init__()

        self.cnn_out_channels = cnn_out_channels
        self.gru_hidden_size = gru_hidden_size
        self.bidirectional = bidirectional

        # CNN encoder
        self.cnn = CNN3(
            in_channels=in_channels,
            out_channels=cnn_out_channels
        )

        # GRU temporal modeling
        self.gru = nn.GRU(
            input_size=cnn_out_channels,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Output dimension
        self.output_dim = gru_hidden_size * 2 if bidirectional else gru_hidden_size

    def forward(self, x):
        """
        Forward pass through SELD audio encoder.

        Args:
            x: [batch, channels, freq, time] audio spectrogram
               For FOA amp_phasediff: [batch, 7, 128, ~128]

        Returns:
            features: [batch, time', output_dim] temporal features
        """
        # Transpose to match SELD format: [batch, channels, time, freq]
        x = x.transpose(2, 3)  # [batch, channels, freq, time] -> [batch, channels, time, freq]

        # CNN encoding
        x = self.cnn(x)  # [batch, cnn_out, time', freq']

        # Average over frequency dimension
        x = torch.mean(x, dim=3)  # [batch, cnn_out, time']

        # Transpose for GRU: [batch, time, features]
        x = x.transpose(1, 2)  # [batch, time', cnn_out]

        # GRU temporal modeling
        self.gru.flatten_parameters()
        x, _ = self.gru(x)  # [batch, time', output_dim]

        return x

    def get_output_dim(self) -> int:
        """Get output feature dimension"""
        return self.output_dim


def load_seld_checkpoint(checkpoint_path: str, model: SELDAudioEncoder) -> SELDAudioEncoder:
    """
    Load pretrained SELD weights into the encoder.

    Args:
        checkpoint_path: Path to SELD checkpoint (.pth file)
        model: SELDAudioEncoder instance

    Returns:
        model: Model with loaded weights
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # The checkpoint might contain the full AudioVisualCRNN
    # We need to extract only the audio encoder parts
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Filter to keep only audio encoder weights
    audio_encoder_state = {}
    for key, value in state_dict.items():
        # Keep weights that belong to audio_encoder or cnn/gru
        if 'audio_encoder' in key:
            # Remove 'audio_encoder.' prefix
            new_key = key.replace('audio_encoder.', '')
            audio_encoder_state[new_key] = value
        elif any(x in key for x in ['conv', 'bn', 'gru']) and 'vision' not in key and 'fc_xyz' not in key:
            # Direct CNN/GRU weights
            audio_encoder_state[key] = value

    # Load weights
    missing, unexpected = model.load_state_dict(audio_encoder_state, strict=False)

    print(f"Loaded SELD audio encoder from {checkpoint_path}")
    if missing:
        print(f"  Missing keys: {missing}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")

    return model


# For backward compatibility with our placeholder
class PlaceholderSELDEncoder(nn.Module):
    """
    Simplified placeholder that matches the real SELD output format.
    Use this when no pretrained weights are available.
    """

    def __init__(self, audio_dim: int = 512):
        super().__init__()
        self.encoder = SELDAudioEncoder(
            in_channels=7,  # Assume FOA amp_phasediff
            cnn_out_channels=64,
            gru_hidden_size=256,
            bidirectional=True
        )
        self.audio_dim = audio_dim

        # If output_dim doesn't match desired audio_dim, add projection
        if self.encoder.output_dim != audio_dim:
            self.projection = nn.Linear(self.encoder.output_dim, audio_dim)
        else:
            self.projection = None

    def forward(self, x):
        """
        Args:
            x: [batch, channels, freq, time]

        Returns:
            features: [batch, time, audio_dim]
        """
        x = self.encoder(x)

        if self.projection is not None:
            x = self.projection(x)

        return x
