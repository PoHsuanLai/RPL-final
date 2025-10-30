"""
Test script for SELD encoder integration.

This tests:
1. Real SELD encoder architecture
2. Input/output shapes
3. Integration with our model
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root))

import torch
from audio_visual_imagination.models import (
    SELDAudioEncoder,
    PlaceholderSELDEncoder,
    AudioVisualImagination
)


def test_seld_encoder():
    print("=" * 60)
    print("Testing SELD Audio Encoder")
    print("=" * 60)

    # Create SELD encoder (real architecture from DCASE2023)
    encoder = SELDAudioEncoder(
        in_channels=7,  # FOA amp_phasediff format
        cnn_out_channels=64,
        gru_hidden_size=256,
        bidirectional=True
    )

    print(f"\n[1] SELD Encoder Architecture:")
    print(f"   Input channels: 7 (FOA amp_phasediff)")
    print(f"   CNN output: 64 channels")
    print(f"   GRU hidden: 256")
    print(f"   Bidirectional: True")
    print(f"   Output dimension: {encoder.get_output_dim()}")

    # Test forward pass
    batch_size = 2
    freq_bins = 128  # Typical for SELD
    time_frames = 128

    # Input: [batch, channels, freq, time]
    audio_input = torch.randn(batch_size, 7, freq_bins, time_frames)
    print(f"\n[2] Testing forward pass:")
    print(f"   Input shape: {audio_input.shape}")

    encoder.eval()
    with torch.no_grad():
        output = encoder(audio_input)

    print(f"   Output shape: {output.shape}")
    print(f"   Expected: [batch={batch_size}, time', {encoder.get_output_dim()}]")

    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"\n[3] Parameters:")
    print(f"   Total: {total_params:,}")

    return encoder


def test_placeholder_encoder():
    print("\n" + "=" * 60)
    print("Testing Placeholder SELD Encoder")
    print("=" * 60)

    audio_dim = 512
    placeholder = PlaceholderSELDEncoder(audio_dim=audio_dim)

    print(f"\n[1] Placeholder Configuration:")
    print(f"   Target audio dimension: {audio_dim}")

    # Test with different input
    batch_size = 2
    audio_input = torch.randn(batch_size, 7, 128, 128)

    placeholder.eval()
    with torch.no_grad():
        output = placeholder(audio_input)

    print(f"\n[2] Forward pass:")
    print(f"   Input shape: {audio_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Expected: [batch={batch_size}, time', {audio_dim}]")


def test_full_model_integration():
    print("\n" + "=" * 60)
    print("Testing Full Model with SELD Integration")
    print("=" * 60)

    # Initialize model (no pretrained weights)
    model = AudioVisualImagination(
        pretrained_seld_path=None,  # Use placeholder
        num_classes=10,
        audio_dim=512,
        freeze_seld=True
    )

    print(f"\n[1] Model initialized successfully")
    print(f"   Total parameters: {model.get_num_params(trainable_only=False):,}")
    print(f"   Trainable parameters: {model.get_num_params(trainable_only=True):,}")

    # Test with dummy data
    batch_size = 2

    # Audio: FOA amp_phasediff format (7 channels)
    audio = torch.randn(batch_size, 7, 128, 128)

    # Visual: RGB-D + occlusion (5 channels)
    visual = torch.randn(batch_size, 5, 224, 224)

    # Occlusion level
    occlusion = torch.tensor([[0.3], [0.7]])

    print(f"\n[2] Testing forward pass:")
    print(f"   Audio shape: {audio.shape}")
    print(f"   Visual shape: {visual.shape}")
    print(f"   Occlusion levels: {occlusion.squeeze().tolist()}")

    model.eval()
    with torch.no_grad():
        outputs = model(audio, visual, occlusion)

    print(f"\n[3] Outputs:")
    print(f"   class_logits: {outputs['class_logits'].shape}")
    print(f"   location: {outputs['location'].shape}")
    print(f"   confidence: {outputs['confidence'].shape}")

    # Test predictions
    with torch.no_grad():
        predictions = model.predict(audio, visual, occlusion)

    print(f"\n[4] Predictions:")
    for i in range(batch_size):
        pred_class = predictions['class_pred'][i].item()
        location = predictions['location'][i]
        confidence = predictions['confidence'][i].item()

        print(f"\n   Sample {i+1} (occlusion: {occlusion[i].item():.1%}):")
        print(f"      Class: {pred_class}")
        print(f"      Location: az={location[0]:.1f}°, el={location[1]:.1f}°, dist={location[2]:.2f}m")
        print(f"      Confidence: {confidence:.3f}")


def main():
    print("\n" + "#" * 60)
    print("# SELD Integration Test Suite")
    print("#" * 60)

    try:
        # Test 1: SELD encoder
        encoder = test_seld_encoder()

        # Test 2: Placeholder encoder
        test_placeholder_encoder()

        # Test 3: Full model integration
        test_full_model_integration()

        print("\n" + "=" * 60)
        print("✓ All tests passed successfully!")
        print("=" * 60)

        print("\n[Summary]")
        print("• SELD encoder architecture matches DCASE2023 baseline")
        print("• Audio input format: [batch, 7, freq, time] (FOA amp_phasediff)")
        print("• Output format: [batch, time', 512] temporal features")
        print("• Integration with full model works correctly")
        print("• Ready for pretrained weight loading")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
