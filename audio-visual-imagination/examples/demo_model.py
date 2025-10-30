"""
Demo script showing how to use the Audio-Visual Imagination model.

This demonstrates:
1. Model initialization
2. Creating dummy input data
3. Forward pass
4. Making predictions
5. Inspecting model parameters
"""

import sys
from pathlib import Path

# Add src and configs to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root))

import torch
from audio_visual_imagination.models import AudioVisualImagination
from configs.default_config import get_default_config


def main():
    print("=" * 60)
    print("Audio-Visual Imagination Model Demo")
    print("=" * 60)

    # Load configuration
    config = get_default_config()

    # Initialize model
    print("\n[1] Initializing model...")
    model = AudioVisualImagination(
        pretrained_seld_path=None,  # Using placeholder for now
        num_classes=config.model.num_classes,
        freeze_seld=True
    )

    # Print model info
    total_params = model.get_num_params(trainable_only=False)
    trainable_params = model.get_num_params(trainable_only=True)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Frozen parameters: {total_params - trainable_params:,}")

    # Create dummy input data
    print("\n[2] Creating dummy input data...")
    batch_size = 4

    # Audio: 4-channel FOA spectrogram
    # Shape: [batch, channels, freq_bins, time_frames]
    audio = torch.randn(batch_size, 4, 64, 100)
    print(f"   Audio shape: {audio.shape}")

    # Visual: RGB-D + occlusion mask
    # Shape: [batch, 5, height, width]
    # Channels: [R, G, B, Depth, Occlusion_Mask]
    visual = torch.randn(batch_size, 5, 224, 224)
    print(f"   Visual shape: {visual.shape}")

    # Occlusion level: percentage of scene occluded [0, 1]
    occlusion_level = torch.tensor([
        [0.1],  # Low occlusion
        [0.5],  # Medium occlusion
        [0.8],  # High occlusion
        [1.0],  # Complete occlusion
    ])
    print(f"   Occlusion levels: {occlusion_level.squeeze().tolist()}")

    # Forward pass
    print("\n[3] Running forward pass...")
    model.eval()
    with torch.no_grad():
        outputs = model(audio, visual, occlusion_level)

    print("   Outputs:")
    print(f"   - class_logits: {outputs['class_logits'].shape}")
    print(f"   - location: {outputs['location'].shape}")
    print(f"   - confidence: {outputs['confidence'].shape}")

    # Make predictions with post-processing
    print("\n[4] Making predictions...")
    with torch.no_grad():
        predictions = model.predict(audio, visual, occlusion_level, temperature=1.0)

    print("   Predictions for each sample:")
    for i in range(batch_size):
        pred_class = predictions['class_pred'][i].item()
        location = predictions['location'][i]
        confidence = predictions['confidence'][i].item()

        print(f"\n   Sample {i+1} (occlusion: {occlusion_level[i].item():.1%}):")
        print(f"      Predicted class: {pred_class}")
        print(f"      Location: azimuth={location[0]:.1f}°, "
              f"elevation={location[1]:.1f}°, distance={location[2]:.2f}m")
        print(f"      Confidence: {confidence:.3f}")

    # Demonstrate two-stage training setup
    print("\n[5] Demonstrating training configuration...")

    print("\n   Stage 1 (SELD frozen):")
    model.freeze_seld()
    trainable_stage1 = model.get_num_params(trainable_only=True)
    print(f"      Trainable parameters: {trainable_stage1:,}")

    print("\n   Stage 2 (top 50% SELD unfrozen):")
    model.unfreeze_seld_top(ratio=0.5)
    trainable_stage2 = model.get_num_params(trainable_only=True)
    print(f"      Trainable parameters: {trainable_stage2:,}")
    print(f"      Additional unfrozen: {trainable_stage2 - trainable_stage1:,}")

    # Inspect adapter reliability weighting
    print("\n[6] Testing occlusion-aware weighting...")
    model.eval()

    test_occlusions = torch.tensor([[0.0], [0.25], [0.5], [0.75], [1.0]])
    audio_test = torch.randn(5, 512)
    visual_test = torch.randn(5, 2048)

    with torch.no_grad():
        # Get weights from adapter
        weights = model.adapter.reliability_net(test_occlusions)

    print("   Occlusion → Audio Weight / Visual Weight:")
    for occ, weight in zip(test_occlusions, weights):
        w_audio, w_visual = weight[0].item(), weight[1].item()
        print(f"      {occ.item():.2f} → {w_audio:.3f} / {w_visual:.3f}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
