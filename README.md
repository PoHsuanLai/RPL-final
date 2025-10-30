# Audio-Visual Cross-Modal Imagination for Robot Hazard Avoidance

This project implements a novel approach for robots to "imagine" hidden hazards by reasoning about sounds when vision is blocked, enabling safer navigation around corners and obstacles.

## Key Features

- **Transfer Learning**: Uses pretrained SELD (Sound Event Localization & Detection) models
- **Cross-Modal Fusion**: Adaptive fusion of audio and visual modalities with occlusion-aware weighting
- **Multi-Task Learning**: Simultaneous prediction of event class, 3D location, and confidence
- **Two-Stage Fine-Tuning**: Efficient adaptation strategy for occluded scenarios

## Architecture Overview

```
Audio (4-ch FOA) ─→ [Pretrained SELD Encoder] ─┐
                                                 ├─→ [Adapter] ─→ [Imagination Head] ─→ Predictions
RGB-D + Occlusion ─→ [ResNet-50 Visual] ────────┘
```

### Components

1. **SELD Encoder** (Pretrained, frozen initially)
   - Extracts audio-spatial features from 4-channel First-Order Ambisonics
   - Temporal modeling with Bi-GRU

2. **Visual Encoder** (ResNet-50)
   - Processes RGB-D + occlusion mask (5 channels)
   - Pretrained on ImageNet, adapted for depth input

3. **Cross-Modal Adapter**
   - Multi-head cross-attention fusion
   - Reliability-aware weighting based on occlusion level
   - Learns to emphasize audio when vision fails

4. **Imagination Head**
   - Event classification (10 classes)
   - 3D localization (azimuth, elevation, distance)
   - Confidence estimation

## Installation

### Prerequisites

- Python >= 3.11
- CUDA-capable GPU (recommended)
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

```bash
# Clone repository
cd audio-visual-imagination

# Dependencies are already installed via uv
# Verify installation
uv run python -c "import torch; print(torch.cuda.is_available())"
```

### Additional Dependencies (Optional)

For full functionality:

```bash
# SoundSpaces 2.0 + Habitat-Sim
uv add habitat-sim habitat-lab soundspaces

# Additional audio processing
uv add pesq pystoi  # For audio quality metrics
```

## Project Structure

```
audio-visual-imagination/
├── src/audio_visual_imagination/
│   ├── models/
│   │   ├── adapter.py              # Cross-modal fusion
│   │   ├── imagination_head.py     # Multi-task prediction
│   │   └── full_model.py           # Complete model
│   ├── training/
│   │   ├── losses.py               # Multi-task losses
│   │   └── trainer.py              # Training loop (TODO)
│   ├── evaluation/
│   │   └── metrics.py              # Evaluation metrics (TODO)
│   ├── data/
│   │   ├── dataset.py              # Dataset classes (TODO)
│   │   └── preprocessing.py        # Audio/visual preprocessing (TODO)
│   └── utils/
├── configs/
│   └── default_config.py           # Configuration
├── experiments/                     # Experiment scripts (TODO)
└── README.md
```

## Usage

### Quick Start

```python
from audio_visual_imagination.models import AudioVisualImagination
import torch

# Initialize model
model = AudioVisualImagination(
    pretrained_seld_path=None,  # Will use placeholder
    num_classes=10,
    freeze_seld=True
)

# Example input
audio = torch.randn(2, 4, 64, 100)  # [batch, channels, freq, time]
visual = torch.randn(2, 5, 224, 224)  # [batch, 5 (RGB-D + mask), H, W]
occlusion = torch.tensor([[0.7], [0.3]])  # [batch, 1] occlusion level

# Forward pass
outputs = model(audio, visual, occlusion)
print(outputs.keys())  # ['class_logits', 'location', 'confidence']

# Get predictions with post-processing
predictions = model.predict(audio, visual, occlusion)
print(f"Predicted class: {predictions['class_pred']}")
print(f"Location: {predictions['location']}")
print(f"Confidence: {predictions['confidence']}")
```

### Training (Two-Stage)

**Stage 1: Train adapter only (SELD frozen)**

```python
from audio_visual_imagination.training import ImaginationLoss, Trainer
from configs.default_config import get_default_config

config = get_default_config()

# Initialize model
model = AudioVisualImagination(freeze_seld=True)

# Train for 30 epochs with SELD frozen
# trainer = Trainer(model, config)
# trainer.train(stage=1)
```

**Stage 2: Fine-tune top SELD layers**

```python
# Unfreeze top 50% of SELD
model.unfreeze_seld_top(ratio=0.5)

# Continue training with lower learning rate
# trainer.train(stage=2, start_epoch=31)
```

## Next Steps

### Immediate TODOs

1. **Download Pretrained SELD**
   - STARSS23 baseline: https://github.com/sharathadavanne/seld-dcase2023
   - EINV2: https://github.com/Jinbo-Hu/EINV2
   - Implement `_load_seld_encoder()` in `full_model.py`

2. **Dataset Preparation**
   - Download STARSS23 dataset
   - Create occlusion scenario generator
   - Implement DataLoader

3. **Training Pipeline**
   - Complete `Trainer` class
   - Add TensorBoard logging
   - Implement checkpointing

4. **Evaluation**
   - Metrics: F1, MAE, calibration
   - Baselines: audio-only, visual-only
   - Qualitative visualization

### Research Questions to Explore

1. How does performance degrade with increasing occlusion?
2. What is the optimal freeze/unfreeze strategy for SELD?
3. Can the model generalize to novel event types?
4. How does reliability weighting adapt across occlusion levels?

## Model Architecture Details

See [ARCHITECTURE_TRANSFER_LEARNING.md](ARCHITECTURE_TRANSFER_LEARNING.md) for:
- Detailed transfer learning strategy
- Adapter design rationale
- Training procedure
- Expected performance

## References

1. **STARSS23**: Shimada et al., "STARSS23 Dataset," DCASE 2023
2. **SoundSpaces 2.0**: Chen et al., NeurIPS 2022
3. **ResNet**: He et al., "Deep Residual Learning," CVPR 2016

## License

[Your License Here]

## Citation

```bibtex
@article{your_paper,
  title={Audio-Visual Cross-Modal Imagination for Robot Hazard Avoidance Under Occlusion},
  author={Your Name},
  year={2025}
}
```

## Contact

[Your contact info]
