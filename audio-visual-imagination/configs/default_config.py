"""Default configuration for training"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    num_classes: int = 10
    audio_dim: int = 512
    visual_dim: int = 2048
    embed_dim: int = 512
    num_heads: int = 8
    hidden_dim: int = 512
    dropout: float = 0.3
    pretrained_seld_path: Optional[str] = None


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    # Dataset
    batch_size: int = 32
    num_workers: int = 4

    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    max_epochs: int = 50

    # Two-stage fine-tuning
    freeze_seld_epochs: int = 30  # Freeze SELD for first 30 epochs
    seld_lr: float = 1e-5  # Lower LR for SELD when unfrozen
    adapter_lr: float = 1e-4
    imagination_lr: float = 1e-4

    # Loss weights
    loss_alpha: float = 1.0  # Classification
    loss_beta: float = 2.0   # Localization
    loss_gamma: float = 0.5  # Confidence

    # Scheduler
    scheduler_type: str = "cosine"  # "cosine", "step", or "plateau"
    warmup_epochs: int = 5
    min_lr: float = 1e-6

    # Regularization
    gradient_clip: float = 1.0
    label_smoothing: float = 0.1

    # Logging
    log_interval: int = 10  # Log every N batches
    val_interval: int = 1   # Validate every N epochs
    save_interval: int = 5  # Save checkpoint every N epochs


@dataclass
class DataConfig:
    """Data configuration"""
    # Paths
    starss23_path: str = "./data/starss23"
    soundspaces_path: str = "./data/soundspaces"
    output_dir: str = "./outputs"

    # Occlusion settings
    min_occlusion: float = 0.0
    max_occlusion: float = 1.0
    occlusion_curriculum: bool = True  # Progressive curriculum

    # Audio processing
    sample_rate: int = 24000
    n_fft: int = 1024
    hop_length: int = 320
    n_mels: int = 64
    audio_duration: float = 3.0  # seconds

    # Visual processing
    image_size: tuple = (224, 224)
    normalize_mean: tuple = (0.485, 0.456, 0.406)
    normalize_std: tuple = (0.229, 0.224, 0.225)

    # Event classes
    event_classes: list = None  # Will be populated from dataset

    def __post_init__(self):
        if self.event_classes is None:
            # Default STARSS23-like classes
            self.event_classes = [
                "background",
                "female_speech",
                "male_speech",
                "clapping",
                "telephone",
                "laughter",
                "domestic_sounds",
                "footsteps",
                "door_slam",
                "music",
                "musical_instrument"
            ]


@dataclass
class Config:
    """Main configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # Experiment
    experiment_name: str = "audio_visual_imagination"
    seed: int = 42
    device: str = "cuda"  # "cuda" or "cpu"
    mixed_precision: bool = True  # Use automatic mixed precision


def get_default_config() -> Config:
    """Get default configuration"""
    return Config()
