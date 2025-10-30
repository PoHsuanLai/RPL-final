#!/usr/bin/env python3
"""
Training script for Audio-Visual Imagination with Two-Stage Fine-Tuning
"""

import sys
from pathlib import Path

# Add paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root.parent / 'sound-spaces'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from datetime import datetime

from audio_visual_imagination.models.full_model import AudioVisualImagination
from audio_visual_imagination.training.losses import ImaginationLoss
from audio_visual_imagination.configs.default_config import TrainingConfig, ModelConfig
from seld_dataset_loader import create_dataloaders


class TwoStageTrainer:
    """
    Two-stage fine-tuning trainer for Audio-Visual Imagination
    Stage 1: Freeze SELD encoder, train adapter + imagination head
    Stage 2: Unfreeze top 50% of SELD encoder, train end-to-end
    """

    def __init__(
        self,
        model_config: ModelConfig,
        train_config: TrainingConfig,
        data_dir: str,
        save_dir: str = "checkpoints",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model_config = model_config
        self.train_config = train_config
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model
        self.model = AudioVisualImagination(
            pretrained_seld_path=model_config.pretrained_seld_path,
            num_classes=model_config.num_classes,
            audio_dim=model_config.audio_dim,
            visual_dim=model_config.visual_dim,
            embed_dim=model_config.embed_dim,
            freeze_seld=True,  # Start with frozen SELD
        ).to(device)

        print(f"\n✓ Model initialized on {device}")
        self._print_model_info()

        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            data_dir=data_dir,
            batch_size=train_config.batch_size,
            num_workers=train_config.num_workers,
        )

        # Initialize loss
        self.criterion = ImaginationLoss(
            alpha=train_config.loss_weights['classification'],
            beta=train_config.loss_weights['localization'],
            gamma=train_config.loss_weights['confidence'],
        ).to(device)

        # TensorBoard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(f"runs/imagination_{timestamp}")

        # Training state
        self.current_epoch = 0
        self.current_stage = 1
        self.best_val_loss = float('inf')

    def _print_model_info(self):
        """Print model parameter information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Frozen parameters: {total_params - trainable_params:,}")

    def _get_optimizer(self, stage: int):
        """Get optimizer for current stage"""
        if stage == 1:
            # Stage 1: Only train adapter + imagination head
            trainable_params = [
                {'params': self.model.adapter.parameters(), 'lr': self.train_config.adapter_lr},
                {'params': self.model.imagination_head.parameters(), 'lr': self.train_config.adapter_lr},
            ]
            print(f"\n=== Stage 1: Training adapter + imagination head ===")
            print(f"  SELD encoder: FROZEN")
            print(f"  Learning rate: {self.train_config.adapter_lr}")

        else:
            # Stage 2: Train top 50% of SELD + adapter + imagination head
            print(f"\n=== Stage 2: Fine-tuning with unfrozen SELD (top 50%) ===")

            # Unfreeze top 50% of SELD
            seld_params = list(self.model.seld_encoder.parameters())
            num_seld_params = len(seld_params)
            unfreeze_from = num_seld_params // 2

            for i, param in enumerate(seld_params):
                if i >= unfreeze_from:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            # Different learning rates for different components
            trainable_params = [
                {'params': [p for p in seld_params[unfreeze_from:]], 'lr': self.train_config.seld_lr},
                {'params': self.model.adapter.parameters(), 'lr': self.train_config.adapter_lr},
                {'params': self.model.imagination_head.parameters(), 'lr': self.train_config.adapter_lr},
            ]

            print(f"  SELD encoder (top 50%): TRAINABLE (lr={self.train_config.seld_lr})")
            print(f"  Adapter + Head: TRAINABLE (lr={self.train_config.adapter_lr})")

        self._print_model_info()

        optimizer = optim.AdamW(
            trainable_params,
            weight_decay=self.train_config.weight_decay
        )

        return optimizer

    def train_epoch(self, optimizer: optim.Optimizer, epoch: int):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        epoch_metrics = {
            'cls_loss': 0.0,
            'loc_loss': 0.0,
            'conf_loss': 0.0,
        }

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.train_config.num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            visual = batch['visual'].to(self.device)  # [B, 5, H, W]
            audio = batch['audio'].to(self.device)  # [B, 7, T, F]
            class_label = batch['class_label'].to(self.device)  # [B]
            location = batch['location'].to(self.device)  # [B, 3]
            occlusion_level = batch['occlusion_level'].to(self.device)  # [B, 1]

            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(audio, visual, occlusion_level)

            # Compute loss
            targets = {
                'class_label': class_label,
                'location': location,
            }
            loss, loss_components = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            # Accumulate metrics
            epoch_loss += loss.item()
            for key, value in loss_components.items():
                epoch_metrics[key] += value

            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'cls': loss_components['cls_loss'],
                'loc': loss_components['loc_loss'],
            })

            # Log to TensorBoard
            global_step = epoch * len(self.train_loader) + batch_idx
            if batch_idx % 10 == 0:
                self.writer.add_scalar('Train/loss', loss.item(), global_step)
                for key, value in loss_components.items():
                    self.writer.add_scalar(f'Train/{key}', value, global_step)

        # Average metrics
        num_batches = len(self.train_loader)
        epoch_loss /= num_batches
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches

        return epoch_loss, epoch_metrics

    def validate(self, epoch: int):
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        val_metrics = {
            'cls_loss': 0.0,
            'loc_loss': 0.0,
            'conf_loss': 0.0,
            'accuracy': 0.0,
        }

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move data to device
                visual = batch['visual'].to(self.device)
                audio = batch['audio'].to(self.device)
                class_label = batch['class_label'].to(self.device)
                location = batch['location'].to(self.device)
                occlusion_level = batch['occlusion_level'].to(self.device)

                # Forward pass
                outputs = self.model(audio, visual, occlusion_level)

                # Compute loss
                targets = {
                    'class_label': class_label,
                    'location': location,
                }
                loss, loss_components = self.criterion(outputs, targets)

                # Compute accuracy
                predicted_classes = torch.argmax(outputs['class_logits'], dim=1)
                accuracy = (predicted_classes == class_label).float().mean().item()

                # Accumulate metrics
                val_loss += loss.item()
                for key, value in loss_components.items():
                    val_metrics[key] += value
                val_metrics['accuracy'] += accuracy

        # Average metrics
        num_batches = len(self.val_loader)
        val_loss /= num_batches
        for key in val_metrics:
            val_metrics[key] /= num_batches

        # Log to TensorBoard
        self.writer.add_scalar('Val/loss', val_loss, epoch)
        for key, value in val_metrics.items():
            self.writer.add_scalar(f'Val/{key}', value, epoch)

        return val_loss, val_metrics

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'stage': self.current_stage,
            'model_state_dict': self.model.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'model_config': self.model_config.__dict__,
            'train_config': self.train_config.__dict__,
        }

        # Save regular checkpoint
        checkpoint_path = self.save_dir / f"checkpoint_epoch{epoch}_stage{self.current_stage}.pth"
        torch.save(checkpoint, checkpoint_path)
        print(f"  Saved checkpoint: {checkpoint_path}")

        # Save best model
        if is_best:
            best_path = self.save_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"  ✓ New best model saved: {best_path}")

    def train(self):
        """Main training loop with two-stage fine-tuning"""
        print(f"\n{'='*60}")
        print(f"Starting Two-Stage Training")
        print(f"{'='*60}")

        # Stage 1: Freeze SELD, train adapter + head
        print(f"\n{'='*60}")
        print(f"STAGE 1: Training with frozen SELD encoder")
        print(f"{'='*60}")

        self.current_stage = 1
        optimizer = self._get_optimizer(stage=1)

        for epoch in range(1, self.train_config.freeze_seld_epochs + 1):
            self.current_epoch = epoch

            # Train
            train_loss, train_metrics = self.train_epoch(optimizer, epoch)

            # Validate
            val_loss, val_metrics = self.validate(epoch)

            # Print epoch summary
            print(f"\nEpoch {epoch}/{self.train_config.freeze_seld_epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")

            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            if epoch % self.train_config.save_every == 0 or is_best:
                self.save_checkpoint(epoch, val_loss, is_best)

        # Stage 2: Unfreeze top 50% of SELD
        print(f"\n{'='*60}")
        print(f"STAGE 2: Fine-tuning with unfrozen SELD (top 50%)")
        print(f"{'='*60}")

        self.current_stage = 2
        optimizer = self._get_optimizer(stage=2)

        stage2_epochs = self.train_config.num_epochs - self.train_config.freeze_seld_epochs

        for epoch in range(1, stage2_epochs + 1):
            total_epoch = self.train_config.freeze_seld_epochs + epoch
            self.current_epoch = total_epoch

            # Train
            train_loss, train_metrics = self.train_epoch(optimizer, total_epoch)

            # Validate
            val_loss, val_metrics = self.validate(total_epoch)

            # Print epoch summary
            print(f"\nEpoch {total_epoch}/{self.train_config.num_epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")

            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            if epoch % self.train_config.save_every == 0 or is_best:
                self.save_checkpoint(total_epoch, val_loss, is_best)

        print(f"\n{'='*60}")
        print(f"✓ Training Complete!")
        print(f"  Best validation loss: {self.best_val_loss:.4f}")
        print(f"  Checkpoints saved to: {self.save_dir}")
        print(f"{'='*60}\n")

        self.writer.close()


if __name__ == "__main__":
    # Configuration
    model_config = ModelConfig(
        num_classes=10,
        audio_dim=512,
        visual_dim=2048,
        embed_dim=512,
        pretrained_seld_path=None,  # Add path to pretrained SELD weights if available
    )

    train_config = TrainingConfig(
        batch_size=16,
        num_epochs=100,
        freeze_seld_epochs=30,  # Stage 1 duration
        seld_lr=1e-5,
        adapter_lr=1e-4,
        weight_decay=1e-4,
        num_workers=4,
        save_every=5,
    )

    # Initialize trainer
    trainer = TwoStageTrainer(
        model_config=model_config,
        train_config=train_config,
        data_dir="/home/r13921098/RPL/final/sound-spaces/data/generated_dataset",
        save_dir="checkpoints",
    )

    # Start training
    trainer.train()
