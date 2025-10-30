# Quick Start Guide - Audio-Visual Imagination

Get started with the Audio-Visual Imagination system in 3 simple steps!

---

## Prerequisites

- ✅ Replica dataset downloaded (running in background)
- ✅ SoundSpaces 2.0 environment installed
- ✅ Model implementation complete

---

## Step 1: Generate Training Data (5-10 minutes)

Generate synthetic training data with occlusion scenarios:

```bash
cd ~/RPL/final/sound-spaces
source ~/miniforge3/bin/activate soundspaces

# Quick test (5 samples)
python test_full_pipeline.py

# Generate full training set (1000 samples, ~30 minutes)
python soundspaces_data_generator.py
```

**What gets generated:**
- RGB images with occlusions
- Depth maps
- Occlusion masks
- Spatial audio (FOA format)
- SELD-compatible audio features
- Ground truth labels

---

## Step 2: Train the Model (Several hours)

Start two-stage training:

```bash
cd ~/RPL/final/audio-visual-imagination

# Activate uv environment (or use conda)
source .venv/bin/activate

# Start training
python train.py

# Monitor with TensorBoard (in another terminal)
tensorboard --logdir runs/
```

**Training stages:**
- **Stage 1** (30 epochs): Freeze SELD encoder, train adapter + head
- **Stage 2** (70 epochs): Fine-tune top 50% of SELD + adapter + head

**Expected time:**
- ~5-10 minutes per epoch with GPU
- Total: 10-15 hours on single GPU

---

## Step 3: Evaluate Results

Check training progress:

```bash
# View TensorBoard
tensorboard --logdir runs/
# Open: http://localhost:6006

# Check saved checkpoints
ls -lh checkpoints/

# Load best model
ls -lh checkpoints/best_model.pth
```

---

## Quick Test Without Full Dataset

If you want to test the system immediately without waiting for data generation:

```bash
cd ~/RPL/final/sound-spaces

# Test with just 5 samples (< 1 minute)
python test_full_pipeline.py
```

This will:
1. ✅ Generate 5 test samples
2. ✅ Test dataset loading
3. ✅ Run model inference

---

## Directory Structure After Setup

```
RPL/final/
├── audio-visual-imagination/
│   ├── src/                          # Model source code
│   ├── examples/                     # Test scripts
│   ├── train.py                      # Training script
│   ├── checkpoints/                  # Model checkpoints (created during training)
│   │   ├── best_model.pth
│   │   └── checkpoint_epoch*.pth
│   └── runs/                         # TensorBoard logs
│
└── sound-spaces/
    ├── data/
    │   ├── scene_datasets/
    │   │   └── ReplicaDataset/      # 18 indoor scenes (~9GB)
    │   ├── generated_dataset/        # Generated training data
    │   │   ├── sample_000000/
    │   │   ├── sample_000001/
    │   │   └── ...
    │   └── test_dataset/             # Small test dataset (from test_full_pipeline.py)
    ├── soundspaces_data_generator.py
    ├── seld_dataset_loader.py
    └── test_full_pipeline.py
```

---

## Configuration

Edit `train.py` to customize:

```python
# Model configuration
model_config = ModelConfig(
    num_classes=10,              # Number of event classes
    audio_dim=512,               # SELD encoder output dim
    visual_dim=2048,             # ResNet-50 output dim
    embed_dim=512,               # Fusion dimension
    pretrained_seld_path=None,   # Path to pretrained SELD (optional)
)

# Training configuration
train_config = TrainingConfig(
    batch_size=16,               # Adjust based on GPU memory
    num_epochs=100,              # Total epochs
    freeze_seld_epochs=30,       # Stage 1 duration
    seld_lr=1e-5,               # SELD fine-tuning LR
    adapter_lr=1e-4,            # Adapter/head LR
    weight_decay=1e-4,
)
```

---

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution**: Reduce batch size in `train.py`:

```python
train_config = TrainingConfig(
    batch_size=8,  # Reduce from 16 to 8
    ...
)
```

### Issue: Data generation too slow

**Solution**: Generate smaller dataset first:

```python
# In soundspaces_data_generator.py, change:
generator.generate_dataset(num_samples=100)  # Instead of 1000
```

### Issue: Scene not found

**Solution**: Wait for Replica download to complete, or specify different scene:

```python
generator.generate_dataset(
    num_samples=100,
    scene_names=["apartment_0"]  # Use available scene
)
```

---

## Next Steps After Training

1. **Evaluate on test set**:
   ```python
   # Add to train.py or create separate eval script
   test_loss, test_metrics = trainer.validate_on_test()
   ```

2. **Visualize predictions**:
   - Create visualization script
   - Show RGB + predicted sound location
   - Compare with ground truth

3. **Export model**:
   ```python
   torch.save(model.state_dict(), 'model_final.pth')
   ```

4. **Deploy on robot**:
   - Convert to ONNX for inference
   - Integrate with robot control system
   - Real-time audio-visual processing

---

## Performance Expectations

**After training on 1000 samples:**
- Classification accuracy: 70-80%
- Localization error: <20 degrees (azimuth/elevation)
- Distance error: <1 meter

**For better performance:**
- Generate 10K+ samples
- Use multiple scenes from Matterport3D
- Use pretrained SELD weights from DCASE2023
- Train for more epochs (200+)

---

## GPU Recommendations

- **Minimum**: 8GB VRAM (batch size 4-8)
- **Recommended**: 16GB+ VRAM (batch size 16-32)
- **Optimal**: 24GB+ VRAM (batch size 32-64)

Without GPU: Training will be 10-20x slower but still possible.

---

## Getting Help

1. **Check logs**: Look at TensorBoard for training curves
2. **Test components**: Run `test_full_pipeline.py`
3. **Verify model**: Run `examples/test_seld_integration.py`
4. **Read docs**: See `IMPLEMENTATION_SUMMARY.md`

---

## Summary Commands

```bash
# 1. Generate data
cd ~/RPL/final/sound-spaces
source ~/miniforge3/bin/activate soundspaces
python soundspaces_data_generator.py

# 2. Train model
cd ~/RPL/final/audio-visual-imagination
source .venv/bin/activate
python train.py

# 3. Monitor training
tensorboard --logdir runs/
```

That's it! 🚀

---

**Estimated Total Time:**
- Data generation: 30-60 minutes (1000 samples)
- Training: 10-15 hours (100 epochs, single GPU)
- **Total**: ~11-16 hours to fully trained model
