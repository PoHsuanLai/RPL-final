# Transfer Learning Architecture: Pretrained SELD → Imagination Task

## Overview
Instead of training from scratch, we adapt a pretrained SELD model for occluded hazard imagination.

---

## Architecture Design

### Original SELD Model Output
```
Audio (4-ch FOA) → SELD Encoder → SELD Head → {
    Event Detection: [T, C] binary (13 classes)
    DOA Estimation: [T, C, 3] (azimuth, elevation, distance)
}
where T=time frames, C=classes
```

### Adapted Architecture for Imagination Task
```
┌─────────────────────────────────────────────────┐
│  INPUT STAGE                                    │
├─────────────────────────────────────────────────┤
│ Audio (4-ch FOA)          RGB-D + Occlusion     │
│      ↓                            ↓             │
│ [PRETRAINED SELD ENCODER - FROZEN]              │
│   • CNN blocks (extract spectro-temporal)       │
│   • Bi-GRU (temporal modeling)                  │
│   • Output: [T, 512] audio features             │
│                                                  │
│ [NEW: VISUAL ENCODER]                           │
│   • ResNet-50 (pretrained, frozen initially)    │
│   • Output: [2048] visual features              │
└─────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│  ADAPTATION LAYER (NEW - TRAINABLE)             │
├─────────────────────────────────────────────────┤
│ Audio Features [T, 512] → Temporal Pooling      │
│                        → [512] global audio     │
│                                                  │
│ Visual Features [2048] → Projection              │
│                        → [512] visual            │
│                                                  │
│ Cross-Modal Fusion (NEW)                        │
│   • Multi-head attention                        │
│   • Reliability weighting based on occlusion    │
│   • Output: [1024] fused embedding              │
└─────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│  IMAGINATION HEAD (NEW - TRAINABLE)             │
├─────────────────────────────────────────────────┤
│ Multi-Task Prediction:                          │
│                                                  │
│ 1. Event Classification Head                    │
│    FC(1024 → 512 → 10) + Softmax                │
│    Output: Hazard class probabilities           │
│                                                  │
│ 2. Localization Head                            │
│    FC(1024 → 512 → 3)                           │
│    Output: [azimuth, elevation, distance]       │
│                                                  │
│ 3. Confidence Head                              │
│    FC(1024 → 256 → 1) + Sigmoid                 │
│    Output: Prediction confidence [0,1]          │
└─────────────────────────────────────────────────┘
                       ↓
              Imagination Output
```

---

## Transfer Learning Strategy

### Phase 1: Feature Extraction (Weeks 1-2)
**Freeze:** Entire SELD encoder
**Train:**
- Visual encoder projection layer
- Cross-modal fusion module
- Imagination heads (all 3 tasks)

**Why:** SELD encoder already learned good audio representations for spatial sound. We just need to:
1. Add visual understanding
2. Learn cross-modal reasoning under occlusion
3. Map to our specific hazard classes

### Phase 2: Fine-Tuning (Week 3)
**Freeze:** Bottom 50% of SELD encoder
**Train:**
- Top 50% of SELD encoder
- All adaptation layers
- All imagination heads

**Why:** Allow model to adjust audio features for occluded scenarios

---

## Model Selection: STARSS23 Baseline

### Why STARSS23?
1. **Already audio-visual:** Has visual branch (YOLOX detector)
2. **Same data:** Trained on STARSS23 dataset (you'll use this too)
3. **360° audio:** Uses FOA format (4-channel)
4. **Multi-ACCDOA:** Advanced output format for multiple events
5. **Available checkpoint:** Official pretrained weights

### What We Keep vs. Replace

| Component | Action | Reason |
|-----------|--------|--------|
| Audio CNN blocks | **KEEP (freeze)** | Already extracts spectro-temporal features |
| Bi-GRU layers | **KEEP (freeze)** | Temporal modeling is universal |
| YOLOX detector | **REPLACE** | Fails under occlusion, use ResNet-50 instead |
| ACCDOA output head | **REPLACE** | Task-specific, need imagination head |
| Fusion mechanism | **ENHANCE** | Add occlusion-aware weighting |

---

## Implementation Details

### 1. SELD Encoder Output
```python
# From pretrained SELD
audio_features = seld_encoder(audio_input)  # [batch, T, 512]

# Temporal pooling for imagination task
# (SELD predicts per-frame, we predict per-scene)
audio_pooled = torch.mean(audio_features, dim=1)  # [batch, 512]
# OR use attention pooling
audio_pooled = attention_pool(audio_features)  # [batch, 512]
```

### 2. Adapter Module
```python
class SELDToImaginationAdapter(nn.Module):
    def __init__(self):
        self.audio_projection = nn.Linear(512, 512)
        self.visual_projection = nn.Linear(2048, 512)

        # Cross-modal fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=8
        )

        # Reliability weighting (learns from occlusion level)
        self.reliability_net = nn.Sequential(
            nn.Linear(1, 64),  # Input: occlusion %
            nn.ReLU(),
            nn.Linear(64, 2),  # Output: [w_audio, w_visual]
            nn.Softmax(dim=-1)
        )

    def forward(self, audio_feat, visual_feat, occlusion_level):
        # Project to same dimension
        audio_proj = self.audio_projection(audio_feat)
        visual_proj = self.visual_projection(visual_feat)

        # Cross-attention
        attn_out, _ = self.cross_attention(
            query=audio_proj.unsqueeze(0),
            key=visual_proj.unsqueeze(0),
            value=visual_proj.unsqueeze(0)
        )

        # Adaptive weighting
        weights = self.reliability_net(occlusion_level)
        w_audio, w_visual = weights[:, 0], weights[:, 1]

        # Fused features
        fused = torch.cat([
            w_audio * audio_proj,
            w_visual * attn_out.squeeze(0)
        ], dim=-1)  # [batch, 1024]

        return fused
```

### 3. Imagination Head
```python
class ImaginationHead(nn.Module):
    def __init__(self, input_dim=1024, num_classes=10):
        # Shared backbone
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Task-specific heads
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

        self.localizer = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # [azimuth, elevation, distance]
        )

        self.confidence = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        shared_feat = self.shared(x)

        return {
            'class_logits': self.classifier(shared_feat),
            'location': self.localizer(shared_feat),
            'confidence': self.confidence(shared_feat)
        }
```

### 4. Full Model
```python
class AudioVisualImagination(nn.Module):
    def __init__(self, pretrained_seld_path):
        # Load pretrained SELD
        self.seld_encoder = load_seld_encoder(pretrained_seld_path)
        self.freeze_seld()  # Freeze initially

        # New components
        self.visual_encoder = resnet50(pretrained=True)
        self.visual_encoder.fc = nn.Identity()  # Remove classifier

        self.adapter = SELDToImaginationAdapter()
        self.imagination_head = ImaginationHead()

    def freeze_seld(self):
        for param in self.seld_encoder.parameters():
            param.requires_grad = False

    def unfreeze_seld_top(self, ratio=0.5):
        # Unfreeze top 50% of layers
        params = list(self.seld_encoder.parameters())
        split_idx = int(len(params) * ratio)
        for param in params[split_idx:]:
            param.requires_grad = True

    def forward(self, audio, visual, occlusion_level):
        # Extract features
        audio_feat = self.seld_encoder(audio)  # [B, T, 512]
        audio_feat = audio_feat.mean(dim=1)    # Pool temporally

        visual_feat = self.visual_encoder(visual)  # [B, 2048]

        # Adapt and fuse
        fused = self.adapter(audio_feat, visual_feat, occlusion_level)

        # Predict
        outputs = self.imagination_head(fused)

        return outputs
```

---

## Training Procedure

### Stage 1: Adapter Training (Epochs 1-30)
```python
# Freeze SELD completely
model.freeze_seld()

# Only train new components
optimizer = Adam([
    {'params': model.adapter.parameters(), 'lr': 1e-3},
    {'params': model.imagination_head.parameters(), 'lr': 1e-3},
    {'params': model.visual_encoder.fc.parameters(), 'lr': 1e-4}
])

# Train on occlusion data
for epoch in range(30):
    train_epoch(model, occluded_data, optimizer)
```

### Stage 2: Fine-Tuning (Epochs 31-50)
```python
# Unfreeze top SELD layers
model.unfreeze_seld_top(ratio=0.5)

# Lower learning rate
optimizer = Adam([
    {'params': model.seld_encoder.parameters(), 'lr': 1e-5},
    {'params': model.adapter.parameters(), 'lr': 1e-4},
    {'params': model.imagination_head.parameters(), 'lr': 1e-4}
])

# Continue training
for epoch in range(31, 51):
    train_epoch(model, occluded_data, optimizer)
```

---

## Loss Function

```python
def compute_loss(outputs, targets, alpha=1.0, beta=2.0, gamma=0.5):
    # Classification loss
    loss_cls = F.cross_entropy(
        outputs['class_logits'],
        targets['class']
    )

    # Localization loss (only for present events)
    mask = targets['class'] != 0  # Background class
    loss_loc = F.mse_loss(
        outputs['location'][mask],
        targets['location'][mask]
    )

    # Confidence calibration loss
    pred_conf = outputs['confidence']
    true_conf = (outputs['class_logits'].softmax(1).max(1)[0]).detach()
    loss_conf = F.mse_loss(pred_conf.squeeze(), true_conf)

    total_loss = alpha * loss_cls + beta * loss_loc + gamma * loss_conf
    return total_loss
```

---

## Advantages of This Approach

1. **Faster Training:** SELD already learned audio-spatial features
2. **Better Generalization:** Pretrained on diverse sounds
3. **Less Data Required:** Only need to learn occlusion reasoning
4. **Proven Architecture:** SELD models are state-of-the-art
5. **Modular:** Can swap SELD backbone easily

---

## Data Requirements (Reduced)

With transfer learning:
- **Training:** 500-1000 occlusion scenarios (vs 5000+ from scratch)
- **Validation:** 200 scenarios
- **Testing:** 300 scenarios

SELD's prior knowledge compensates for smaller dataset.

---

## Next Steps

1. Download STARSS23 pretrained checkpoint
2. Extract SELD encoder weights
3. Implement adapter + imagination head
4. Generate small occlusion dataset
5. Train Stage 1 (freeze SELD)
6. Evaluate and iterate
