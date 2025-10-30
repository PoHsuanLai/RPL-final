# Audio-Visual Imagination for Occluded Hazard Avoidance

---

## SLIDE 1: Title

**Audio-Visual Cross-Modal Imagination for Robot Hazard Avoidance Under Occlusion**

Team: [Your Name]  
Course: [Course Name]  
Date: [Date]

---

## SLIDE 2: One-Sentence Take-Away

**Robots can "imagine" hidden hazards by reasoning about sounds when vision is blocked, enabling safer navigation around corners and obstacles.**

---

## SLIDE 3: Motivation (1) - Visual Examples

**The Problem: Real Environments Have Occlusions**

- **Warehouse**: Robot can't see forklift around corner, but hears beeping
- **Intersection**: Emergency vehicle behind building - no visual, but siren audible
- **Hospital**: Crash sound behind partition indicates hazard

**Key Insight**: Audio is omnidirectional (360°), vision is limited by line-of-sight.

---

## SLIDE 4: Motivation (2) - Why Important?

**Applications:**
- Autonomous vehicles detecting hidden hazards at blind intersections
- Warehouse robots avoiding collisions with occluded equipment
- Search & rescue locating victims behind debris
- Service robots navigating crowded spaces with obstructions

**The Task:**
- **Input**: Partial visual + omnidirectional audio
- **Output**: Hidden event type, 3D location, danger level
- **Goal**: Plan safe avoidance

**Impact**: Safety-critical for real deployment where occlusions are constant

---

## SLIDE 5: Prior Works (1) - Audio-Visual SELD

**Category: Sound Event Localization & Detection**

Works focus on detecting sounds when sources are *visible*.

| Paper | Method | Limitation |
|-------|--------|------------|
| **STARSS23 Baseline** (DCASE 2023)<br/>Shimada et al. | 360° recordings + Multi-ACCDOA<br/>YOLOX detector + GRU fusion | ❌ Assumes visible sources<br/>❌ Object detector fails when occluded<br/>❌ No downstream avoidance |
| **Fusion of AV Embeddings** (ICASSP 2024)<br/>Berghi et al. | ResNet50 + audio encoder<br/>Cross-Modal Fusion (CMAF) | ❌ Requires visibility for visual branch<br/>❌ No occlusion reasoning |
| **Enhanced AV-SELD** (Jan 2024)<br/>Roman et al. | YOLO5/8 + data augmentation<br/>SELDnet23 architecture | ❌ Detector fails under occlusion<br/>❌ Not robot-centric |
| **DOA-Aware Self-Supervised** (Oct 2024)<br/>Fujita et al. | Contrastive learning on FOA mics<br/>DOA-wise embeddings | ❌ No imagination component<br/>❌ Assumes visible training sources |
| **Distance Estimation** (Oct 2024)<br/>Berghi & Jackson | Reverberation features + depth maps<br/>AV-Conformer | ✓ Considers depth<br/>❌ Still requires visibility |

**Key Gap**: All assume visible sources - fail completely under occlusion.

---

## SLIDE 6: Prior Works (2) - Other Categories

**Audio-Visual Navigation**

| Paper | Method | Limitation |
|-------|--------|------------|
| **SoundSpaces 1.0** (ECCV 2020)<br/>Chen et al. | RL agents navigate TO sounds<br/>Matterport3D + Replica | ❌ Approaches sound (wrong goal)<br/>❌ No occlusion scenarios |
| **SoundSpaces 2.0** (NeurIPS 2022)<br/>Chen et al. | Fast geometric acoustic rendering<br/>Continuous spatial sampling | ✓ Can simulate occlusions<br/>❌ No imagination model |

**Audio in Manipulation**

| Paper | Method | Limitation |
|-------|--------|------------|
| **Audio-Visual Occlusion** (CoRL 2022)<br/>Du et al. | Contact sounds for occluded gripper<br/>Near-field manipulation | ✓ Uses audio under occlusion<br/>❌ Contact only (cm range)<br/>❌ Not for distant hazards |

**Other Relevant Work**

- **Acoustic Vehicle Detection** (Schulz et al.): Around-corner detection, but no vision integration or policies
- **Multimodal Perception Survey** (ACM THRI 2024): Identifies occlusion as open problem, no solution

**Critical Gap**: No work combines (1) audio reasoning, (2) occluded event inference, (3) robot avoidance.

---

## SLIDE 7: High-Level Idea

**Our Novel Approach: Audio as Causal Evidence for Latent Scene Inference**

**Key Difference from Prior Work:**

| Prior Work | Our Approach |
|------------|--------------|
| Audio + Visual → Detect *visible* events | Audio + Partial Visual → **Infer *hidden* events** |
| Navigate *toward* sound | Navigate *away from* inferred hazard |
| Both modalities = direct evidence | Audio = indirect clue, Visual = context |
| Passive perception | **Active imagination** (predict → act) |

**The "Imagination" Mechanism:**

Human analogy: Hearing a crash behind a building → infer accident without seeing it

Robot version:
```
Partial Scene + Audio → Cross-Modal Fusion → Imagination Module
    → Predict: event class, 3D location, confidence
    → Action: Avoidance policy
```

**Novel Contributions:**
1. First to use audio for *hidden hazard imagination* in robotics
2. Adaptive fusion that weighs modalities by reliability under occlusion
3. End-to-end: perception → imagination → avoidance

---

## SLIDE 8: Mid-Level Idea

**System Architecture:**

```
Audio (4-ch FOA)          Visual (RGB-D)        Occlusion Mask
       ↓                        ↓                      ↓
  Audio Encoder            Visual Encoder         Context Info
  (CNN + GRU)              (ResNet-50)
       ↓                        ↓                      ↓
       └────────────────────────┴──────────────────────┘
                                ↓
                   Cross-Modal Attention Fusion
                   (Reliability-aware weighting)
                                ↓
                      Imagination Head
                   • Event classifier (10 classes)
                   • 3D localizer (azimuth, elev, dist)
                   • Uncertainty estimator
                                ↓
                       Avoidance Policy
                         (RL or rules)
```

**Key Components:**

1. **Audio Encoder**: CNN (spectrogram) + Bi-GRU (temporal)
2. **Visual Encoder**: ResNet-50 + depth channel + occlusion mask
3. **Fusion**: Cross-attention with adaptive weighting based on occlusion level
4. **Imagination**: Multi-task head predicting event properties
5. **Policy**: Navigate away from predicted hazards

**Datasets:**
- **STARSS23**: 42hrs real 360° audio-visual recordings
- **SoundSpaces 2.0**: 100+ hrs synthetic with programmed occlusions

**Environment:** SoundSpaces 2.0 (fast rendering, realistic acoustics, programmable occlusions)

**Backbone:** ResNet-50 (visual), CNN+GRU (audio), Cross-Attention (fusion)

---

## SLIDE 9: Experimental Setup

**Experiment 1: Imagination Capability**
- **Setup**: STARSS23 + synthetic data, occlusion levels 0-100%
- **Metrics**: Classification accuracy, localization error (MAE), F1 score
- **Baselines**: Audio-only, Visual-only, Simple concat, STARSS23 baseline, Oracle
- **Expected**: Outperform baselines at 50%+ occlusion; audio-only poor localization

**Experiment 2: Ablation Studies**
- Remove: attention, reliability weighting, uncertainty, occlusion mask
- **Expected**: Attention +5% F1, reliability weighting +8% F1, uncertainty improves calibration

**Experiment 3: Hazard Avoidance (Downstream)**
- **Setup**: Navigate to goal while avoiding hidden hazards in SoundSpaces
- **Metrics**: Avoidance success rate, task completion, reaction time, false positives
- **Baselines**: Vision-only, AV without imagination, random, ours
- **Expected**: 90% success vs 60% baseline, 1-2s faster reaction

**Experiment 4: Generalization**
- Test on unseen environments, novel event types
- **Expected**: F1 drop <10% on new scenes

**Experiment 5: Qualitative Analysis**
- Visualize attention maps, prediction trajectories, failure cases

**Why These Results:**
- Audio provides directional cues without vision
- Attention enables adaptive reliance on audio when visual fails
- Training on occlusion forces cross-modal inference learning

---

## SLIDE 10: Development Roadmap

**10-Week Timeline:**

**Weeks 1-2: Setup & Data**
- Install SoundSpaces 2.0 + Habitat
- Download STARSS23 dataset
- Create occlusion scenarios (1000 samples)

**Weeks 3-4: Model Development**
- Implement audio/visual encoders (use pretrained)
- Build cross-attention fusion module
- Implement imagination head

**Weeks 5-6: Training**
- Phase 1: Train on full supervision
- Phase 2: Train with occlusion
- Hyperparameter tuning

**Weeks 7-8: Evaluation**
- Run all baselines
- Ablation studies
- Collect metrics

**Weeks 9: Downstream Task**
- Implement avoidance policy (RL or rule-based)
- Test navigation + avoidance

**Week 10: Analysis & Report**
- Qualitative analysis
- Write report, prepare presentation

**Key Milestones:**
- Week 2: Data ready
- Week 4: Model architecture complete
- Week 6: Initial results
- Week 8: All experiments done
- Week 10: Final deliverables

**Deliverables:**
- Trained imagination model
- Benchmark results on occlusion scenarios
- Avoidance policy demo
- Analysis of when/why model succeeds/fails

---

**References:**

[1] Shimada et al., "STARSS23 Dataset," DCASE 2023  
[2] Berghi et al., "Fusion of Audio-Visual Embeddings," ICASSP 2024  
[3] Roman et al., "Enhanced AV-SELD," Jan 2024  
[4] Fujita et al., "DOA-Aware Self-Supervised," Oct 2024  
[5] Chen et al., "SoundSpaces 2.0," NeurIPS 2022  
[6] Du et al., "Audio-Visual Occlusion Learning," CoRL 2022
