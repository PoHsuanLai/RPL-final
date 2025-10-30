# SoundSpaces 2.0 + Habitat-Sim Installation Guide

## Prerequisites

- CUDA-capable GPU (already confirmed: CUDA 12.8 available)
- Python 3.11 (already set up with uv)
- ~10 GB disk space for installation

---

## Installation Steps

### Option 1: Install via pip (Recommended)

```bash
cd ~/RPL/final/audio-visual-imagination

# Install Habitat-Sim with audio support
uv add habitat-sim --index-url https://aihabitat.org/whl/nightly/

# Install Habitat-Lab
uv add habitat-lab

# Install SoundSpaces
uv add soundspaces
```

### Option 2: Install from conda (if pip fails)

SoundSpaces often requires conda for complex dependencies. If Option 1 fails:

```bash
# You'll need to use conda instead of uv for these packages
conda install habitat-sim withbullet headless -c conda-forge -c aihabitat
conda install -c aihabitat habitat-lab
pip install soundspaces
```

### Option 3: Build from source (most control)

```bash
# Clone repositories
git clone https://github.com/facebookresearch/habitat-sim.git
git clone https://github.com/facebookresearch/habitat-lab.git
git clone https://github.com/facebookresearch/sound-spaces.git

# Build habitat-sim with audio
cd habitat-sim
python setup.py install --with-cuda --headless --audio

# Install habitat-lab
cd ../habitat-lab
pip install -e .

# Install sound-spaces
cd ../sound-spaces
pip install -e .
```

---

## Verification

After installation, verify:

```bash
uv run python -c "import habitat; print('Habitat:', habitat.__version__)"
uv run python -c "import habitat_sim; print('Habitat-Sim:', habitat_sim.__version__)"
uv run python -c "import soundspaces; print('SoundSpaces: OK')"
```

---

## Download Scene Data

SoundSpaces requires 3D scene meshes:

```bash
# Create data directory
mkdir -p data/scene_datasets

# Download Matterport3D scenes (requires registration)
# Visit: https://niessner.github.io/Matterport/

# Or use Replica dataset (freely available)
wget http://dl.fbaipublicfiles.com/habitat/habitat-test-scenes.zip
unzip habitat-test-scenes.zip -d data/scene_datasets/
```

---

## Common Issues

### Issue 1: CUDA version mismatch
**Solution**: Install habitat-sim with matching CUDA version (12.8 in our case)

### Issue 2: Audio backend not found
**Solution**: Ensure you install habitat-sim with `--audio` flag

### Issue 3: Missing dependencies
**Solution**: Install build essentials:
```bash
sudo apt-get update
sudo apt-get install build-essential cmake libglew-dev
```

---

## Quick Test

```python
import habitat
import habitat_sim

# Test basic functionality
cfg = habitat_sim.Configuration()
sim = habitat_sim.Simulator(cfg)
print("✓ Habitat-Sim works!")
```

