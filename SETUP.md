# Audio-Visual Imagination Setup

## Quick Setup (Local Machine with Display)

### 1. Clone Repository
```bash
git clone --recurse-submodules <your-repo-url>
cd <repo-name>
```

### 2. Install Habitat-Sim (with Audio Support)
```bash
cd habitat-sim
git checkout RLRAudioPropagationUpdate
python setup.py install --headless --audio
cd ..
```

### 3. Install Dependencies
```bash
# Habitat-Lab
cd habitat-lab
pip install -e .
cd ..

# SoundSpaces
cd sound-spaces
pip install -e .
cd ..

# Audio-Visual Imagination Model
cd audio-visual-imagination
uv sync
cd ..
```

### 4. Download Replica Dataset
```bash
cd sound-spaces/data/scene_datasets
git clone https://github.com/facebookresearch/Replica-Dataset.git
cd Replica-Dataset
bash download.sh ../ReplicaDataset
cd ../../..
```

### 5. Generate Training Data
```bash
cd sound-spaces
export PYTHONPATH=../audio-visual-imagination/src:$PYTHONPATH
python soundspaces_data_generator.py  # Generates 1000 samples
cd ..
```

### 6. Train Model
```bash
cd audio-visual-imagination
uv run python train.py
```

## For SSH/Headless Servers

Use `xvfb-run` and set environment variables:
```bash
export MAGNUM_DRIVER=egl
export DISPLAY=:99
Xvfb :99 -screen 0 1024x768x24 &
python soundspaces_data_generator.py
```

## Project Structure
- `audio-visual-imagination/` - Main model implementation (28.7M params)
- `habitat-lab/` - Habitat simulator (v0.2.2)
- `sound-spaces/` - Audio simulation + data generation
- `habitat-sim/` - 3D simulator with audio (build from source)
