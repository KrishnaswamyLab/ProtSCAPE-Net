# ProtSCAPE-Net

**ProtSCAPE-Net** - Learning Protein Conformational Landscapes from Molecular Dynamics for Ensemble and Transition Path Generation

![ProtSCAPE-Net Architecture](data/schematic(3).png)

---

## Overview

ProtSCAPE-Net combines multiple state-of-the-art techniques to learn and generate protein conformational landscapes:

- **SE(3)-Equivariant Graph Networks**: Respects the symmetries of 3D protein structures
- **Scattering Transforms**: Multi-scale geometric feature extraction
- **Transformer Encoders**: Captures long-range dependencies between atoms/residues
- **Latent Diffusion Models**: Generates novel conformational ensembles
- **Energy-Guided Path Generation**: NEB for transition pathway discovery

### Key Features

✨ **Structure Reconstruction**: Atomic-level protein structure prediction from graph representations  
🧬 **Conformational Ensemble Generation**: Sample diverse protein conformations via latent diffusion  
🛤️ **Transition Path Discovery**: Generate minimum energy paths between conformational states  
📊 **MolProbity Integration**: Automated structure quality assessment  
⚡ **Efficient Training**: PyTorch Lightning with mixed precision and distributed training support

---

## Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Usage](#-usage)
  - [Training](#training)
  - [Inference](#inference)
  - [Ensemble Generation](#ensemble-generation)
  - [Path Generation](#path-generation)
- [Configuration](#-configuration)
- [Datasets](#-datasets)
- [Path Generation Methods](#-path-generation-methods)

---

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- [uv](https://docs.astral.sh/uv/) (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/ProtSCAPE-Net.git
cd ProtSCAPE-Net

# Install uv (Linux/macOS)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate a virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv sync

# On clusters with CUDA 12.8 drivers, replace the default CUDA 13.0 wheel.
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0
```

If `torch.cuda.is_available()` warns that the NVIDIA driver is too old, the environment usually has a newer CUDA wheel than the node driver supports. This repo has been tested on cluster nodes with CUDA 12.8 drivers, so install the matching `cu128` PyTorch build before training on GPU.

### Optional Dependencies

For advanced visualization:
```bash
uv pip install "phate>=0.2.5"
```

For MolProbity metrics:
```bash
# Requires phenix.molprobity (install separately)
# See: https://www.phenix-online.org/
```

---

## Quick Start

### 1. Train a Model

```bash
# Use a pre-configured setup
python train.py --config configs/config.yaml

# Or specify a protein
python train.py --config configs/config.yaml --protein 7lp1
```

### 2. Run Inference

```bash
# Evaluate on test data
python inference.py --config configs/config_inference.yaml --ckpt_path checkpoints/best_model.pt
```

### 3. Generate Conformational Ensembles

```bash
# Run the complete pipeline: AE training → DDPM training → generation
python ensemble_gen.py --config configs/config_ensemble.py
```

---

## 📁 Project Structure

```
ProtSCAPE-Net/
├── protscape/              # Core model implementations
│   ├── protscape.py        # Main ProtSCAPE model
│   ├── autoencoder.py      # Variational autoencoder
│   ├── transformer.py      # Transformer encoder
│   ├── bottleneck.py       # Latent space bottleneck
│   ├── generate.py         # Path generation algorithms
│   ├── neb.py              # Nudged Elastic Band
│   └── wavelets.py         # Scattering transform layer
├── utils/                  # Utility functions
│   ├── generation_helpers.py
│   ├── generation_viz.py
│   ├── geometry.py         # Kabsch alignment, RMSD
│   └── config.py           # Configuration loading
├── configs/                # Configuration files
│   ├── config.yaml         # Training config
│   ├── config_inference.yaml
│   ├── config_ensemble.py
│   └── CONFIG_GUIDE.md     # Configuration documentation
├── data/                   # Data preparation scripts
│   ├── prepare_atlas.py
│   ├── prepare_deshaw.py
│   └── download_*.py
├── docs/                   # Documentation
│   └── PATH_GENERATION_METHODS.md
├── train.py                # Training script
├── inference.py            # Inference/evaluation script
├── ensemble_gen.py         # Ensemble generation pipeline
└── requirements.txt        # Python dependencies
```

---

## Usage

### Training

Train ProtSCAPE on protein conformational data:

```bash
python train.py --config configs/config.yaml
```

**Key training parameters** (in config.yaml):
- `dataset`: Dataset name (e.g., "atlas", "deshaw")
- `protein`: Protein ID (e.g., "7lp1", "1bx7")
- `pkl_path`: Path to preprocessed graph data
- `latent_dim`: Dimensionality of latent space (default: 128)
- `n_epochs`: Number of training epochs
- `batch_size`: Batch size
- `lr`: Learning rate

Training outputs:
- Checkpoints in `checkpoints/`
- Training logs in `train_logs/`
- Weights & Biases logging (if configured)

### Inference

Evaluate a trained model:

```bash
python inference.py --config configs/config_inference.yaml --ckpt_path checkpoints/best_model.pt
```

**Outputs**:
- `pdb_frames/`: Predicted and ground truth PDB files
- `latents_zrep.npy`: Latent space representations
- `energies.npy`: Energy values
- `pca_energy.png`, `phate_energy.png`: Dimensionality reduction visualizations

**Key metrics**:
- Kabsch-aligned RMSD (Å)
- Coordinate MSE
- Classification accuracy (atomic number, residue, amino acid)

### Ensemble Generation

Generate conformational ensembles using latent diffusion:

```bash
python ensemble_gen.py --config configs/config_ensemble.py
```

**Pipeline stages**:
1. **Autoencoder Training**: Compress conformational space
2. **DDPM Training**: Learn generative model in latent space
3. **Sampling**: Generate novel conformations
4. **Evaluation**: Compute MolProbity scores and structural metrics

### Path Generation

Generate transition paths between conformational states:

```bash
# LEP method (Langevin dynamics)
python ensemble_gen.py --config configs/config_generation.yaml --method LEP

# NEB method (Nudged Elastic Band)
python ensemble_gen.py --config configs/config_generation_neb.yaml --method NEB
```

See [docs/PATH_GENERATION_METHODS.md](docs/PATH_GENERATION_METHODS.md) for detailed comparison of methods.

---

## Configuration

All parameters are managed via YAML configuration files. See [configs/CONFIG_GUIDE.md](configs/CONFIG_GUIDE.md) for detailed documentation.

**Example config.yaml**:

```yaml
# Dataset
dataset: "atlas"
protein: "7lp1"
pkl_path: "data/graphs/7lp1_graphs.pkl"

# Model architecture
latent_dim: 128
hidden_dim: 256
embedding_dim: 128
n_layers: 4
n_heads: 8

# Training
n_epochs: 1000
batch_size: 32
lr: 0.0001
weight_decay: 0.0001

# Normalization
normalize_xyz: true
normalize_energy: true

# Logging
wandb_project: "protscape"
save_dir: "checkpoints/"
```

**Command-line overrides**:
```bash
python train.py --config config.yaml --batch_size 64 --lr 0.0005
```

---

## Datasets

### Supported Datasets

1. **ATLAS**: High-quality MD simulations of folding transitions
2. **DE Shaw**: Anton ultra-long MD simulations
3. **Custom**: Your own molecular dynamics trajectories

### Data Preparation

```bash
# Download and prepare ATLAS dataset
cd data/
python download_atlas.py
python prepare_atlas.py --protein 7lp1

# Prepare DE Shaw data
python download_deshaw.py
python prepare_deshaw.py --protein ubiquitin
```

**Data format**: Preprocessed graphs stored as `.pkl` files with:
- `x`: Node features [atomic_number, residue_idx, aa_idx, xyz(3)]
- `edge_index`: Graph connectivity
- `edge_attr`: Edge features
- `energy`: Potential energy (optional)
- `time`: Simulation time (optional)

---

## Path Generation Methods

### LEP (Low Energy Path)

Stochastic trajectory generation using Langevin dynamics with momentum in latent space.

**Pros**: Explores multiple pathways, handles conformational heterogeneity  
**Cons**: Stochastic, may not find true minimum energy path

```yaml
method: "LEP"
steps: 1000
step_size: 1e-10
momentum: 0.9
```

### NEB (Nudged Elastic Band)

Deterministic optimization to find minimum energy pathways.

**Pros**: Finds true MEP, identifies transition states  
**Cons**: Deterministic, computationally intensive

```yaml
method: "NEB"
n_pivots: 20
neb_steps: 50
neb_lr: 0.05
```

---

## Model Architecture

ProtSCAPE combines several key components:

1. **EGNN Layers**: SE(3)-equivariant message passing preserves geometric structure
2. **Scattering Transform**: Multi-scale wavelet-based feature extraction
3. **Transformer Encoder**: Self-attention over atomic features
4. **Bottleneck Module**: Compresses to low-dimensional latent space
5. **Multi-Task Decoder**: Predicts atomic features and 3D coordinates

**Loss Functions**:
- Cross-entropy for discrete features (atomic number, residue, amino acid)
- Kabsch-aligned MSE for 3D coordinates (Procrustes distance)
- Optional energy prediction loss

---

## Evaluation Metrics

### Structure Quality
- **Kabsch RMSD**: Rotation-invariant coordinate accuracy
- **MolProbity Score**: Overall structure quality
- **Clashscore**: Steric clash detection
- **Ramachandran**: Backbone dihedral angle validation

### Latent Space
- **PCA/PHATE**: Visualization of learned manifold
- **Energy Correlation**: Latent space energy landscape fidelity

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---