# ProtSCAPE-Net Ablation Studies Guide

This directory contains configuration files for running ablation studies on ProtSCAPE-Net.

## Quick Start

Run any ablation with:
```bash
python train.py --config configs/ablations/<ablation_config>.yaml
```

## Available Ablations

### 1. Feature Extractor Ablations (Scattering vs GCN)

**Purpose**: Test whether the scattering/wavelet transform is important or if standard GCN layers work just as well.

- **`ablate_scattering_to_gcn.yaml`**: Replace scattering with multi-layer GCN that mimics scattering's multi-scale structure
- **`ablate_simple_gcn.yaml`**: Replace scattering with a simpler GCN baseline

**Usage**:
```bash
# GCN with multi-scale structure
python train.py --config configs/ablations/ablate_scattering_to_gcn.yaml

# Simple GCN baseline
python train.py --config configs/ablations/ablate_simple_gcn.yaml
```

### 2. Node Feature Ablations

**Purpose**: Determine which node features are most important for the model.

- **`ablate_no_atomic_number.yaml`**: Remove atomic number (Z) information
- **`ablate_no_residue_aa.yaml`**: Remove residue index and amino acid type (no sequence info)
- **`ablate_xyz_only.yaml`**: Use only XYZ coordinates (remove all categorical features)
- **`ablate_randomize_features.yaml`**: Randomize categorical features to test if model actually uses them

**Usage**:
```bash
# Test without atomic numbers
python train.py --config configs/ablations/ablate_no_atomic_number.yaml

# Test without sequence information
python train.py --config configs/ablations/ablate_no_residue_aa.yaml

# Test with only coordinates
python train.py --config configs/ablations/ablate_xyz_only.yaml

# Test with randomized features
python train.py --config configs/ablations/ablate_randomize_features.yaml
```

### 3. Edge Feature Ablations

**Purpose**: Test the importance of edge attributes in the EGNN message passing.

- **`ablate_no_edge_features.yaml`**: Don't use edge features at all
- **`ablate_zero_edge_features.yaml`**: Set edge features to zero (maintains architecture)

**Usage**:
```bash
# Remove edge features entirely
python train.py --config configs/ablations/ablate_no_edge_features.yaml

# Zero out edge features
python train.py --config configs/ablations/ablate_zero_edge_features.yaml
```

## Custom Ablations

You can create custom ablation configurations by modifying the ablation parameters in any config file:

```yaml
# Feature extractor options
feature_extractor: "scattering"  # or "gcn" or "simple_gcn"
gcn_num_layers: 4
gcn_hidden_channels: null

# Node feature ablations
ablate_node_features:
  use_atomic_number: true        # Set to false to mask
  use_residue_index: true
  use_amino_acid: true
  use_xyz: true
  randomize_atomic_number: false # Set to true to randomize
  randomize_residue_index: false
  randomize_amino_acid: false
  randomize_xyz: false

# Edge feature ablations
ablate_edge_features:
  use_edge_features: true        # Set to false to remove
  randomize_edge_features: false # Set to true to randomize
  zero_edge_features: false      # Set to true to zero out
```

## Running Ablation Sweeps

To run multiple ablations in sequence:

```bash
# Using a bash loop
for config in configs/ablations/*.yaml; do
    echo "Running ablation: $config"
    python train.py --config $config
done
```

Or create a batch script:

```bash
#!/bin/bash
# run_ablations.sh

CONFIGS=(
    "configs/ablations/ablate_scattering_to_gcn.yaml"
    "configs/ablations/ablate_simple_gcn.yaml"
    "configs/ablations/ablate_no_atomic_number.yaml"
    "configs/ablations/ablate_no_residue_aa.yaml"
    "configs/ablations/ablate_xyz_only.yaml"
    "configs/ablations/ablate_no_edge_features.yaml"
)

for config in "${CONFIGS[@]}"; do
    echo "========================================"
    echo "Running: $config"
    echo "========================================"
    python train.py --config "$config"
done
```

Then run:
```bash
chmod +x run_ablations.sh
./run_ablations.sh
```

## Tracking Results

All ablation runs will log to Weights & Biases under the project name specified in the config (default: `protscape-ablations`). Compare runs by:

1. Going to your W&B project
2. Selecting multiple runs
3. Comparing metrics like:
   - Reconstruction loss
   - Coordinate loss (Kabsch MSE)
   - Feature classification accuracy
   - Training time

## Adding New Ablations

To add a new ablation type:

1. Add the ablation logic to [protscape/protscape.py](protscape/protscape.py)
2. Add config parameters to [configs/config.yaml](configs/config.yaml)
3. Create a config file in `configs/ablations/`
4. Update this README

## Tips

- Start with the default config to get baseline performance
- Run ablations with the same random seed for fair comparison
- Consider running multiple seeds for statistical significance
- Monitor both validation loss and reconstruction quality metrics
