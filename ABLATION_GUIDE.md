# ProtSCAPE-Net Ablation System

A comprehensive ablation study framework for ProtSCAPE-Net.

## What Are Ablations?

Ablations **completely remove** components from the model architecture, not just zero them out. When you ablate a feature:
- It's removed from the input
- Network dimensions adjust accordingly
- The model never sees that information

This is different from masking/zeroing, which keeps the feature dimension but sets values to zero.

## Summary

This system allows you to easily run ablation studies on three key components:

1. **Feature Extractor**: Scattering/Wavelets vs GCN layers
2. **Node Features**: Completely remove different node attributes
3. **Edge Features**: Completely remove edge attributes from message passing

## What Was Added

### New Files

1. **`protscape/gcn_layers.py`**: Two GCN alternatives to scattering transform
   - `GCN_Layer`: Multi-layer GCN that mimics scattering's multi-scale structure
   - `SimpleGCN_Layer`: Simpler GCN baseline without multi-scale design

2. **8 Pre-configured Ablation Configs** in `configs/ablations/`:
   - `ablate_scattering_to_gcn.yaml`
   - `ablate_simple_gcn.yaml`
   - `ablate_no_atomic_number.yaml`
   - `ablate_no_residue_aa.yaml`
   - `ablate_xyz_only.yaml`
   - `ablate_randomize_features.yaml`
   - `ablate_no_edge_features.yaml`
   - `ablate_zero_edge_features.yaml`

3. **`scripts/run_ablations.sh`**: Bash script to run all ablations sequentially

4. **`configs/ablations/README.md`**: Detailed documentation

### Modified Files

1. **`configs/config.yaml`**: Added ablation parameters section
2. **`protscape/protscape.py`**: Added ablation support:
   - Import GCN layers
   - Configurable feature extractor (scattering/gcn/simple_gcn)
   - Node feature ablation methods (complete removal)
   - Edge feature ablation methods (complete removal)
   - Dynamic dimension adjustment based on active features

## Quick Start

### Run a Single Ablation

```bash
# Replace scattering with GCN
python train.py --config configs/ablations/ablate_scattering_to_gcn.yaml

# Remove atomic numbers
python train.py --config configs/ablations/ablate_no_atomic_number.yaml

# Remove edge features
python train.py --config configs/ablations/ablate_no_edge_features.yaml
```

### Run All Ablations

```bash
chmod +x scripts/run_ablations.sh
./scripts/run_ablations.sh
```

### Custom Ablation

Edit any config or create a new one with these parameters:

```yaml
# 1. Feature Extractor Ablation
feature_extractor: "scattering"  # or "gcn" or "simple_gcn"
gcn_num_layers: 4
gcn_hidden_channels: null

# 2. Node Feature Ablation
ablate_node_features:
  use_atomic_number: true         # false = REMOVE from input entirely
  use_residue_index: true         # false = REMOVE from input entirely
  use_amino_acid: true            # false = REMOVE from input entirely
  use_xyz: true                   # false = REMOVE coordinates entirely
  
  # Optional: randomization for control experiments
  randomize_atomic_number: false  # true = replace with random values
  randomize_residue_index: false
  randomize_amino_acid: false
  randomize_xyz: false

# 3. Edge Feature Ablation
ablate_edge_features:
  use_edge_features: true         # false = REMOVE entirely (edge_dim=0)
  randomize_edge_features: false  # true = random edge features (control)
  zero_edge_features: false       # deprecated, same as use_edge_features: false
```

## Ablation Categories

### 1. Architecture Ablation (Scattering vs GCN)

**Question**: Is the scattering/wavelet transform important, or would standard GCN layers work just as well?

**Configs**:
- `ablate_scattering_to_gcn.yaml`: Multi-layer GCN with multi-scale structure
- `ablate_simple_gcn.yaml`: Simpler GCN baseline

**Expected Insight**: If scattering performs better, it suggests multi-scale spectral features are important. If GCN performs similarly, the architecture can be simplified.

### 2. Node Feature Ablation

**Question**: Which node features contribute most to model performance?

**How it works**: When you set `use_X: false`, that feature is **completely removed** from the input. For example:
- Full model: Input is `[Z, residue, aa]` → 3D embedding
- Ablate Z: Input is `[residue, aa]` → 2D embedding

The network architecture automatically adjusts to the reduced dimensionality.

**Configs**:
- `ablation_no_atom_type.yaml`: Remove atomic number Z
- `ablation_no_xyz.yaml`: Remove 3D coordinates (geometry-free)

**Expected Insight**: Identify which features are critical vs redundant. Performance drop indicates importance.

### 3. Edge Feature Ablation

**Question**: How important are edge attributes in the EGNN message passing?

**How it works**: When you set `use_edge_features: false`, edge features are **completely removed**:
- Full model: `EGNN(dim=input_dim, edge_dim=edge_attr_dim)`
- Ablated: `EGNN(dim=input_dim, edge_dim=0)` ← No edge features

Only graph topology (connectivity) is used, not edge attributes.

**Configs**:
- `ablation_no_edges.yaml`: Remove edge features entirely

**Expected Insight**: Understand if edge features provide useful information or if topology alone is sufficient.

## Additional Ablation Ideas

Here are more ablations you could implement following the same pattern:

### 4. Graph Connectivity Ablation
- Test different edge definitions (distance cutoffs, k-NN, etc.)
- Remove certain edge types (e.g., only backbone edges)

### 5. Loss Function Ablation
- Different alignment methods (Kabsch vs others)
- Different loss weightings
- Remove certain loss terms

### 6. EGNN Layer Ablation
- Vary number of EGNN layers (1, 2, 3, 5)
- Test coordinate update vs frozen coordinates
- Test with/without edge updates

### 7. Data Augmentation Ablation
- Random rotations
- Random noise addition
- Random subsampling

## Tracking and Analysis

All runs log to Weights & Biases (project: `protscape-ablations`). Compare:

- **Reconstruction loss**: Overall model performance
- **Coordinate loss**: Geometric accuracy (Kabsch MSE)
- **Feature losses**: Classification accuracy for Z, residue, AA
- **Training time**: Computational efficiency

## Implementation Details

The ablation system works by:

1. **Config-driven**: All ablations controlled via YAML config files
2. **Drop-in replacement**: GCN layers implement same API as scattering
3. **True removal**: Features completely removed, dimensions adjusted automatically
4. **No architecture changes needed**: Same training script works for all ablations

### How True Ablations Work

```python
# BEFORE (3 features):
feat_embed = nn.Linear(3, 128)  # Takes [Z, residue, aa]
input = [Z, residue, aa]        # Shape: (B, N, 3)

# AFTER ablating Z (2 features):
feat_embed = nn.Linear(2, 128)  # Takes [residue, aa]
input = [residue, aa]           # Shape: (B, N, 2)

# EGNN with edges:
egnn = EGNN(dim=input_dim, edge_dim=5)  # Uses edge features

# EGNN without edges (ablated):
egnn = EGNN(dim=input_dim, edge_dim=0)  # No edge dimension
```

This is fundamentally different from zeroing/masking, which would keep the dimension at 3 but set values to 0.

## Files Overview

```
ProtSCAPE-Net/
├── protscape/
│   ├── protscape.py          # Modified: Added ablation support
│   ├── gcn_layers.py         # New: GCN alternatives
│   └── wavelets.py           # Original scattering layer
├── configs/
│   ├── config.yaml           # Modified: Added ablation params
│   └── ablations/            # New directory
│       ├── README.md
│       ├── ablate_scattering_to_gcn.yaml
│       ├── ablate_simple_gcn.yaml
│       ├── ablate_no_atomic_number.yaml
│       ├── ablate_no_residue_aa.yaml
│       ├── ablate_xyz_only.yaml
│       ├── ablate_randomize_features.yaml
│       ├── ablate_no_edge_features.yaml
│       └── ablate_zero_edge_features.yaml
├── scripts/
│   └── run_ablations.sh      # New: Run all ablations
└── train.py                  # Unchanged: Works with all configs
```

## Notes

- All ablations preserve the model architecture where possible
- The system is backward compatible - default config runs standard model
- GCN layers produce same output shape as scattering for drop-in replacement
- Node/edge ablations are applied at runtime, not at data preprocessing
