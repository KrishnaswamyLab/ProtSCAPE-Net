# Ensemble Generation Configuration Guide

## Overview

The ensemble generation pipeline uses a single YAML configuration file (`config_ensemble.yaml`) that you edit for different proteins and experiments. This ensures consistency across all your runs.

## Quick Start

1. **Edit the config file:**
   ```bash
   nano configs/config_ensemble.yaml
   # or use your preferred editor
   ```

2. **Update key parameters:**
   - Set `protein` name
   - Update `original_data_path` to your latent embeddings
   - Set `model_path` to your trained ProtSCAPE model
   - Update `pdb` path to reference structure
   - Adjust paths for `checkpoint_dir` and `output_dir`

3. **Run the pipeline:**
   ```bash
   python ensemble_gen.py --config configs/config_ensemble.yaml
   ```

## Usage Examples

### Basic Usage

```bash
# Run full pipeline (AE training + DDPM training + generation + PDB export)
python ensemble_gen.py --config configs/config_ensemble.yaml
```

### Skip Training Stages

```bash
# Skip autoencoder training if already done
python ensemble_gen.py --config configs/config_ensemble.yaml --skip_ae

# Skip both training stages (only generate samples)
python ensemble_gen.py --config configs/config_ensemble.yaml --skip_ae --skip_ddpm
```

### Override Parameters

```bash
# Override specific parameters without editing YAML
python ensemble_gen.py --config configs/config_ensemble.yaml \
    --n_pdb_samples 200 \
    --epochs 500 \
    --output_dir custom_output/
```

## Configuration File Structure

### Essential Parameters to Edit

```yaml
# 1. Set your protein name
protein: "6h86"  # Change to: "7jfl", "7lp1", "6h86_gcn", etc.

# 2. Path to your latent embeddings
original_data_path: "Inference/6h86/latents_zrep_10k.npy"

# 3. Path to trained ProtSCAPE model
model_path: "train_logs/.../model_FINAL_6h86.pt"

# 4. Reference PDB structure
pdb: "data/datasets/6h86_A_protein/6h86_A.pdb"

# 5. Output directories (usually based on protein name)
checkpoint_dir: "checkpoints/6h86"
output_dir: "Generation/Ensemble/6h86"
```

### Common Scenarios

#### Scenario 1: Standard Protein (e.g., 7jfl)

```yaml
protein: "7jfl"
original_data_path: "Inference/7jfl/latents_zrep_10k.npy"
model_path: "train_logs/.../model_FINAL_7jfl.pt"
pdb: "data/datasets/7jfl_A_protein/7jfl_A.pdb"
checkpoint_dir: "checkpoints/7jfl"
output_dir: "Generation/Ensemble/7jfl"
```

#### Scenario 2: Ablation Study (e.g., 6h86 GCN)

```yaml
protein: "6h86_gcn"
original_data_path: "Ablations/6h86/inference_test/6h86_GCN/latents_zrep_10k.npy"
model_path: "train_logs/.../model_FINAL_6h86_gcn.pt"
pdb: "data/datasets/6h86_A_protein/6h86_A.pdb"
checkpoint_dir: "checkpoints/6h86"
output_dir: "Generation/Ensemble/Ablations/6h86_gcn"
```

## Configuration Parameters Reference

### Data Configuration
- `input_dim`: Original latent dimension (usually 128)
- `latent_dim`: Compressed latent dimension for DDPM (usually 16)
- `original_data_path`: Path to input latent representations (.npy file)

### Model Architecture
- `data_dim`: Should match `latent_dim`
- `hidden_dims`: DDPM hidden layers (default: [512, 1024, 1024, 512])
- `dropout`: Dropout rate (default: 0.1)

### Training Parameters
- `batch_size`: Training batch size (default: 256)
- `epochs`: Number of training epochs (default: 1000)
- `learning_rate`: Learning rate (default: 0.0001)
- `weight_decay`: L2 regularization (default: 0.0001)

### Diffusion Parameters
- `timesteps`: Number of diffusion steps (default: 1000)
- `beta_schedule`: Noise schedule ("cosine" or "linear")
- `clip_x0`: Clipping range for generation (default: 3.5)

### PDB Export
- `decode_to_coords`: Enable coordinate decoding (default: true)
- `n_pdb_samples`: Number of structures to export (default: 100)
- `model_path`: Path to trained ProtSCAPE model checkpoint
- `dataset_path`: Directory containing graph datasets
- `pdb`: Reference PDB structure path

### XYZ Normalization (Optional)
Only needed if coordinates were normalized during training:
- `normalize_xyz`: Enable denormalization (default: false)
- `xyz_mu_path`: Path to mean values
- `xyz_sd_path`: Path to std dev values

## Pipeline Stages

The pipeline runs in 4 stages:

1. **Autoencoder Training** - Compress latent space (skip with `--skip_ae`)
2. **DDPM Training** - Train diffusion model (skip with `--skip_ddpm`)
3. **Generation** - Generate new conformational samples
4. **PDB Export** - Decode to coordinates and export structures

## Output Files

After completion, the output directory contains:

```
Generation/Ensemble/{protein}/
├── generated_samples.npy          # Generated latent samples
├── molprobity_scores.csv          # Structure quality metrics
└── generated_pdbs/                # PDB structure files
    ├── pred_frame_00000.pdb
    ├── pred_frame_00001.pdb
    └── ...
```

## Tips & Best Practices

1. **Reusing Checkpoints**: Use `--skip_ae` or `--skip_ddpm` if checkpoints already exist from previous runs

2. **Memory Management**: Reduce `batch_size` if you encounter OOM errors

3. **Quick Testing**: Use fewer `epochs` (e.g., 100) and smaller `n_pdb_samples` (e.g., 50) for testing

4. **Organized Outputs**: Use descriptive protein names and output directories for ablations

5. **Version Control**: Keep a copy of your config file with results for reproducibility

## Common Workflows

### Workflow 1: First Run (Full Pipeline)
```bash
# 1. Edit config for your protein
nano configs/config_ensemble.yaml

# 2. Run full pipeline
python ensemble_gen.py --config configs/config_ensemble.yaml
```

### Workflow 2: Generate More Samples (Reuse Training)
```bash
# Skip training, only generate and export new samples
python ensemble_gen.py --config configs/config_ensemble.yaml --skip_ae --skip_ddpm
```

### Workflow 3: Different Sample Count
```bash
# Generate more/fewer PDB structures
python ensemble_gen.py --config configs/config_ensemble.yaml \
    --skip_ae --skip_ddpm \
    --n_pdb_samples 500
```

### Workflow 4: Quick Test Run
```bash
# Test with minimal training
python ensemble_gen.py --config configs/config_ensemble.yaml \
    --epochs 100 \
    --n_pdb_samples 50
```

### Workflow 5: Ablation Study
```bash
# 1. Edit config for ablation
#    protein: "6h86_gcn"
#    original_data_path: "Ablations/6h86/.../latents_zrep_10k.npy"
#    output_dir: "Generation/Ensemble/Ablations/6h86_gcn"

# 2. Run pipeline
python ensemble_gen.py --config configs/config_ensemble.yaml
```

## Troubleshooting

**Q: "Config file not found" error**
- Check that you're running from the project root directory
- Use absolute path: `--config /full/path/to/config_ensemble.yaml`

**Q: Out of memory during training**
- Reduce `batch_size` in config (try 128 or 64)
- Use `--skip_ae` or `--skip_ddpm` to skip memory-intensive stages

**Q: PDB export fails**
- Verify `model_path` exists and matches your protein
- Check `pdb` reference structure exists
- Ensure `dataset_path` contains matching graph files

**Q: Want to generate more samples later**
- Use `--skip_ae --skip_ddpm` to reuse trained models
- Adjust `n_pdb_samples` as needed

## Example: Complete Setup for New Protein

```yaml
# configs/config_ensemble.yaml

# 1. Basic info
protein: "myprotein"

# 2. Input data
original_data_path: "Inference/myprotein/latents_zrep_10k.npy"

# 3. Paths
checkpoint_dir: "checkpoints/myprotein"
output_dir: "Generation/Ensemble/myprotein"

# 4. Model and reference
model_path: "train_logs/run_xyz/model_FINAL_myprotein.pt"
pdb: "data/datasets/myprotein_A_protein/myprotein_A.pdb"

# 5. Dataset
dataset_path: "data/graphs/"

# Keep other parameters as defaults
```

Then run:
```bash
python ensemble_gen.py --config configs/config_ensemble.yaml
```

