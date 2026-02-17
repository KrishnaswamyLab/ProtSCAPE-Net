# Ensemble Generation Configuration

## Quick Start

1. **Edit the configuration file:**
   ```bash
   nano config_ensemble.yaml
   ```

2. **Update these key fields:**
   - `protein`: Your protein name (e.g., "6h86", "7jfl", "6h86_gcn")
   - `original_data_path`: Path to your latent embeddings
   - `model_path`: Path to your trained ProtSCAPE model
   - `pdb`: Path to reference PDB structure
   - `checkpoint_dir` and `output_dir`: Output paths

3. **Run the pipeline:**
   ```bash
   python ensemble_gen.py --config configs/config_ensemble.yaml
   ```

## Configuration File

**`config_ensemble.yaml`** - Main configuration file for ensemble generation
- Edit this file for different proteins and experiments
- Contains all parameters for AE training, DDPM training, and PDB export
- Includes commented examples for common scenarios

## Documentation

See **`ENSEMBLE_CONFIG_GUIDE.md`** for:
- Complete parameter reference
- Configuration examples
- Common workflows
- Troubleshooting tips

## Usage Examples

```bash
# Full pipeline
python ensemble_gen.py --config configs/config_ensemble.yaml

# Skip training (reuse checkpoints)
python ensemble_gen.py --config configs/config_ensemble.yaml --skip_ae --skip_ddpm

# Override parameters
python ensemble_gen.py --config configs/config_ensemble.yaml --n_pdb_samples 200
```

## File Structure

```
configs/
├── config_ensemble.yaml          # Main configuration (EDIT THIS)
├── ENSEMBLE_CONFIG_GUIDE.md      # Detailed documentation
└── README_ENSEMBLE.md            # This file
```
