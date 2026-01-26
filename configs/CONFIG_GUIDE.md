# Configuration Guide

This project uses YAML config files to manage training and inference parameters instead of command-line arguments.

## Quick Start

### Training

1. Create a config file based on `config.yaml`:
```bash
cp config.yaml my_experiment.yaml
```

2. Edit `my_experiment.yaml` with your desired parameters

3. Run training:
```bash
python train.py --config my_experiment.yaml
```

### Inference

1. Create a config file based on `config_inference.yaml`:
```bash
cp config_inference.yaml my_inference.yaml
```

2. Edit `my_inference.yaml` (especially set `ckpt_path` to your model checkpoint)

3. Run inference:
```bash
python inference.py --config my_inference.yaml
```

## Config File Format

Config files are YAML format. Example:

```yaml
# Dataset
dataset: "atlas"
protein: "1bx7"
pkl_path: "path/to/graphs.pkl"

# Model hyperparameters
latent_dim: 128
hidden_dim: 256
embedding_dim: 128

# Training
batch_size: 32
n_epochs: 500
lr: 1e-3

# ... more parameters
```

## Command-Line Overrides

You can override any config parameter from the command line without editing the config file:

```bash
# Override single parameters
python train.py --config config.yaml --batch_size 64 --n_epochs 1000

# Override multiple parameters
python train.py --config config.yaml --latent_dim 256 --hidden_dim 512 --lr 5e-4
```

## Utility Functions

The config system is implemented in `utils/config.py` with three main functions:

### `load_config(config_path)`
Loads a config file (YAML or JSON) and returns a dictionary.

```python
from utils.config import load_config
config = load_config("config.yaml")
```

### `config_to_hparams(config)`
Converts a config dictionary to `argparse.Namespace` for easy attribute access.

```python
from utils.config import config_to_hparams
args = config_to_hparams(config)
print(args.batch_size)  # Access as attribute
```

### `save_config(config, output_path, format='yaml')`
Saves a config dictionary to a file (useful for recording training configs).

```python
from utils.config import save_config
save_config(config, "saved_config.yaml")
```

## Config Files Included

- **`config.yaml`**: Default training config with typical hyperparameters
- **`config_inference.yaml`**: Default inference config

Create experiment-specific config files by copying and modifying these templates.

## Benefits

- **Reproducibility**: Easily save and share configs used for experiments
- **Organization**: Group related parameters by section
- **Flexibility**: Mix config file defaults with command-line overrides
- **Version Control**: Track config changes alongside code changes
