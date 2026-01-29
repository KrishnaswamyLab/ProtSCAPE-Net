# ProtScape Latent Diffusion

latent diffusion model for generating protein representations.

This model compresses 128D protein embeddings into a 16D latent space using an Autoencoder (AE), then trains a Denoising Diffusion Probabilistic Model (DDPM) on that compressed space.

Note: I found that training directly on 128D or even 32D wasn't working well due to limited sample size. This 2-stage approach (128D -> 16D -> Diffusion) was necessary to get the generated distributions to match the real data.

## How to Run

We provide a main script to run the entire pipeline easily.

### 1. Run Pipeline
```bash
python run_pipeline.py --input_data data/latents_zrep_10k.npy --checkpoint_dir checkpoints --output_dir outputs
```
This single command will:
1.  Train the Autoencoder (compress to 16D).
2.  Train the DDPM on the compressed latents.
3.  Generate samples, metrics, and visualisations.

### 2. Manual Execution
You can still run individual steps if you prefer:

**Train Autoencoder:**
```bash
python train_ae.py
```

**Train Diffusion:**
```bash
python train_ddpm.py
```

**Generate Results:**
```bash
python generate.py
```

## Options
The `run_pipeline.py` script supports several flags:
- `--input_data`: Path to input .npy file (default: `data/latents_zrep_10k.npy`)
- `--checkpoint_dir`: Directory for checkpoints (default: `checkpoints`)
- `--output_dir`: Directory for outputs (default: `outputs`)
- `--skip_ae`: Skip autoencoder training (use existing checkpoint).
- `--skip_ddpm`: Skip diffusion training (use existing checkpoint).

## Key Files
- `run_pipeline.py`: Master orchestration script.
- `config.py`: Main configuration.
- `train_ae.py`, `train_ddpm.py`, `generate.py`: Individual stage scripts.
- `utils.py`: Helper classes (EMA, Normalizer, Metrics).
