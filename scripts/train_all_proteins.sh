#!/bin/bash

#SBATCH --job-name=protscape_train_all
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_rtx6000
#SBATCH --gpus=1
#SBATCH --mem=512G
#SBATCH --output=./logs/slurm/%x_%j.out
#SBATCH --error=./logs/slurm/%x_%j.err

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
module load uv
source .venv/bin/activate

# Disable wandb to reduce memory overhead during batch training.
export WANDB_DISABLED=true
export WANDB_MODE=disabled

# Run training on all proteins in parallel
uv run "$REPO_ROOT/train_all_proteins.py" --config "$REPO_ROOT/configs/config.yaml" --graphs_dir "$REPO_ROOT/data/graphs"
