#!/bin/bash

#SBATCH --job-name=atlas_6p5h
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --mem=512G
#SBATCH --output=./logs/slurm/%x_%j.out
#SBATCH --error=./logs/slurm/%x_%j.err

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
module load miniconda
conda activate mfcn

# Run training on all proteins in parallel
python "$REPO_ROOT/train.py" --config "$REPO_ROOT/configs/config_ubiquitin.yaml" --protein 6p5h
