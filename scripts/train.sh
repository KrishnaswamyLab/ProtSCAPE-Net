#!/bin/bash

#SBATCH --job-name=ATLAS
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=256G
#SBATCH --output=./logs/slurm/%x_%j.out
#SBATCH --error=./logs/slurm/%x_%j.err

cd /gpfs/gibbs/pi/krishnaswamy_smita/sv496/ProtSCAPE-Net
module load miniconda
conda activate mfcn

# Run training on all proteins in parallel
python train.py --config configs/config.yaml --protein 7jfl