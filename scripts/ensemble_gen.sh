#!/bin/bash

#SBATCH --job-name=ensemble_gen
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=devel
#SBATCH --mem=128G
#SBATCH --output=./logs/slurm/%x_%j.out
#SBATCH --error=./logs/slurm/%x_%j.err

cd /gpfs/gibbs/pi/krishnaswamy_smita/sv496/ProtSCAPE-Net
module load miniconda
conda activate mfcn

# Run training on all proteins in parallel
python ensemble_gen.py --n_pdb_samples 1000