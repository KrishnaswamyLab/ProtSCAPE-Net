#!/bin/bash

#SBATCH --job-name=ATLAS
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=256G
#SBATCH --output=./logs/slurm/%x_%j.out
#SBATCH --error=./logs/slurm/%x_%j.err

cd /gpfs/gibbs/pi/krishnaswamy_smita/sv496/PROTSCAPE
module load miniconda
conda activate mfcn

# Run training on all proteins in parallel
python train.py --config configs/config.yaml --protein 7lp1 &
PID1=$!

python train.py --config configs/config.yaml --protein 7jfl &
PID2=$!

python train.py --config configs/config.yaml --protein 6p5h &
PID3=$!

# Wait for all background processes to complete
wait $PID1 $PID2 $PID3
echo "All training jobs completed!"