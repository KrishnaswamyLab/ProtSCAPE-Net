#!/bin/bash

#SBATCH --job-name=atlas_selected_train_energy_ablated
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=256G
#SBATCH --output=./logs/slurm/%x_%j.out
#SBATCH --error=./logs/slurm/%x_%j.err

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
module load uv
source .venv/bin/activate

# Disable wandb to reduce memory overhead during batch training.
export WANDB_DISABLED=true
export WANDB_MODE=disabled

# Train specific proteins by PDB/protein ID.
# Priority: CLI args > PDB_IDS env var (space-separated) > default (6h86).
declare -a TARGET_IDS
if [[ "$#" -gt 0 ]]; then
	TARGET_IDS=("$@")
elif [[ -n "${PDB_IDS:-}" ]]; then
	# shellcheck disable=SC2206
	TARGET_IDS=(${PDB_IDS})
else
	TARGET_IDS=("6h86")
fi

echo "[info] Training selected IDs: ${TARGET_IDS[*]}"

uv run "$REPO_ROOT/train_all_proteins.py" \
	--config "$REPO_ROOT/configs/config.yaml" \
	--graphs_dir "$REPO_ROOT/data/graphs" \
	--protein_ids "${TARGET_IDS[@]}"
