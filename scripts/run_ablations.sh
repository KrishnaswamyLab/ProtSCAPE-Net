#!/bin/bash
#SBATCH --job-name=7lp1_ablations
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=512G
#SBATCH --output=./logs/slurm/%x_%j.out
#SBATCH --error=./logs/slurm/%x_%j.err

# SLURM script to run all ablation studies
# Usage: sbatch scripts/run_ablations.sh

echo "======================================"
echo "ProtSCAPE-Net Ablation Studies"
echo "True Ablations: Components completely removed"
echo "======================================"

PROTEIN=${1:-"7lp1"}  # Default to 7lp1 if not provided

ABLATIONS=(
    # "configs/ablation_gcn.yaml"              # Architecture ablation
    "configs/ablation_no_atom_type.yaml"     # Node feature: remove atom types
    "configs/ablation_no_edges.yaml"         # Edge feature ablation
    "configs/ablation_no_atom_noAA.yaml"         # Multiple ablations
)

for config in "${ABLATIONS[@]}"; do
    echo ""
    echo "======================================"
    echo "Running ablation: $(basename $config .yaml)"
    echo "Config: $config"
    echo "Protein: $PROTEIN"
    echo "======================================"
    python train.py --config "$config" --protein "$PROTEIN"
    
    if [ $? -eq 0 ]; then
        echo "✓ Completed: $(basename $config)"
    else
        echo "✗ Failed: $(basename $config)"
    fi
done

echo ""
echo "======================================"
echo "All ablations completed!"
echo "Check WandB: project 'protscape-ablations'"
echo "======================================"
