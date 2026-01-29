#!/bin/bash
# Example script to run ensemble generation with PDB export
# 
# Usage: bash run_ensemble_generation.sh

# PROTEIN="7jfl"
# MODEL_PATH="train_logs/progsnn_logs_run_atlas_2026-01-26-120329/model_FINAL_7jfl.pt"
# OUTPUT_DIR="Generation/Ensemble/${PROTEIN}"

echo "================================================================"
echo "Running Ensemble Generation for ${PROTEIN}"
echo "================================================================"
echo ""
echo "Output directory: ${OUTPUT_DIR}"
echo "Model path: ${MODEL_PATH}"
echo ""

python ensemble_gen.py --n_pdb_samples 1000

echo ""
echo "================================================================"
echo "Ensemble generation complete!"
echo "================================================================"
echo ""
echo "Generated files:"
echo "  - Latent samples: ${OUTPUT_DIR}/generated_samples.npy"
echo "  - PDB files: ${OUTPUT_DIR}/generated_pdbs/"
echo "  - Metrics: ${OUTPUT_DIR}/metrics.json"
echo "  - Visualizations: ${OUTPUT_DIR}/*.png"
echo ""
