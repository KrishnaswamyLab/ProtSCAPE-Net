#!/bin/bash
# Example: Run a quick test of each ablation type
# This script shows examples without actually running them

echo "ProtSCAPE-Net Ablation System Examples"
echo "======================================"
echo ""
echo "Note: These ablations COMPLETELY REMOVE components"
echo "(not just masking or zeroing)"
echo ""

# Test 1: GCN vs Scattering
echo "Example 1: Architecture ablation - GCN instead of scattering"
echo "  What's removed: Wavelet scattering transform"
echo "  What's added: Multi-layer GCN"
echo "  Command: python train.py --config configs/ablation_gcn.yaml --protein BPTI"
echo ""

# Test 2: Node feature ablation - no xyz
echo "Example 2: Node feature ablation - Remove coordinates"
echo "  What's removed: XYZ coordinates from input"
echo "  Input changes: [Z, res, aa, xyz] → [Z, res, aa]"
echo "  Command: python train.py --config configs/ablation_no_xyz.yaml --protein BPTI"
echo ""

# Test 3: Node feature ablation - no atom types
echo "Example 3: Node feature ablation - Remove atomic numbers"
echo "  What's removed: Atomic number (Z) from input"
echo "  Input changes: [Z, res, aa, xyz] → [res, aa, xyz]"
echo "  Command: python train.py --config configs/ablation_no_atom_type.yaml --protein BPTI"
echo ""

# Test 4: Edge feature ablation
echo "Example 4: Edge feature ablation - Remove edge attributes"
echo "  What's removed: Edge features in EGNN"
echo "  Architecture change: EGNN(edge_dim=N) → EGNN(edge_dim=0)"
echo "  Command: python train.py --config configs/ablation_no_edges.yaml --protein BPTI"
echo ""

# Test 5: Combined ablation
echo "Example 5: Combined ablation - Multiple components"
echo "  What's removed: Scattering + atom types + amino acids + edge features"
echo "  What remains: GCN + residue index + xyz only"
echo "  Command: python train.py --config configs/ablation_combined.yaml --protein BPTI"
echo ""

# Test 6: Custom ablation
echo "Example 6: Create custom ablation"
cat << 'EOF'
# Create configs/my_custom_ablation.yaml:

# Use GCN instead of scattering
feature_extractor: "gcn"
gcn_num_layers: 3

# Remove atomic numbers (completely)
ablate_node_features:
  use_atomic_number: false  # ← REMOVED from input
  use_residue_index: true
  use_amino_acid: true
  use_xyz: true

# Remove edge features (completely)
ablate_edge_features:
  use_edge_features: false  # ← No edge features (edge_dim=0)

# Then run:
python train.py --config configs/my_custom_ablation.yaml --protein BPTI
EOF

echo ""
echo "========================================="
echo "Available ablation configs:"
ls -1 configs/ablation*.yaml 2>/dev/null || echo "Run from project root directory"
echo "========================================="
echo ""
echo "Run all ablations: ./run_ablations.sh PROTEIN_NAME"
