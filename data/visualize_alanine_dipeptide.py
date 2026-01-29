"""
Visualize Alanine Dipeptide energy landscape and conformational states.

This script creates Ramachandran-like plots showing the two local minima
in the energy landscape of alanine dipeptide.

Usage:
    python visualize_alanine_dipeptide.py [--graphs <FILE>] [--output <DIR>]
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def load_graphs(graphs_path: str):
    """Load pickled graph dataset."""
    with open(graphs_path, "rb") as f:
        graphs = pickle.load(f)
    return graphs


def plot_ramachandran(graphs, output_dir: str = "visualizations"):
    """
    Plot Ramachandran plot colored by energy.
    
    Args:
        graphs: List of PyG graph objects
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    phis = np.array([g.phi.item() for g in graphs])
    psis = np.array([g.psi.item() for g in graphs])
    energies = np.array([g.energy.item() for g in graphs])
    
    # Create Ramachandran plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Scatter with energy coloring
    ax = axes[0]
    scatter = ax.scatter(phis, psis, c=energies, cmap='viridis', s=10, alpha=0.6)
    ax.set_xlabel('Phi (degrees)', fontsize=12)
    ax.set_ylabel('Psi (degrees)', fontsize=12)
    ax.set_title('Alanine Dipeptide Ramachandran Plot (Energy)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Energy (kJ/mol)', fontsize=11)
    
    # Mark conformational states
    c7eq_mask = (phis < 0) & (psis > 0)
    c7ax_mask = (phis > 0) & (psis < 0)
    
    if c7eq_mask.sum() > 0:
        c7eq_center = (phis[c7eq_mask].mean(), psis[c7eq_mask].mean())
        ax.plot(c7eq_center[0], c7eq_center[1], 'r*', markersize=20, 
                label=f'C7eq center', markeredgecolor='white', markeredgewidth=1)
    
    if c7ax_mask.sum() > 0:
        c7ax_center = (phis[c7ax_mask].mean(), psis[c7ax_mask].mean())
        ax.plot(c7ax_center[0], c7ax_center[1], 'b*', markersize=20, 
                label=f'C7ax center', markeredgecolor='white', markeredgewidth=1)
    
    ax.legend(fontsize=10)
    
    # Plot 2: 2D histogram
    ax = axes[1]
    h, xedges, yedges = np.histogram2d(phis, psis, bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(h.T, extent=extent, origin='lower', cmap='hot', aspect='auto')
    ax.set_xlabel('Phi (degrees)', fontsize=12)
    ax.set_ylabel('Psi (degrees)', fontsize=12)
    ax.set_title('Conformational Density', fontsize=14)
    ax.grid(False)
    cbar2 = plt.colorbar(im, ax=ax)
    cbar2.set_label('Count', fontsize=11)
    
    plt.tight_layout()
    output_file = output_path / "ramachandran_plot.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[saved] {output_file}")
    plt.close()


def plot_energy_landscape(graphs, output_dir: str = "visualizations"):
    """
    Plot energy landscape and time evolution.
    
    Args:
        graphs: List of PyG graph objects
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    times = np.array([g.time.item() for g in graphs])
    energies = np.array([g.energy.item() for g in graphs])
    phis = np.array([g.phi.item() for g in graphs])
    psis = np.array([g.psi.item() for g in graphs])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Energy vs time
    ax = axes[0, 0]
    ax.plot(times, energies, linewidth=0.5, alpha=0.7)
    ax.set_xlabel('Time (ps)', fontsize=11)
    ax.set_ylabel('Energy (kJ/mol)', fontsize=11)
    ax.set_title('Potential Energy vs Time', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Energy histogram
    ax = axes[0, 1]
    ax.hist(energies, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Energy (kJ/mol)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Energy Distribution', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    ax.axvline(energies.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {energies.mean():.2f}')
    ax.axvline(energies.min(), color='g', linestyle='--', linewidth=2, label=f'Min: {energies.min():.2f}')
    ax.legend(fontsize=9)
    
    # Plot 3: Phi vs time
    ax = axes[1, 0]
    ax.plot(times, phis, linewidth=0.5, alpha=0.7, label='Phi')
    ax.plot(times, psis, linewidth=0.5, alpha=0.7, label='Psi')
    ax.set_xlabel('Time (ps)', fontsize=11)
    ax.set_ylabel('Dihedral Angle (degrees)', fontsize=11)
    ax.set_title('Phi/Psi Dihedrals vs Time', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.axhline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Plot 4: Conformational state occupancy
    ax = axes[1, 1]
    c7eq_mask = (phis < 0) & (psis > 0)
    c7ax_mask = (phis > 0) & (psis < 0)
    other_mask = ~(c7eq_mask | c7ax_mask)
    
    occupancies = [
        c7eq_mask.sum(),
        c7ax_mask.sum(),
        other_mask.sum()
    ]
    labels = ['C7eq\n(extended)', 'C7ax\n(compact)', 'Transition']
    colors = ['#ff7f0e', '#1f77b4', '#d62728']
    
    wedges, texts, autotexts = ax.pie(
        occupancies, labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=90, textprops={'fontsize': 11}
    )
    ax.set_title('Conformational State Occupancy', fontsize=12)
    
    plt.tight_layout()
    output_file = output_path / "energy_landscape.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[saved] {output_file}")
    plt.close()


def plot_state_transitions(graphs, output_dir: str = "visualizations"):
    """
    Identify and plot conformational state transitions.
    
    Args:
        graphs: List of PyG graph objects
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    times = np.array([g.time.item() for g in graphs])
    phis = np.array([g.phi.item() for g in graphs])
    psis = np.array([g.psi.item() for g in graphs])
    energies = np.array([g.energy.item() for g in graphs])
    
    # Identify conformational states
    states = np.zeros(len(graphs), dtype=int)  # 0: other, 1: C7eq, 2: C7ax
    c7eq_mask = (phis < 0) & (psis > 0)
    c7ax_mask = (phis > 0) & (psis < 0)
    states[c7eq_mask] = 1
    states[c7ax_mask] = 2
    
    # Find transitions
    transitions = []
    for i in range(1, len(states)):
        if states[i] != states[i-1] and states[i] != 0 and states[i-1] != 0:
            transitions.append(i)
    
    print(f"[info] Found {len(transitions)} direct state transitions")
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Color by state
    colors = ['gray', 'orange', 'blue']
    for state_id, state_name in enumerate(['Transition', 'C7eq', 'C7ax']):
        mask = states == state_id
        if mask.sum() > 0:
            ax.scatter(times[mask], energies[mask], c=colors[state_id], 
                      label=state_name, s=10, alpha=0.6)
    
    # Mark transitions
    for t_idx in transitions:
        ax.axvline(times[t_idx], color='red', linestyle='--', 
                  linewidth=0.5, alpha=0.3)
    
    ax.set_xlabel('Time (ps)', fontsize=12)
    ax.set_ylabel('Energy (kJ/mol)', fontsize=12)
    ax.set_title(f'Conformational States and Transitions ({len(transitions)} transitions)', fontsize=14)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_path / "state_transitions.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[saved] {output_file}")
    plt.close()


def print_summary(graphs):
    """Print summary statistics."""
    energies = np.array([g.energy.item() for g in graphs])
    phis = np.array([g.phi.item() for g in graphs])
    psis = np.array([g.psi.item() for g in graphs])
    
    c7eq_mask = (phis < 0) & (psis > 0)
    c7ax_mask = (phis > 0) & (psis < 0)
    
    print("\n" + "="*70)
    print("ALANINE DIPEPTIDE DATASET SUMMARY")
    print("="*70)
    print(f"Total frames: {len(graphs)}")
    print(f"Total atoms per frame: {graphs[0].num_nodes}")
    print(f"Total edges per frame: {graphs[0].edge_index.shape[1]}")
    
    print(f"\nEnergy Statistics:")
    print(f"  Min:  {energies.min():.2f} kJ/mol")
    print(f"  Max:  {energies.max():.2f} kJ/mol")
    print(f"  Mean: {energies.mean():.2f} kJ/mol")
    print(f"  Std:  {energies.std():.2f} kJ/mol")
    
    print(f"\nConformational States:")
    print(f"  C7eq (extended, Phi<0, Psi>0): {c7eq_mask.sum()} frames ({100*c7eq_mask.sum()/len(graphs):.1f}%)")
    if c7eq_mask.sum() > 0:
        print(f"    Mean energy: {energies[c7eq_mask].mean():.2f} kJ/mol")
        print(f"    Mean Phi: {phis[c7eq_mask].mean():.2f}°, Psi: {psis[c7eq_mask].mean():.2f}°")
    
    print(f"  C7ax (compact, Phi>0, Psi<0): {c7ax_mask.sum()} frames ({100*c7ax_mask.sum()/len(graphs):.1f}%)")
    if c7ax_mask.sum() > 0:
        print(f"    Mean energy: {energies[c7ax_mask].mean():.2f} kJ/mol")
        print(f"    Mean Phi: {phis[c7ax_mask].mean():.2f}°, Psi: {psis[c7ax_mask].mean():.2f}°")
    
    if c7eq_mask.sum() > 0 and c7ax_mask.sum() > 0:
        energy_diff = abs(energies[c7eq_mask].mean() - energies[c7ax_mask].mean())
        print(f"\n  Energy difference between states: {energy_diff:.2f} kJ/mol")
        print(f"  ✓ Two distinct local minima detected!")
    else:
        print(f"\n  ⚠ Warning: Only one conformational state present")
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Alanine Dipeptide energy landscape and conformational states"
    )
    parser.add_argument(
        "--graphs",
        default="graphs/alanine_dipeptide_graphs.pkl",
        help="Path to graphs pickle file"
    )
    parser.add_argument(
        "--output",
        default="visualizations/alanine_dipeptide",
        help="Output directory for plots"
    )
    
    args = parser.parse_args()
    
    # Load graphs
    try:
        print(f"[info] Loading graphs from {args.graphs}")
        graphs = load_graphs(args.graphs)
        print(f"[info] Loaded {len(graphs)} frames")
    except Exception as e:
        print(f"[error] Failed to load graphs: {e}")
        return 1
    
    # Print summary
    print_summary(graphs)
    
    # Create visualizations
    print(f"\n[info] Creating visualizations...")
    try:
        plot_ramachandran(graphs, args.output)
        plot_energy_landscape(graphs, args.output)
        plot_state_transitions(graphs, args.output)
        print(f"\n[done] All visualizations saved to {args.output}/")
    except Exception as e:
        print(f"[error] Failed to create visualizations: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
