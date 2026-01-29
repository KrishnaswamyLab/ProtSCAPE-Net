"""Visualization utilities for ProtSCAPE."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from utils.normalize import normalize_energy

def visualize_energy_distribution(full_dataset):
    frames = np.arange(len(full_dataset))
    # Plot energy after normalization
    # Plot energy before normalization
    energies_before = np.array([float(getattr(g, "energy", 0.0)) for g in full_dataset], dtype=np.float64)
    energy_mean, energy_std = normalize_energy(full_dataset)
    energies_after = np.array([float(g.energy) for g in full_dataset], dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Before normalization
    axes[0].plot(frames, energies_before, linewidth=0.5)
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("Energy")
    axes[0].set_title("Energy Before Normalization")
    axes[0].grid(True, alpha=0.3)

    # After normalization
    axes[1].plot(frames, energies_after, linewidth=0.5)
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Normalized Energy")
    axes[1].set_title("Energy After Normalization")
    axes[1].grid(True, alpha=0.3)
    print(f"[plot] Energy after norm: min={energies_after.min():.6f}, max={energies_after.max():.6f}, mean={energies_after.mean():.6f}, std={energies_after.std():.6f}")

    plt.tight_layout()
    plt.savefig("energy_before_after_normalization.png", dpi=150)
    plt.close()

# -----------------------------
# Embedding helpers (PCA + PHATE)
# -----------------------------
def compute_pca_2d(X: np.ndarray) -> np.ndarray:
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        Z = pca.fit_transform(X)
        return Z.astype(np.float32)
    except ImportError:
        X = np.asarray(X, dtype=np.float64)
        Xc = X - X.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        Z = Xc @ Vt.T[:, :2]
        return Z.astype(np.float32)


def try_compute_phate_2d(X: np.ndarray, seed: int = 0, knn: int = 15, t: int = 30):
    try:
        import phate
    except Exception:
        return None

    X = np.asarray(X, dtype=np.float64)
    op = phate.PHATE(
        n_components=2,
        knn=knn,
        random_state=seed,
        verbose=False,
    )
    Z = op.fit_transform(X)
    return np.asarray(Z, dtype=np.float32)


def replace_energy_with_linear_ramp(dataset, normalize=True):
    """
    Replace g.energy with a simple linear signal based on frame index.
    """
    N = len(dataset)
    energies = torch.linspace(0.0, 1.0, steps=N)

    if normalize:
        energies = (energies - energies.mean()) / energies.std()

    for i, g in enumerate(dataset):
        g.energy = energies[i].clone().detach()

    print("[energy] Replaced energies with linear ramp")


def plot_embedding(embed_2d: np.ndarray, color: np.ndarray, out_path: str, title: str):
    import matplotlib.pyplot as plt

    embed_2d = np.asarray(embed_2d)
    color = np.asarray(color).reshape(-1)

    plt.figure()
    sc = plt.scatter(embed_2d[:, 0], embed_2d[:, 1], c=color, s = 8, alpha = 0.25)
    plt.colorbar(sc, label="Energy")
    plt.title(title)
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()