import numpy as np
from scipy.linalg import sqrtm
import ot



def _sym(A: np.ndarray) -> np.ndarray:
    """Force symmetry."""
    return 0.5 * (A + A.T)

def _sqrtm_sym_psd(A, eps=1e-12):
    A = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(A)
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w + eps)) @ V.T

from scipy.spatial.distance import jensenshannon

def jsd_empirical(Z1, Z2, bins=100):
    Z1 = np.asarray(Z1, dtype=float).ravel()
    Z2 = np.asarray(Z2, dtype=float).ravel()

    # Common histogram range
    zmin = min(Z1.min(), Z2.min())
    zmax = max(Z1.max(), Z2.max())

    p, edges = np.histogram(Z1, bins=bins, range=(zmin, zmax), density=True)
    q, _     = np.histogram(Z2, bins=bins, range=(zmin, zmax), density=True)

    eps = 1e-12
    p = p + eps
    q = q + eps
    p /= p.sum()
    q /= q.sum()

    return jensenshannon(p, q)

# w2 computations

def w2_emperical_sq(Z1: np.ndarray, Z2: np.ndarray):

    Z1 = np.asarray(Z1, dtype=float)
    Z2 = np.asarray(Z2, dtype=float)

    n1 = Z1.shape[0]
    n2 = Z2.shape[0]

    # Uniform weights
    a = np.ones(n1) / n1
    b = np.ones(n2) / n2

    # Squared Euclidean cost matrix
    M = ot.dist(Z1, Z2, metric="euclidean") ** 2

    # Solve OT: returns W2^2
    W2_sq = ot.emd2(a, b, M)

    return W2_sq

def w2_gaussian_sq(mu1: np.ndarray, S1: np.ndarray, mu2: np.ndarray, S2: np.ndarray) -> tuple[float, float, float]:
    mu1 = np.asarray(mu1, dtype=float)
    mu2 = np.asarray(mu2, dtype=float)
    S1 = _sym(np.asarray(S1, dtype=float))
    S2 = _sym(np.asarray(S2, dtype=float))

    translation_sq = float(np.sum((mu1 - mu2) ** 2))

    # variance term: Tr(S1 + S2 - 2 * (S1^{1/2} S2 S1^{1/2})^{1/2})
    S1_sqrt = _sqrtm_sym_psd(S1)
    middle = _sym(S1_sqrt @ S2 @ S1_sqrt)
    middle_sqrt = _sqrtm_sym_psd(middle)

    variance_sq = float(np.trace(S1 + S2 - 2.0 * middle_sqrt))

    if variance_sq < 0 and variance_sq > -1e-8:
        variance_sq = 0.0
    elif variance_sq < -1e-8:
        raise ValueError(f"Variance term became negative ({variance_sq}); check covariances/data.")

    return translation_sq + variance_sq, translation_sq, variance_sq


# metrics

from protscape.featureize import load_aligned_trajectories, featurize_coordinates, featurize_dihedrals
from sklearn.decomposition import PCA

# import MDAnalysis as mda

# def subsampled_rmwd(
#     top: str,
#     traj: str,
#     point_sel: str = "protein and name CA",
#     subsampling: int = 100,  # number of frames to randomly select for RMWD computation
#     trials: int = 10,
# ):  
#     u = mda.Universe(top, traj)
#     X_ref = featurize_coordinates(u, sel=point_sel)
#     N = X_ref.shape[1]

    
#     results = np.zeros(trials)
#     for trial in range(trials):

#         frame_indices = np.random.choice(X_ref.shape[0], size=subsampling, replace=False)
#         X_sub = X_ref[frame_indices]  # (subsampling, N, 3)

#         trans_terms = np.zeros(N, dtype=np.float64)
#         var_terms = np.zeros(N, dtype=np.float64)

#         for i in range(N):
#             A = X_sub[:, i, :]  # (subsampling, 3)
#             B = X_ref[:, i, :]  # (full, 3)

#             mu1 = A.mean(axis=0)
#             mu2 = B.mean(axis=0)

#             S1 = np.cov(A.T, bias=False)
#             S2 = np.cov(B.T, bias=False)

#             w2_sq, dmu2, vterm = w2_gaussian_sq(mu1, S1, mu2, S2)

#             trans_terms[i] = dmu2
#             var_terms[i] = vterm

#         translation_sq = float(np.mean(trans_terms))
#         variance_sq = float(np.mean(var_terms))
#         rmwd_sq = translation_sq + variance_sq

#         results[trial] = np.sqrt(rmwd_sq)

#     return {
#         "rmwd_mean": float(np.mean(results)),
#         "rmwd_std": float(np.std(results)),
#         "trials": trials,
#         "subsampling": subsampling,
#     }
   

    
def compute_rmwd(
    top1: str,
    traj1: str,
    top2: str,
    traj2: str,
    align_to: str | None = None,
    align_sel: str = "protein and name CA",
    point_sel: str = "protein and name CA",

):

    u1, u2 = load_aligned_trajectories(top1, traj1, top2, traj2, align_to, align_sel)
    X1 = featurize_coordinates(u1, sel=point_sel)
    X2 = featurize_coordinates(u2, sel=point_sel)  # shape: (f, N, 3)
    N = X1.shape[1] # n residues

    # print(X1.shape, X2.shape)

    assert X2.shape[1] == N, "Number of points must match between trajectories"

    trans_terms = np.zeros(N, dtype=np.float64)
    var_terms = np.zeros(N, dtype=np.float64)
    for i in range(N):
        A = X1[:, i, :]  # (m,3)
        B = X2[:, i, :]  # (m,3)

        mu1 = A.mean(axis=0)
        mu2 = B.mean(axis=0)

        S1 = np.cov(A.T, bias=False)
        S2 = np.cov(B.T, bias=False)

        w2_sq, dmu2, vterm = w2_gaussian_sq(mu1, S1, mu2, S2)

        trans_terms[i] = dmu2
        var_terms[i] = vterm


    translation_sq = float(np.mean(trans_terms))
    variance_sq = float(np.mean(var_terms))
    rmwd_sq = translation_sq + variance_sq

    return {
        "rmwd": float(np.sqrt(rmwd_sq)),
        "translation": float(np.sqrt(translation_sq)),
        "variance": float(np.sqrt(variance_sq)),
        "rmwd_sq": rmwd_sq,
        "translation_sq": translation_sq,
        "variance_sq": variance_sq,
        # "per_residue": per_res,
        "n_residues": N,
    }

def _unsigned_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(abs(np.dot(a, b) / (na * nb)))

def compute_pca_dist(
    top1: str, traj1: str,
    top2: str, traj2: str,
    mode: str = "ca",                 # "ca" or "dihedral"
    point_sel: str = "protein and name CA",
    dihedral_embed: str = "sincos",   # "sincos" or "raw"
    pca_mode: str = "ref",           # "pool" or "ref" (ref = traj1)
    k: int = 10,
    seed: int = 0,
    align_to: str | None = None,
    align_sel: str = "protein and name CA",
    metric: str = "gaussian_w2",
):
    u1, u2 = load_aligned_trajectories(top1, traj1, top2, traj2, align_to, align_sel)

    # Featurize
    if mode == "ca":
        X1 = featurize_coordinates(u1, sel=point_sel)
        X2 = featurize_coordinates(u2, sel=point_sel)

        X1 = X1.reshape(X1.shape[0], -1)  # (f, N*3)
        X2 = X2.reshape(X2.shape[0], -1)
    elif mode == "dihedral":
        X1 = featurize_dihedrals(u1, embed=dihedral_embed)
        X2 = featurize_dihedrals(u2, embed=dihedral_embed)

        X1 = X1.reshape(X1.shape[0], -1)  # (f, N*4) if sincos
        X2 = X2.reshape(X2.shape[0], -1)
    else:
        raise ValueError("mode must be 'ca' or 'dihedral'")
    
    # print(X1.shape, X2.shape)

    # Fit PCA
    if pca_mode == "ref":
        pca = PCA(n_components=min(k, X1.shape[1]), svd_solver="auto", random_state=seed)
        pca.fit(X1)
    elif pca_mode == "pool":
        Xpool = np.vstack([X1, X2])
        pca = PCA(n_components=min(k, Xpool.shape[1]), svd_solver="auto", random_state=seed)
        pca.fit(Xpool)
    else:
        raise ValueError("pca_mode must be 'ref' or 'pool'")

    # Project into first k components
    Z1 = pca.transform(X1)[:, :k]
    Z2 = pca.transform(X2)[:, :k]

    mu1 = Z1.mean(axis=0)
    mu2 = Z2.mean(axis=0)
    S1 = np.cov(Z1.T, bias=False) if Z1.shape[0] > 1 else np.eye(k)
    S2 = np.cov(Z2.T, bias=False) if Z2.shape[0] > 1 else np.eye(k)

    # print(S1, S2)

    out = {
        "pca": pca,
        "Z1": Z1,
        "Z2": Z2,
        "mu1": mu1,
        "mu2": mu2,
        "S1": S1,
        "S2": S2,
        "k": k,
        "mode": mode,
        "pca_mode": pca_mode,
        "metric": metric,
        "n_frames_used_1": Z1.shape[0],
        "n_frames_used_2": Z2.shape[0],
    }

    if metric == "gaussian_w2":
        w2_sq, _, _ = w2_gaussian_sq(mu1, S1, mu2, S2)
        w2 = float(np.sqrt(w2_sq))

        out['w2_sq'] = w2_sq
        out['w2'] = w2
    elif metric == "empirical_w2":

        w2_sq = w2_emperical_sq(Z1, Z2)
        w2 = float(np.sqrt(w2_sq))

        out['w2_sq'] = w2_sq
        out['w2'] = w2

    elif metric == "jsd":
        
        jsd = jsd_empirical(Z1, Z2)
        out['jsd'] = jsd

    else:

        raise ValueError("metric must be 'gaussian_w2', 'empirical_w2', or 'jsd'")

    return out

from MDAnalysis.analysis import rms

def compute_rmsf_corr(top1: str, 
                      traj1: str, 
                      top2: str, 
                      traj2: str,
                      align_to: str | None = None,
                      align_sel: str = "protein and name CA", 
                      point_sel: str = "protein and name CA"):
    
    u1, u2 = load_aligned_trajectories(top1, traj1, top2, traj2, align_to, align_sel)

    rmsf1 = rms.RMSF(u1.select_atoms(point_sel)).run().rmsf
    rmsf2 = rms.RMSF(u2.select_atoms(point_sel)).run().rmsf
    corr = np.corrcoef(rmsf1, rmsf2)[0, 1]

    return {
        "corr": corr,
        "corr_sq": corr ** 2,
        "rmsf1": rmsf1.tolist(),
        "rmsf2": rmsf2.tolist(),
    }


import numpy as np
from sklearn.decomposition import PCA



# def compute_pc_similarity(
#     top1: str,
#     traj1: str,
#     top2: str,
#     traj2: str,
#     mode: str = "ca",                 # "ca" or "dihedral"
#     point_sel: str = "protein and name CA",
#     dihedral_embed: str = "sincos",
#     align_to: str | None = None,
#     align_sel: str = "protein and name CA",
#     threshold: float = 0.5,
#     seed: int = 0,
# ):

#     u1, u2 = load_aligned_trajectories(top1, traj1, top2, traj2, align_to, align_sel)


#     if mode == "ca":
#         X1 = featurize_coordinates(u1, sel=point_sel)
#         X2 = featurize_coordinates(u2, sel=point_sel)

#         X1 = X1.reshape(X1.shape[0], -1)
#         X2 = X2.reshape(X2.shape[0], -1)

#     elif mode == "dihedral":
#         X1 = featurize_dihedrals(u1, embed=dihedral_embed)
#         X2 = featurize_dihedrals(u2, embed=dihedral_embed)

#         X1 = X1.reshape(X1.shape[0], -1)
#         X2 = X2.reshape(X2.shape[0], -1)

#     else:
#         raise ValueError("mode must be 'ca' or 'dihedral'")

#     # ---- Max possible PCs (rank limit) ----
#     m = min(
#         X1.shape[0] - 1,
#         X1.shape[1],
#         X2.shape[0] - 1,
#         X2.shape[1],
#     )

#     if m <= 0:
#         raise ValueError("Not enough frames for PCA.")

#     # ---- PCA on each ensemble ----
#     p1 = PCA(n_components=m, random_state=seed).fit(X1)
#     p2 = PCA(n_components=m, random_state=seed).fit(X2)

#     # ---- unsigned cosine similarity ----
#     def unsigned_cos(a, b):
#         return abs(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

#     pc_cos_sims = np.array([
#         unsigned_cos(p1.components_[i], p2.components_[i])
#         for i in range(m)
#     ])

#     pc_success_mask = pc_cos_sims > threshold
#     pc_sim_fraction = float(pc_success_mask.mean())
#     pc_sim_percent = 100.0 * pc_sim_fraction

#     return {
#         "pc_cos_sims": pc_cos_sims,
#         "pc_success_mask": pc_success_mask,
#         "pc_sim_fraction": pc_sim_fraction,
#         "pc_sim_percent": pc_sim_percent,
#         "m": m,
#         "threshold": threshold,
#     }


# viz

def _draw_cov_ellipse(ax, mean, cov, n_std=1.0, **kwargs):
    """
    Draw n-sigma covariance ellipse (2D).
    """
    from matplotlib.patches import Ellipse

    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)

    ell = Ellipse(xy=mean, width=width, height=height, angle=angle,
                  fill=False, linewidth=2, **kwargs)
    ax.add_patch(ell)

import matplotlib.pyplot as plt

def plot_pca_space(result, alpha=0.4, s=10, show_gaussian=True):

    Z1 = result["Z1"]
    Z2 = result["Z2"]
    mu1 = result["mu1"]
    mu2 = result["mu2"]
    S1 = result["S1"]
    S2 = result["S2"]

    if Z1.shape[1] < 2:
        raise ValueError("Need at least k>=2 PCs to plot.")

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(Z1[:, 0], Z1[:, 1], s=s, alpha=alpha, label="Ref")
    ax.scatter(Z2[:, 0], Z2[:, 1], s=s, alpha=alpha, label="Pred")

    ax.scatter(mu1[0], mu1[1], marker="x", s=80, label="Ref mean")
    ax.scatter(mu2[0], mu2[1], marker="x", s=80, label="Pred mean")

    if show_gaussian:
        _draw_cov_ellipse(ax, mu1[:2], S1[:2, :2])
        _draw_cov_ellipse(ax, mu2[:2], S2[:2, :2])

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"PCA space (W2 = {result['w2']:.3f})")
    ax.legend()

    fig.tight_layout()

    return fig


import numpy as np
import matplotlib.pyplot as plt


def plot_rmsf_overlay(
    result: dict,
    linewidth: float = 2.5,
    alpha: float = 0.75,
):
    rmsf1 = np.asarray(result["rmsf1"], dtype=float)
    rmsf2 = np.asarray(result["rmsf2"], dtype=float)

    if rmsf1.shape != rmsf2.shape:
        raise ValueError(
            f"rmsf1 and rmsf2 must have the same shape, got {rmsf1.shape} vs {rmsf2.shape}"
        )

    n = len(rmsf1)
    x = np.arange(n) + 1


    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    ax.plot(
        x, rmsf1,
        color="tab:blue",
        label="Ref",
        marker="o",
        lw=linewidth,
        alpha=alpha,
    )
    ax.plot(
        x, rmsf2,
        color="tab:orange",
        marker="o",
        label="Pred",
        lw=linewidth,
        alpha=alpha,
    )

    corr = result["corr_sq"]
    ax.set_title(f"RMSF Pearson $r^2$ = {corr:.3f}")

    ax.set_xlabel("Residue index")
    ax.set_ylabel("RMSF ($\\AA$)")
    ax.legend()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    return fig