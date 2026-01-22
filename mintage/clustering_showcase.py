"""
Toy example: 4 clusters with very different densities (1 diffuse + 3 tight),
and compare:

  - single linkage (one global distance cut)
  - average linkage (same cut)
  - AGE (Adaptive Linkage Clustering) pre-clustering

Requirements: numpy, matplotlib, scipy

Run:
  python demo_age_vs_linkage.py
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, fcluster, to_tree, dendrogram
from scipy.spatial.distance import pdist


# ---------------------------
# Data
# ---------------------------
def make_dataset(seed: int = 7):
    rng = np.random.default_rng(seed)

    # Diffuse cluster (hard case for a single global cut)
    n0, mu0, sig0 = 230, (-10.0, 5.0), 2.7
    X0 = rng.normal(loc=mu0, scale=sig0, size=(n0, 2))

    # Three compact clusters
    n1, mu1, sig1 = 65, (0.0, 5.0), 0.35
    n2, mu2, sig2 = 60, (0.0, -1.0), 0.22
    n3, mu3, sig3 = 55, (1.0, 0.0), 0.22

    X1 = rng.normal(loc=mu1, scale=sig1, size=(n1, 2))
    X2 = rng.normal(loc=mu2, scale=sig2, size=(n2, 2))
    X3 = rng.normal(loc=mu3, scale=sig3, size=(n3, 2))

    X = np.vstack([X0, X1, X2, X3])
    y_true = np.array([0] * n0 + [1] * n1 + [2] * n2 + [3] * n3, dtype=int)
    return X, y_true


def choose_threshold_from_compact_clusters(X: np.ndarray, y_true: np.ndarray) -> float:
    """
    Pick a single cut height t that keeps the compact clusters intact.
    (Then it's expected to split the diffuse cluster.)
    """
    qs = []
    for k in [1, 2, 3]:
        Xk = X[y_true == k]
        dk = pdist(Xk)
        qs.append(np.quantile(dk, 0.95))
    return float(1.15 * max(qs))


# ---------------------------
# AGE implementation (Algorithm 1)
# ---------------------------
def age_clustering(
    X: np.ndarray,
    *,
    method: str = "single",
    d_max: float,
    kappa: int = 40,
    q: float = 0.0,
):
    """
    Adaptive Linkage Clustering (AGE) as in Algorithm 1.

    Interpretation of Step 6(b):
      The paper describes that a split should occur only if the parent's node distance
      is "significant in relation to the greatest distance value of its child nodes".
      A standard way to encode this is:
          parent_dist > (1 + q) * max(child_left_dist, child_right_dist)
      so q=0 reduces to parent_dist > max(child_dist), which is typically true except for ties,
      and larger q makes splitting stricter.

    Returns:
      labels: array of length n with cluster ids 1..K and 0 for outliers
      outliers: boolean mask of outliers
    """
    n = X.shape[0]
    labels = np.zeros(n, dtype=int)
    outlier_mask = np.zeros(n, dtype=bool)

    # Remaining points (original indices)
    P = np.arange(n, dtype=int)

    clusters = []  # list of arrays of original indices

    def linkage_on_P(P_idx: np.ndarray):
        if len(P_idx) <= 1:
            return None
        return linkage(X[P_idx], method=method, metric="euclidean")

    # Step loop
    while len(P) > 0:
        if len(P) == 1:
            # Nothing to cluster, leftover becomes its own cluster
            clusters.append(P.copy())
            break

        # 1) linkage tree on P
        Z = linkage_on_P(P)

        # 2) cut at d_max, move small clusters to outliers
        flat = fcluster(Z, t=d_max, criterion="distance")
        # flat labels are 1..m for points in X[P]
        removed_any = False
        for lab in np.unique(flat):
            idx_local = np.where(flat == lab)[0]  # indices into P
            if idx_local.size < kappa:
                outlier_mask[P[idx_local]] = True
                removed_any = True

        if removed_any:
            P = P[~outlier_mask[P]]

        if len(P) == 0:
            break
        if len(P) == 1:
            clusters.append(P.copy())
            break

        # 3) linkage tree for new P
        Z2 = linkage_on_P(P)

        # 4) s_P
        sP = np.sqrt(len(P)) + (kappa / 2.0)

        # 5) candidate list L (as tuples of leaf indices in *local* numbering)
        L_local = []
        last_big_node = None

        # 6) traverse dendrogram trunk
        root, _ = to_tree(Z2, rd=True)

        node = root
        while node.left is not None and node.right is not None:
            left, right = node.left, node.right
            if left.count >= right.count:
                big, small = left, right
            else:
                big, small = right, left

            if small.count > sP:
                # Step 6(b): "significant" branching check
                child_max = max(left.dist, right.dist)
                if node.dist > (1.0 + q) * child_max:
                    # store local leaf indices for "small"
                    L_local.append(tuple(sorted(small.pre_order())))
                    last_big_node = big

            node = big  # follow the larger branch

        # 7) also add last "big" branch if we ever added a smaller one
        if last_big_node is not None:
            L_local.append(tuple(sorted(last_big_node.pre_order())))

        # 8) extract
        if len(L_local) == 0:
            # move all remaining P to clusters as one cluster
            clusters.append(P.copy())
            P = np.array([], dtype=int)
        else:
            # choose largest candidate
            cand = max(L_local, key=len)
            cand_local = np.array(cand, dtype=int)
            cand_global = P[cand_local]  # map local leaf ids -> original indices
            clusters.append(cand_global)

            # remove extracted points from P
            keep = np.ones(len(P), dtype=bool)
            keep[cand_local] = False
            P = P[keep]

    # assign final labels
    cid = 1
    for idx in clusters:
        labels[idx] = cid
        cid += 1
    labels[outlier_mask] = 0
    return labels, outlier_mask


# ---------------------------
# Plot helpers
# ---------------------------
def relabel_to_consecutive(labels: np.ndarray):
    """Map labels (0.., arbitrary) to consecutive (keeping 0 as outlier if present)."""
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    if 0 in uniq:
        nonzero = uniq[uniq != 0]
        mapping = {0: 0, **{u: i + 1 for i, u in enumerate(nonzero)}}
    else:
        mapping = {u: i + 1 for i, u in enumerate(uniq)}
    return np.array([mapping[u] for u in labels], dtype=int)


def scatter(ax, X, labels, title, outlier_mask=None):
    labels = relabel_to_consecutive(labels)
    if outlier_mask is None:
        outlier_mask = (labels == 0)

    # Plot outliers (if any)
    if np.any(outlier_mask):
        ax.scatter(
            X[outlier_mask, 0],
            X[outlier_mask, 1],
            s=28,
            marker="x",
            linewidths=1.8,
            label="outliers",
        )

    # Plot clusters 1..K
    cl = labels.copy()
    cl[outlier_mask] = 0
    K = cl.max()
    cmap = plt.get_cmap("tab10", max(K, 1))

    for k in range(1, K + 1):
        pts = X[cl == k]
        ax.scatter(pts[:, 0], pts[:, 1], s=16, color=cmap(k - 1), label=str(k))

    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    # ax.legend(loc="upper right", frameon=True)


# ---------------------------
# Main
# ---------------------------
def main():
    X, y_true = make_dataset(seed=7)

    # Use a single cut height for both linkages (same idea as your earlier plot)
    t = choose_threshold_from_compact_clusters(X, y_true)

    # Single linkage clustering
    Z_single = linkage(X, method="single", metric="euclidean")
    y_single = fcluster(Z_single, t=t, criterion="distance")

    # Average linkage clustering
    Z_avg = linkage(X, method="average", metric="euclidean")
    y_avg = fcluster(Z_avg, t=t, criterion="distance")

    # AGE (using single linkage by default, like the paper's Fig. 6 discussion)
    # d_max plays the role of the tree-cut for "too sparse" connections.
    # kappa = minimal cluster size (outlier threshold).
    y_age, out_age = age_clustering(X, method="single", d_max=t, kappa=40, q=0.0)

    # Scatter plots figure
    fig1, axes = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)
    scatter(axes[0, 0], X, y_true + 1, "original data (ground truth)")
    scatter(axes[0, 1], X, y_single, f"single linkage")
    scatter(axes[1, 0], X, y_avg, f"average linkage")
    scatter(
        axes[1, 1],
        X,
        y_age,
        f"AGE (single linkage)",
        outlier_mask=out_age,
    )

    # Single linkage dendrogram figure
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8), constrained_layout=True)
    dendrogram(
        Z_single,
        ax=ax2,
        color_threshold=t,
        above_threshold_color="gray",
    )
    ax2.axhline(y=t, color="r", linestyle="--", linewidth=1.5, label=f"cut at t={t:.2f}")
    ax2.set_title("Single linkage dendrogram")
    ax2.set_xlabel("Sample index")
    ax2.set_ylabel("Distance")
    ax2.legend()
    ax2.grid(True, alpha=0.25)

    # Average linkage dendrogram figure
    fig3, ax3 = plt.subplots(1, 1, figsize=(12, 8), constrained_layout=True)
    dendrogram(
        Z_avg,
        ax=ax3,
        color_threshold=t,
        above_threshold_color="gray",
    )
    ax3.axhline(y=t, color="r", linestyle="--", linewidth=1.5, label=f"cut at t={t:.2f}")
    ax3.set_title("Average linkage dendrogram")
    ax3.set_xlabel("Sample index")
    ax3.set_ylabel("Distance")
    ax3.legend()
    ax3.grid(True, alpha=0.25)

    plt.show()


if __name__ == "__main__":
    main()
