#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")  # safe for non-interactive script runs
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# ====================================================
# PATHS (edit low_res_path if needed)
# ====================================================
low_res_path = os.path.join("postclustering_results", "no_cluster_sep.pkl")
output_dir = "confusion_outputs_styled"
os.makedirs(output_dir, exist_ok=True)

# Explicit high-res paths by conformation
HIGH_RES_PATHS = {
    "c2c2": "/Users/kaisardauletbek/Downloads/mintage-code-dihedrals/out/saved_suite_lists/cluster_indices_mode_c2c2_qfold_0.05.pickle",
    "c2c3": "/Users/kaisardauletbek/Downloads/mintage-code-dihedrals/out/saved_suite_lists/cluster_indices_mode_c2c3_qfold_0.07.pickle",
    "c3c2": "/Users/kaisardauletbek/Downloads/mintage-code-dihedrals/out/saved_suite_lists/cluster_indices_mode_c3c2_qfold_0.05.pickle",
    "c3c3": "/Users/kaisardauletbek/Downloads/mintage-code-dihedrals/out/saved_suite_lists/cluster_indices_mode_c3c3_qfold_0.09.pickle",
}
# HIGH_RES_PATHS = {
#     "c2c2": "/Users/kaisardauletbek/Downloads/mintage-code-dihedrals/out/saved_suite_lists/cluster_indices_c2c2_qfold_0.05.pickle",
#     "c2c3": "/Users/kaisardauletbek/Downloads/mintage-code-dihedrals/out/saved_suite_lists/cluster_indices_c2c3_qfold_0.07.pickle",
#     "c3c2": "/Users/kaisardauletbek/Downloads/mintage-code-dihedrals/out/saved_suite_lists/cluster_indices_c3c2_qfold_0.05.pickle",
#     "c3c3": "/Users/kaisardauletbek/Downloads/mintage-code-dihedrals/out/saved_suite_lists/cluster_indices_c3c3_qfold_0.09.pickle",
# }

# ====================================================
# HELPERS
# ====================================================
def create_confusion_matrix(clustering1, clustering2):
    """Compute (i,j) = |cluster1[i] ∩ cluster2[j]|."""
    n1, n2 = len(clustering1), len(clustering2)
    cm = np.zeros((n1, n2), dtype=int)
    for i, c1 in enumerate(clustering1):
        for j, c2 in enumerate(clustering2):
            if len(c1) and len(c2):
                cm[i, j] = np.intersect1d(c1, c2).size
    return cm

def create_label_vector(clustering, n_items):
    """Convert clustering into label vector."""
    labels = np.full(n_items, -1, dtype=int)
    for k, cl in enumerate(clustering):
        labels[cl] = k
    return labels

def styled_heatmap_counts(cm, row_clusters, col_clusters, title, out_png):
    """
    Plot a heatmap styled like the provided example:
    - YlGnBu colormap (light yellow -> dark blue)
    - Black gridlines, bold black annotations
    - No percentages, counts only
    """
    vmax = int(cm.max()) or 1

    fig, ax = plt.subplots(figsize=(10, 10))

    # Seaborn heatmap with explicit gridlines
    sns.heatmap(
        cm,
        ax=ax,
        cmap="YlGnBu",
        vmin=0,
        vmax=vmax,
        annot=True,
        fmt="d",
        cbar=True,
        linewidths=1.0,
        linecolor="black",
        square=False,  # similar look; keep cells rectangular like the example
        annot_kws={"fontsize": 9, "fontweight": "bold", "color": "black"},
    )

    # Axis labels and ticks (compact, like your example)
    ax.set_xlabel("High-res clusters", fontsize=12)
    ax.set_ylabel("Low-res clusters", fontsize=12)
    ax.set_title(title, fontsize=14, pad=12)

    # Tick labels: C1.. with sizes, but short to keep it tidy
    ax.set_xticks(np.arange(len(col_clusters)) + 0.5)
    ax.set_yticks(np.arange(len(row_clusters)) + 0.5)
    ax.set_xticklabels([f"C{i+1}" for i in range(len(col_clusters))], rotation=0)
    ax.set_yticklabels([f"{i+1}" for i in range(len(row_clusters))], rotation=0)

    # Make the background light (close to the example's feel)
    ax.set_facecolor("#fff8dc")  # cornsilk-like light yellow

    # Colorbar label simple
    cbar = ax.collections[0].colorbar
    cbar.set_label("Count", rotation=270, labelpad=14)

    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ====================================================
# MAIN
# ====================================================
def main():
    print(f"Loading low-res results from: {low_res_path}")
    with open(low_res_path, "rb") as f:
        result_low_res = pickle.load(f)

    if not isinstance(result_low_res, list):
        raise ValueError("Expected a list of dicts in result_low_res")

    summary = []

    for entry in result_low_res:
        if not isinstance(entry, dict) or "name" not in entry or "mode_clusters" not in entry:
            continue

        name = str(entry["name"]).strip()
        if name not in HIGH_RES_PATHS:
            print(f"\n[WARN] No high-res path registered for '{name}', skipping.")
            continue

        high_path = HIGH_RES_PATHS[name]
        print("\n" + "=" * 70)
        print(f"Processing pucker: {name}")
        print(f"→ High-res file: {high_path}")

        if not os.path.exists(high_path):
            print(f"[WARN] High-res file not found for {name}, skipping.")
            continue

        with open(high_path, "rb") as f:
            high_clusters_raw = pickle.load(f)
        high_clusters = [np.asarray(c).astype(int) for c in high_clusters_raw]
        low_clusters = [np.asarray(c).astype(int) for c in entry["mode_clusters"]]

        # Confusion matrix
        conf_matrix = create_confusion_matrix(low_clusters, high_clusters)
        np.set_printoptions(linewidth=200)
        print("Confusion Matrix:\n", conf_matrix)

        # Metrics (still printed for reference)
        all_low = np.concatenate(low_clusters) if sum(map(len, low_clusters)) else np.array([], int)
        all_high = np.concatenate(high_clusters) if sum(map(len, high_clusters)) else np.array([], int)
        common = np.intersect1d(all_low, all_high)
        ari = nmi = np.nan
        if common.size:
            n_items = int(max(np.max(all_low), np.max(all_high))) + 1
            labels1 = create_label_vector(low_clusters, n_items)
            labels2 = create_label_vector(high_clusters, n_items)
            ari = adjusted_rand_score(labels1[common], labels2[common])
            nmi = normalized_mutual_info_score(labels1[common], labels2[common])
            print(f"ARI: {ari:.4f}, NMI: {nmi:.4f}")
        else:
            print("[INFO] No overlapping indices between low and high clusters.")

        # Save CSV
        csv_path = os.path.join(output_dir, f"confusion_{name}.csv")
        np.savetxt(csv_path, conf_matrix, fmt="%d", delimiter=",")
        print(f"Saved CSV: {csv_path}")

        # Styled heatmap PNG (counts only)
        png_path = os.path.join(output_dir, f"confusion_{name}.png")
        styled_heatmap_counts(
            conf_matrix,
            low_clusters,
            high_clusters,
            title=f"{name.upper()}",
            out_png=png_path,
        )
        print(f"Saved PNG: {png_path}")

        summary.append((name, len(low_clusters), len(high_clusters), common.size, ari, nmi))

    # Summary table
    print("\n" + "=" * 70)
    print("Summary across puckers:")
    print("(pucker | n_low | n_high | common | ARI | NMI)")
    for name, n_low, n_high, n_common, ari, nmi in summary:
        ari_s = "nan" if np.isnan(ari) else f"{ari:.4f}"
        nmi_s = "nan" if np.isnan(nmi) else f"{nmi:.4f}"
        print(f"{name:6s} | {n_low:5d} | {n_high:6d} | {n_common:6d} | {ari_s:>5s} | {nmi_s:>5s}")

if __name__ == "__main__":
    main()
