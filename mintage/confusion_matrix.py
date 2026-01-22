#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Confusion Matrix Comparisons

Compares clustering results against Richardson clusters:
1. Other lab's high-res clusters vs Richardson clusters
2. Our low-res clusters vs Richardson clusters
"""

import os
import sys
import pickle
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Add utils to path for Suite_class (needed for unpickling)
sys.path.insert(0, str(Path(__file__).resolve().parent / 'utils'))

# ====================================================
# PATHS
# ====================================================
BASE_DIR = Path(__file__).resolve().parent

# Other lab's high-res clustering results
OTHER_LAB_HIGH_RES_PATH = BASE_DIR / "postclustering_results" / "RNA_clusters.pickle"

# Our low-res clustering results
OUR_LOW_RES_PATH = BASE_DIR / "postclustering_results" / "average_linkage_scaling_updated.pkl"

# Richardson clusters from Suite data
RICHARDSON_DATA_PATH = Path("/Users/kaisardauletbek/Documents/GitHub/RNA-Classification/data/RNA_suites_with_classes.pickle")

# Index mapping: original (low-res) indices -> pruned (Richardson) indices
INDEX_MAPPING_PATH = Path("/Users/kaisardauletbek/Documents/GitHub/RNA-Classification/data/original_to_pruned_mapping.pkl")

# Output directories
output_dir_highres = BASE_DIR / "confusion_outputs_highres_vs_richardson"
output_dir_lowres = BASE_DIR / "confusion_outputs_lowres_vs_richardson"
os.makedirs(output_dir_highres, exist_ok=True)
os.makedirs(output_dir_lowres, exist_ok=True)

PUCKER_ORDER = ["c2c2", "c2c3", "c3c2", "c3c3"]


# ====================================================
# HELPERS
# ====================================================
def load_richardson_clusters(path):
    """
    Load Richardson cluster assignments from Suite data.

    Returns dict: {'clusters': [arrays], 'class_names': [names]}
    """
    try:
        with open(path, "rb") as f:
            suites = pickle.load(f)
    except Exception as exc:
        print(f"[ERROR] Could not load Richardson data from {path}: {exc}")
        return None

    # Group suites by Richardson class
    class_indices = defaultdict(list)

    for i, suite in enumerate(suites):
        richardson_class = getattr(suite, 'richardson_class', None)
        if richardson_class is None or richardson_class == '!!':
            continue
        class_indices[richardson_class].append(i)

    # Convert to cluster format
    clusters = []
    class_names = []
    for class_name in sorted(class_indices.keys()):
        indices = class_indices[class_name]
        if len(indices) > 0:
            clusters.append(np.array(indices, dtype=np.intp))
            class_names.append(class_name)

    return {'clusters': clusters, 'class_names': class_names}


def load_other_lab_highres(path):
    """Load other lab's high-res clustering results."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    # Format: dict {pucker: [list of index arrays]}
    return {k: [np.asarray(c, dtype=np.intp) for c in v] for k, v in data.items()}


def load_index_mapping(path):
    """Load the original -> pruned index mapping."""
    if not path.exists():
        print(f"[WARN] Index mapping not found at {path}")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_our_lowres(path, index_mapping=None):
    """Load our low-res clustering results, optionally converting indices."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    # Format: list of dicts with 'name' and 'mode_clusters'
    result = {}
    for item in data:
        name = item['name']
        clusters = []
        for c in item['mode_clusters']:
            if index_mapping is not None:
                # Convert original indices to pruned indices
                converted = [index_mapping[idx] for idx in c if idx in index_mapping]
                clusters.append(np.asarray(converted, dtype=np.intp))
            else:
                clusters.append(np.asarray(c, dtype=np.intp))
        result[name] = clusters
    return result


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


def styled_heatmap(cm, title, out_png, row_labels=None, col_labels=None,
                   xlabel="Richardson clusters", ylabel="Clusters", vmax=12):
    """Plot a styled heatmap with fixed color scale."""
    n_rows, n_cols = cm.shape

    # Determine figure size
    fig_width = max(6, min(18, n_cols * 0.6 + 3))
    fig_height = max(4, min(14, n_rows * 0.5 + 2))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    sns.heatmap(
        cm, ax=ax, cmap="YlGnBu", vmin=0, vmax=vmax,
        annot=True, fmt="d", cbar=True,
        linewidths=0.5, linecolor="black",
        annot_kws={"fontsize": 7, "fontweight": "bold", "color": "black"},
    )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, pad=12)

    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_yticks(np.arange(n_rows) + 0.5)

    if col_labels is not None:
        ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=8)
    else:
        ax.set_xticklabels([f"R{i+1}" for i in range(n_cols)], rotation=0)

    if row_labels is not None:
        ax.set_yticklabels(row_labels, rotation=0, fontsize=8)
    else:
        ax.set_yticklabels([f"C{i+1}" for i in range(n_rows)], rotation=0)

    ax.set_facecolor("#fff8dc")
    cbar = ax.collections[0].colorbar
    cbar.set_label("Count", rotation=270, labelpad=14)

    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def filter_nonempty_columns(cm, col_labels):
    """
    Remove columns (Richardson clusters) that have no intersection with any row cluster.

    Returns filtered confusion matrix and corresponding labels.
    """
    # Find columns with at least one non-zero entry
    col_sums = cm.sum(axis=0)
    nonempty_cols = col_sums > 0

    filtered_cm = cm[:, nonempty_cols]
    filtered_labels = [col_labels[i] for i in range(len(col_labels)) if nonempty_cols[i]]

    return filtered_cm, filtered_labels


def compare_clusters_vs_richardson(clusters_by_pucker, richardson_clusters,
                                    output_dir, comparison_name):
    """
    Compare each pucker's clusters against ALL Richardson clusters.
    """
    print(f"\n{'='*70}")
    print(f"COMPARISON: {comparison_name} vs RICHARDSON CLUSTERS")
    print(f"{'='*70}")

    rich_clusters = richardson_clusters['clusters']
    rich_names = richardson_clusters['class_names']
    print(f"Richardson clusters: {len(rich_clusters)} classes")

    summary = []

    for pucker in PUCKER_ORDER:
        if pucker not in clusters_by_pucker:
            print(f"\n[WARN] Missing {comparison_name} clusters for {pucker}")
            continue

        print(f"\n{'-'*70}")
        print(f"Processing pucker: {pucker}")

        our_clusters = clusters_by_pucker[pucker]
        print(f"  {comparison_name} clusters: {len(our_clusters)}")
        print(f"  {comparison_name} cluster sizes: {[len(c) for c in our_clusters]}")

        # Confusion matrix against ALL Richardson classes
        conf_matrix = create_confusion_matrix(our_clusters, rich_clusters)

        # Metrics
        all_ours = np.concatenate(our_clusters) if sum(map(len, our_clusters)) else np.array([], int)
        all_rich = np.concatenate(rich_clusters) if sum(map(len, rich_clusters)) else np.array([], int)
        common = np.intersect1d(all_ours, all_rich)

        ari = nmi = np.nan
        if common.size:
            n_items = int(max(np.max(all_ours), np.max(all_rich))) + 1
            labels1 = create_label_vector(our_clusters, n_items)
            labels2 = create_label_vector(rich_clusters, n_items)
            ari = adjusted_rand_score(labels1[common], labels2[common])
            nmi = normalized_mutual_info_score(labels1[common], labels2[common])
            print(f"  Common indices: {common.size}, ARI: {ari:.4f}, NMI: {nmi:.4f}")
        else:
            print(f"  [INFO] No overlapping indices")

        # Save CSV
        csv_path = output_dir / f"confusion_{pucker}.csv"
        np.savetxt(csv_path, conf_matrix, fmt="%d", delimiter=",")

        # Filter to only show Richardson clusters with intersections
        filtered_cm, filtered_labels = filter_nonempty_columns(conf_matrix, rich_names)

        if filtered_cm.size > 0 and filtered_cm.shape[1] > 0:
            print(f"  Richardson classes with intersections: {filtered_labels}")

            png_path = output_dir / f"confusion_{pucker}.png"
            styled_heatmap(
                filtered_cm,
                title=f"{pucker.upper()}: {comparison_name} vs Richardson",
                out_png=png_path,
                row_labels=[f"C{i+1}" for i in range(len(our_clusters))],
                col_labels=filtered_labels,
                xlabel="Richardson clusters",
                ylabel=f"{comparison_name} clusters",
                vmax=12,
            )
            print(f"  Saved: {png_path}")

        summary.append((pucker, len(our_clusters), common.size, ari, nmi))

    # Summary table
    print(f"\n{'-'*70}")
    print(f"Summary ({comparison_name} vs Richardson):")
    print(f"{'Pucker':<8} | {'N_ours':>6} | {'Common':>6} | {'ARI':>7} | {'NMI':>7}")
    print("-" * 50)
    for pucker, n_ours, n_common, ari, nmi in summary:
        ari_s = "nan" if np.isnan(ari) else f"{ari:.4f}"
        nmi_s = "nan" if np.isnan(nmi) else f"{nmi:.4f}"
        print(f"{pucker:<8} | {n_ours:>6} | {n_common:>6} | {ari_s:>7} | {nmi_s:>7}")

    return summary


# ====================================================
# MAIN
# ====================================================
def main():
    print("Loading Richardson clusters...")
    richardson_clusters = load_richardson_clusters(RICHARDSON_DATA_PATH)
    if richardson_clusters is None:
        print("[ERROR] Failed to load Richardson data. Exiting.")
        return

    total = sum(len(c) for c in richardson_clusters['clusters'])
    print(f"Richardson: {len(richardson_clusters['clusters'])} classes, {total} total suites")

    # Load other lab's high-res results
    print(f"\nLoading other lab high-res results from: {OTHER_LAB_HIGH_RES_PATH}")
    other_lab_clusters = load_other_lab_highres(OTHER_LAB_HIGH_RES_PATH)

    # Load index mapping for low-res data
    print(f"\nLoading index mapping from: {INDEX_MAPPING_PATH}")
    index_mapping = load_index_mapping(INDEX_MAPPING_PATH)
    if index_mapping:
        print(f"  Mapping size: {len(index_mapping)} indices")

    # Load our low-res results (with index conversion)
    print(f"Loading our low-res results from: {OUR_LOW_RES_PATH}")
    our_lowres_clusters = load_our_lowres(OUR_LOW_RES_PATH, index_mapping)

    # Comparison 1: Other lab high-res vs Richardson
    compare_clusters_vs_richardson(
        other_lab_clusters,
        richardson_clusters,
        output_dir_highres,
        "High-Res"
    )

    # Comparison 2: Our low-res vs Richardson
    compare_clusters_vs_richardson(
        our_lowres_clusters,
        richardson_clusters,
        output_dir_lowres,
        "Low-Res"
    )

    print(f"\n{'='*70}")
    print(f"Results saved to:")
    print(f"  High-Res: {output_dir_highres}")
    print(f"  Low-Res: {output_dir_lowres}")


if __name__ == "__main__":
    main()
