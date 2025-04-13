#!/usr/bin/env python3
"""
age.py

Implementation of AGE pre-clustering (Algorithm C.1):
 - Builds an average-linkage clustering tree from data in a metric space (here, Euclidean).
 - Adaptively cuts the dendrogram based on a maximal outlier distance, minimal cluster size,
   and a relative branching distance. 
 - Returns a list of pre-clusters and a separate list of outlier indices.

To use:
  from age import age_clustering

Main function:
  age_clustering(X, dmax, kappa, q)

References:
  Mardia et al. (2022), "MINT-AGE" algorithm (pre-clustering step).
"""

import numpy as np
import math
from scipy.cluster.hierarchy import linkage, fcluster

def build_tree_structure(Z, n):
    """
    Convert a SciPy linkage matrix (Z) to a dictionary-based tree representation.
    Leaves: 0..n-1, internal nodes: n..2n-2, root = 2n-2.

    :param Z: np.array of shape (n-1, 4) from scipy.cluster.hierarchy.linkage
    :param n: number of data points (leaves)
    :return: dict with node_id -> {'left', 'right', 'distance', 'count'}
    """
    tree = {}
    # Initialize leaves
    for i in range(n):
        tree[i] = {
            'left': None,
            'right': None,
            'distance': 0.0,
            'count': 1
        }
    # Build internal nodes
    current_id = n
    for row in Z:
        left_id, right_id = int(row[0]), int(row[1])
        dist = row[2]
        merged_count = int(row[3])
        tree[current_id] = {
            'left': left_id,
            'right': right_id,
            'distance': dist,
            'count': merged_count
        }
        current_id += 1
    return tree

def get_leaf_indices(tree, node_id):
    """
    Recursively retrieve all leaf indices under the given node_id.

    :param tree: the dictionary-based tree
    :param node_id: current node in the tree
    :return: list of leaf IDs (which correspond to original data indices)
    """
    node = tree[node_id]
    if node['left'] is None and node['right'] is None:
        return [node_id]
    leaves = []
    if node['left'] is not None:
        leaves.extend(get_leaf_indices(tree, node['left']))
    if node['right'] is not None:
        leaves.extend(get_leaf_indices(tree, node['right']))
    return leaves

def traverse_adaptive_cut(root_id, tree, q, sP):
    """
    Traverse the dendrogram from the root, splitting off subclusters adaptively.
    
    Step 5 of Algorithm C.1:
      - Always follow the 'bigger' child.
      - Split off the 'smaller' child if it is large enough (count > sP) AND
        its parent's distance * q > smaller child's distance.
      - Collect all 'split-off' sub-nodes in a list.

    :param root_id: int, ID of the root node in the tree
    :param tree: dict, see build_tree_structure
    :param q: float, relative branching distance
    :param sP: float, threshold ~ sqrt(|P|) + kappa^2
    :return: list of node IDs that form subclusters.
    """
    smaller_nodes = []
    stack = [root_id]
    path_of_splits = []

    while stack:
        nid = stack.pop()
        node = tree[nid]
        # If leaf, no further children to explore
        if node['left'] is None or node['right'] is None:
            if path_of_splits:
                smaller_nodes.append(nid)
            continue

        # Identify bigger vs smaller child by 'count'
        left_id = node['left']
        right_id = node['right']
        left_count = tree[left_id]['count']
        right_count = tree[right_id]['count']

        if left_count >= right_count:
            bigger_id, smaller_id = left_id, right_id
            bigger_count, smaller_count = left_count, right_count
        else:
            bigger_id, smaller_id = right_id, left_id
            bigger_count, smaller_count = right_count, left_count

        parent_dist = node['distance']
        smaller_dist = tree[smaller_id]['distance']

        # Conditions for splitting off the smaller child
        condA = (smaller_count > sP)
        condB = (q * parent_dist) > smaller_dist

        if condA and condB:
            # We do a split: the smaller child is appended
            smaller_nodes.append(smaller_id)
            path_of_splits.append(nid)
            # Continue down the bigger branch
            stack.append(bigger_id)
        else:
            # No split => treat entire node as one path
            stack.append(bigger_id)

    # The last parent that did a split => also add the bigger child
    if path_of_splits:
        last_parent = path_of_splits[-1]
        left_id = tree[last_parent]['left']
        right_id = tree[last_parent]['right']
        if tree[left_id]['count'] >= tree[right_id]['count']:
            bigger_id = left_id
        else:
            bigger_id = right_id
        smaller_nodes.append(bigger_id)

    return smaller_nodes

def age_clustering(X, dmax, kappa, q):
    """
    AGE pre-clustering (Algorithm C.1) on dataset X using average-linkage.

    :param X: np.array, shape (n, d). Data in R^d (or can be any metric if using custom code).
    :param dmax: float, maximal outlier distance (threshold for initial tree-cut).
    :param kappa: int, minimal cluster size.
    :param q: float, relative branching distance in [0,1].
    :return: (preclusters, outliers)
      preclusters: list of lists (each list is the indices of one precluster).
      outliers: list of indices deemed outliers.
    """
    from collections import defaultdict

    n = X.shape[0]
    P = set(range(n))  # current pool
    R = set()          # outliers
    C = []             # preclusters

    while len(P) > 0:
        # Handle trivially small sets
        if len(P) <= 1:
            leftover = list(P)
            if len(leftover) == 1 and len(P) >= kappa:
                C.append(leftover)
            else:
                R |= set(leftover)
            P.clear()
            break

        current_inds = np.array(list(P))
        Xsub = X[current_inds, :]

        # If there's only 1 point in Xsub
        if len(Xsub) <= 1:
            if len(Xsub) < kappa:
                R |= set(current_inds)
            else:
                C.append(list(current_inds))
            P.difference_update(current_inds)
            break

        # Step 1: average linkage
        Z = linkage(Xsub, method='average')

        # Step 2: cut at dmax => identify outliers
        labels = fcluster(Z, t=dmax, criterion='distance')
        cluster_dict = defaultdict(list)
        for i, lab in enumerate(labels):
            cluster_dict[lab].append(current_inds[i])

        # Mark clusters below size kappa as outliers
        to_remove = []
        for lab, inds_clust in cluster_dict.items():
            if len(inds_clust) < kappa:
                R |= set(inds_clust)
                to_remove.extend(inds_clust)
        for idx in to_remove:
            P.discard(idx)

        # If P is empty or has 1 point left, handle it
        if len(P) <= 1:
            leftover = list(P)
            if len(leftover) == 1 and len(P) >= kappa:
                C.append(leftover)
            else:
                R |= set(leftover)
            P.clear()
            break

        # Step 3: new average linkage after removing outliers
        current_inds = np.array(list(P))
        if len(current_inds) <= 1:
            if len(current_inds) < kappa:
                R |= set(current_inds)
            else:
                C.append(list(current_inds))
            P.difference_update(current_inds)
            break

        Xsub = X[current_inds, :]
        Z = linkage(Xsub, method='average')
        tree = build_tree_structure(Z, len(current_inds))
        root_id = 2 * len(current_inds) - 2

        # Step 4: sP
        sP = math.sqrt(len(P)) + kappa**2

        # Step 5: adaptive cut
        node_list = traverse_adaptive_cut(root_id, tree, q, sP)

        # Convert node IDs to sets of global indices
        sub_clusters = []
        for nid in node_list:
            leaf_ids = get_leaf_indices(tree, nid)
            sc = set(current_inds[leaf_id] for leaf_id in leaf_ids)
            sub_clusters.append(sc)

        # Steps 6 & 7
        if len(sub_clusters) == 0:
            # everything in P => single cluster
            C.append(list(P))
            P.clear()
        else:
            # add largest sub-cluster, remove from P
            sub_clusters.sort(key=lambda s: len(s), reverse=True)
            biggest = sub_clusters[0]
            C.append(sorted(biggest))
            P.difference_update(biggest)

    preclusters = [sorted(clust) for clust in C]
    outliers = sorted(list(R))
    return preclusters, outliers

# Optional test or demo
if __name__ == "__main__":
    np.random.seed(42)
    # Example usage
    # Cluster 1
    c1 = np.random.normal(loc=[0,0], scale=0.5, size=(20,2))
    # Cluster 2
    c2 = np.random.normal(loc=[5,5], scale=0.5, size=(20,2))
    # Some outliers
    o = np.array([[10,10],[9.5,9.8],[10.1,10.2]])
    X = np.vstack([c1, c2, o])

    preclusters, outliers = age_clustering(
        X,
        dmax=3.0,  # maximal outlier distance
        kappa=5,   # minimal cluster size
        q=0.15     # relative branching distance
    )
    print("=== AGE RESULTS ===")
    print("Preclusters:")
    for i, c in enumerate(preclusters, 1):
        print(f"  Cluster {i} (size={len(c)}): {c}")
    print("Outliers:", outliers)
