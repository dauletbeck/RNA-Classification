from parsing.parse_functions import parse_pdb_files
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401  (needed for 3-D projection)
from scipy.cluster.hierarchy import average as average_linkage
from scipy.cluster.hierarchy import single as single_linkage
from scipy.cluster.hierarchy import fcluster
# for computing the distance matrix:
from scipy.spatial.distance import pdist, squareform
import pickle
import os

from utils.scale_low_res_coordinates import scale_low_res_coords
from utils.pucker_data_functions import determine_pucker_data
from shape_analysis import pre_clustering
from utils.pucker_data_functions import sort_data_into_cluster
from pnds.PNDS_RNA_clustering import new_multi_slink
from utils import plot_functions
from pnds.PNDS_PNS import PNS, unfold_points, as_matrix
from pnds.PNS_mode_hunter import PNSModeHunter, refine_clusters_with_pns

input_pdb_dir = "/Users/kaisardauletbek/Documents/GitHub/RNA-Classification/data/rna2020_pruned_pdbs/"
suites = parse_pdb_files(input_pdb_dir, input_pdb_folder=input_pdb_dir)

puckers = ['c2c2', 'c2c3', 'c3c2', 'c3c3']
pucker_indices = {}
for pucker in puckers:
    indices, _ = determine_pucker_data(suites, pucker)
    pucker_indices[pucker] = indices

pucker_indices = {k: np.asarray(v, dtype=np.intp)   
                  for k, v in pucker_indices.items()}

def spherical_to_vec(theta_deg: np.ndarray, phi_deg: np.ndarray) -> np.ndarray:
    t, p = np.radians(theta_deg), np.radians(phi_deg)
    return np.column_stack([np.sin(t) * np.cos(p),
                            np.sin(t) * np.sin(p),
                            np.cos(t)])

def arc_distance(a: np.ndarray, b: float) -> np.ndarray:
    """Shortest signed arc distance between vectors of angles a and scalar b (radians)."""
    d = (a - b + np.pi) % (2 * np.pi) - np.pi
    return d

def exponential_map(V, p):
    """
    V - point cloud N x R^m
    p - point of tangency R^m
    """
    N, M = V.shape[0], V.shape[1]
    V_mean = V.mean(axis=0)
    # check if points are centered at 0, center them
    V -= V_mean
    V = np.column_stack([V, np.zeros(N)])
    V_norm = np.linalg.norm(V, axis=1)[:, None]
    
    return np.cos(V_norm) * p + np.sin(V_norm) * (V / V_norm)
    
    
# if p[0] == 1:
#     rest_basis = gram_schmidt(np.vstack[p, np.eye(d)[1:]]))[1:]
# else:
#     rest_basis = gram_schmidt(np.vstack[p, np.eye(d)[:-1]]))[1:]

# scale only the distance‐variance, leave α‐variance and both means alone
scaled_coords, lambda_d, lambda_alpha = scale_low_res_coords(
    suites,
    scale_distance_variance=True,
    scale_alpha_variance=False,
    preserve_distance_mean=True,
    preserve_alpha_mean=True,
    store_attr="scaled_dvar_only"
)

d2_s, d3_s, alpha_s, theta1, phi1, theta2, phi2 = scaled_coords.T
N = len(d2_s)

min_size = 3
scale = 12000

# Where you saved them
result_dir = '/Users/kaisardauletbek/Documents/GitHub/RNA-Classification/mintage/preclustering_results'

mode_clusters_res = []

for filename in os.listdir(result_dir):
    if filename.endswith('_precluster.pkl'):
        # Load
        with open(os.path.join(result_dir, filename), 'rb') as f:
            result = pickle.load(f)

        name = result['name']
        clusters = result['clusters']
        outliers = result['outliers']
        scaled_coords_by_pucker = result['scaled_coords']

        print(name)

        # Map to sphere and angles for PNS input
        d2_s, d3_s, alpha_s, theta1, phi1, theta2, phi2 = scaled_coords_by_pucker.T

        # TODO run PN_G_S on (theta_i, phi_i), self.dists_ will return residuals, dists_[0] = theta, dists_[1] = phi since we're on S2
        S2_1 = spherical_to_vec(theta1, phi1)
        pns_S2_1 = PNS(mode='great', verbose=True).fit(S2_1)
        theta1, phi1 = pns_S2_1.dists_

        S2_2 = spherical_to_vec(theta2, phi2)
        pns_S2_2 = PNS(mode='great', verbose=True).fit(S2_2)
        theta2, phi2 = pns_S2_2.dists_

        # TODO substract the mean from d_i_s 
        V = np.column_stack([d2_s, d3_s])
        V -= np.mean(V, axis=0)

        # TODO run PN_G_S on S2_d (distances)
        # S2_pts = exponential_map(V, p=np.array([0,0,1]))
        S2_d = exponential_map(V, p=np.array([0,0,1])) # try this should be better
        pns_S2_d = PNS(mode='great', verbose=True).fit(S2_d)
        theta_d, phi_d = pns_S2_d.dists_

        # r = np.linalg.norm(S2_pts, axis=1)
        # theta_e_deg = np.degrees(np.arccos(np.clip(z/r, -1, 1)))
        # phi_e_deg = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0

        # TODO maybe add 180 for plots
        angle_matrix = np.column_stack([
            theta_d + 180,
            phi_d + 180,
            alpha_s,
            theta1 + 180,
            phi1 + 180,
            theta2 + 180,
            phi2 + 180
        ])
        # if filename == 'c2c2_precluster.pkl':
        #     break

        # Run PNS-based clustering
        mode_clusters, _ = refine_clusters_with_pns(
            scale=scale,
            data=angle_matrix,
            cluster_list=clusters,
            outlier_list=outliers,
            min_cluster_size=min_size
        )

        mode_clusters_res.append({
            'name': name,
            'mode_clusters': mode_clusters
        })

        print(f"[{name}] mode clusters done: {len(mode_clusters)} clusters")

# You can save final mode_clusters_res if needed
with open('mode_clusters_results.pkl', 'wb') as f:
    pickle.dump(mode_clusters_res, f)




