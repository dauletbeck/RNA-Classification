import numpy as np
from shape_analysis import circular_mean
from typing import Sequence, Tuple
from utils.data_functions import arc_distance, spherical_to_vec

# ---------------------------------------------------------------------------
# main routine
# ---------------------------------------------------------------------------
def scale_low_res_coords(
    suites: Sequence,
    *,
    # NEW flags:
    scale_distance_variance: bool = True,
    scale_alpha_variance:    bool = True,
    preserve_distance_mean:  bool = True,
    preserve_alpha_mean:     bool = True,
    store_attr: str = "scaled_low_res_coords"
) -> Tuple[np.ndarray, float, float]:
    """Return scaled coords plus the two scale‐factors (lambda_d, lambda_alpha)."""

    # pull raw coords
    coords = np.array([s.low_resolution_coordinates() for s in suites])
    d2, d3, alpha_deg = coords[:,0], coords[:,1], coords[:,2]
    theta1, phi1     = coords[:,3], coords[:,4]
    theta2, phi2     = coords[:,5], coords[:,6]

    # 1) compute Frechét variances
    # 1a) Euclidean on (d2,d3)
    dist_mat  = np.column_stack([d2,d3])
    mean_dist = dist_mat.mean(axis=0)
    var_d     = np.mean(np.sum((dist_mat - mean_dist)**2, axis=1))

    # 1b) intrinsic on alpha
    alpha_rad      = np.radians(alpha_deg) % (2*np.pi)
    alpha_mu_mod, var_a = circular_mean(alpha_rad)
    alpha_mu      = (alpha_mu_mod + np.pi) % (2*np.pi) - np.pi

    # 1c) spherical atoms
    vec1 = spherical_to_vec(theta1, phi1)
    vec2 = spherical_to_vec(theta2, phi2)
    var_b1 = np.mean(np.sum((vec1 - vec1.mean(axis=0))**2, axis=1))
    var_b2 = np.mean(np.sum((vec2 - vec2.mean(axis=0))**2, axis=1))

    # 2) the “target” variances from your supervisor’s formula
    v_d_target = (var_b1 + var_b2)/3
    v_a_target = (var_b1 + var_b2)/6

    # 3) raw scale‐factors
    lambda_d     = np.sqrt(v_d_target / var_d)
    lambda_alpha = np.sqrt(v_a_target / var_a)

    # 4) if user turned variance‐scaling off, reset these to 1
    if not scale_distance_variance:
        lambda_d = 1.0
    if not scale_alpha_variance:
        lambda_alpha = 1.0

    # 5) apply scaling (+ optional “preserve mean” shifts)
    if preserve_distance_mean:
        new_dist = mean_dist + lambda_d*(dist_mat - mean_dist)
    else:
        new_dist = lambda_d*dist_mat
    d2_s, d3_s = new_dist[:,0], new_dist[:,1]

    if preserve_alpha_mean:
        a_shifted = alpha_mu + lambda_alpha*arc_distance(alpha_rad, alpha_mu)
        a_shifted = (a_shifted + np.pi) % (2*np.pi) - np.pi
        alpha_s = np.degrees(a_shifted)
    else:
        alpha_s = lambda_alpha*alpha_deg

    # 6) package & store
    scaled = np.column_stack([
        d2_s, d3_s, alpha_s,
        theta1, phi1, theta2, phi2
    ])
    for suite,row in zip(suites, scaled):
        setattr(suite, store_attr, row.tolist())

    return scaled, lambda_d, lambda_alpha
