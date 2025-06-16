# -*- coding: utf-8 -*-
"""
Copyright (C) 2014-2018 Benjamin Eltzner

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 3
of the License, or (at your option) any later version.

This software is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.

You should have received a copy of the GNU General Public
License along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA
or see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import numpy.random as arandom
import numpy.linalg as la
import scipy.optimize as opt
import sys, warnings
from math import sqrt, radians, degrees
from scipy.stats.mstats import gmean
from pnds.PNDS_PNS import as_matrix

################################################################################
################################   Constants   #################################
################################################################################

PI = np.pi
RAD = radians(1)
DEG = degrees(1)
EPS = 1e-10

################################################################################
###################################   RESH   ###################################
################################################################################

# -----------------------------------------------------------------------------
# RESHify_1D: Refactored
# -----------------------------------------------------------------------------
def RESHify_1D(
    data: np.ndarray,
    invert: bool,
    mode: str = 'gap',
    codata: np.ndarray = None
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Take 1D‐angle data (rows = observations, columns = angles) and produce a 
    “spherical embedding” version of it, returning:
      1) the spherical‐embedded data (shape = [n_obs, n_dims]),
      2) the chosen 'cuts' array,
      3) the boolean 'half' flag.

    Steps:
      1. Pick a set of candidate cuts (either by `best_cuts` or by `circular_mean_cuts`).
      2. Compute a spread‐score for each candidate cut.
      3. Sort the cuts by that spread (largest first if invert=True, else smallest first).
      4. Decide whether we are in half‐range (True/False).
      5. If `codata` is provided, stack it onto the original data.
      6. Finally call `sort_data(...)` and `to_sphere(...)` to return the transformed data.
    """
    # 1) Choose cut candidates
    if mode == 'gap':
        cuts = best_cuts(data)
    else:
        cuts = circular_mean_cuts(data)

    # 2) Compute the "spread" of data relative to each cut's central angle
    #    cuts[:, 1] is assumed to be the center‐angle for that cut (in degrees).
    #    data is [n_angles, n_samples] or [n_vars, n_obs] – we subtract to get an
    #    array of the same shape to measure how far each angle in 'data' is from
    #    the cut’s center. Then adjust any difference > 180° by subtracting 360°.
    diffs = np.abs(data - cuts[:, 1])  # shape: [n_cuts, n_dims]
    diffs[diffs > 180.0] -= 360.0

    #    Now compute root‐mean‐square over each column (i.e. each cut),
    #    dividing by # of angles (len(data)) – this yields a 1D array of length n_cuts.
    #    In effect: spread[c] = sqrt( sum_j (diffs[c, j]^2 ) / n_dims ).
    spread = np.sqrt(np.sum(diffs ** 2, axis=0) / float(data.shape[1]))

    # 3) Sort 'cuts' by spread
    #    If invert=True, we want the largest‐spread cuts to come first (outermost cuts).
    #    Otherwise, we want the smallest‐spread cuts to come first (innermost).
    sort_indices = np.argsort(spread)
    if invert:
        sort_indices = sort_indices[::-1]
    cuts = cuts[sort_indices]

    # 4) Decide the 'half' flag
    #    If mode='gap', check whether any cut's third column (angle) < 180° (excluding the last row).
    #    If any are < 180°, then half = True; otherwise False.
    #    If mode!='gap', we always set half=True.
    if mode == 'gap':
        # cuts[:-1, 2] < 180. checks all but the last row’s angle
        half = bool((cuts[:-1, 2] < 180.0).any())
    else:
        half = True

    # 5) If codata is provided, stack it underneath the original data
    if codata is not None:
        data = np.vstack((data, codata))  # shape grows in rows

    # 6) Perform the final sorting & spherical mapping
    #    - sort_data(data, cuts, half) → returns a cartesian‐oned‐arr that we multiply by RAD
    #    - to_sphere(..., half) → returns final spherical coords.
    sorted_cartesian = sort_data(data, cuts, half) * RAD
    spherical_data = to_sphere(sorted_cartesian, half)

    return spherical_data, cuts, half



# -----------------------------------------------------------------------------
# unRESHify_1D: Refactored
# -----------------------------------------------------------------------------
def unRESHify_1D(
    sphere_data: np.ndarray,
    angles_info: np.ndarray,
    half: bool
) -> np.ndarray:
    """
    Reverse the transformation done by RESHify_1D.
    Given:
      - sphere_data: the output of `to_sphere(...)` (i.e. the spherical embedding),
      - angles_info: the 'cuts' array that was returned by RESHify_1D,
      - half: the boolean half‐flag that was returned by RESHify_1D,

    This routine reconstructs the original 1D angles (in degrees) from the spherical form.

    Internally it:
      1. Iteratively “unwinds” each dimension by dividing out sin(theta) terms.
      2. Recomputes each angle by arccos or arctan2 as appropriate.
      3. Scales and shifts the angles by 270° (and multiplies by 2 if half=True).
      4. Reorders the columns back to the original dimension order using `angles_info`.
      5. Adds the final global shift via angle_shift(angles_info).
    """
    # Make a local copy to avoid modifying the original input
    tmp = sphere_data.copy()  # shape: [n_obs, n_dims]
    n_dims = tmp.shape[1]

    # We'll fill angle_tmp[:, i] with the “intermediate” polar/radial values (in radians)
    angle_tmp = np.zeros_like(tmp)

    # 1) Iteratively recover each “layer” of angles from dimension 0 up to n_dims-2
    #    At step i, we know that tmp[:, i] = cos(angle_i) * Π_{j<i} sin(angle_j).
    #    Therefore, to isolate cos(angle_i), we divide tmp[:, i] by prod_{j<i} sin(angle_tmp[:, j]).
    for i in range(n_dims - 1):
        # Divide out all previous sin(angle) factors
        if i > 0:
            prod_of_sines = np.ones(tmp.shape[0])
            for j in range(i):
                prod_of_sines *= np.sin(angle_tmp[:, j]).clip(EPS, 1.0)
            tmp[:, i] = tmp[:, i] / prod_of_sines

        # Now angle_tmp[:, i] = arccos(tmp[:, i]) (clip to [-1,1] to avoid numerical error)
        angle_tmp[:, i] = np.arccos(np.clip(tmp[:, i], -1.0, 1.0))

    # 2) Handle the last column of tmp (index = n_dims-1). Its formula is
    #    tmp[:, -1] = sin(angle_{n_dims-2}) * sin(angle_{n_dims-1}), 
    #    so first divide out sin(angle_{n_dims-2}) to isolate sin(angle_{n_dims-1}).
    if n_dims >= 2:
        prod_of_sines = np.ones(tmp.shape[0])
        for j in range(n_dims - 1):
            prod_of_sines *= np.sin(angle_tmp[:, j]).clip(EPS, 1.0)
        tmp[:, -1] = tmp[:, -1] / prod_of_sines

        # Then we reconstruct the second‐to‐last angle via arctan2:
        # angle_{n_dims-2} = arctan2( tmp[:, -1], tmp[:, -2] )
        # (plus 2π to ensure we stay in [0, 2π))
        angle_tmp[:, -2] = (2.0 * PI + np.arctan2(tmp[:, -1], tmp[:, -2])) % (2.0 * PI)

    # 3) Now angle_tmp[:, :-1] are all in [0, π]. We want degrees and shift by +270°.
    #    Then if half=True, we scaled angles originally by ½, so we multiply by 2 now.
    angle_deg = angle_tmp[:, :-1] * DEG + 270.0  # shift
    if half:
        angle_deg[:, :] *= 2.0

    # 4) Build the final output array, placing each dimension back into its original index.
    #    angles_info is assumed to have the information for where each “intermediate” angle
    #    belongs in the final ordering. We create an array of shape [n_obs, n_original_dims],
    #    fill it with zeros, then assign `angle_deg[:, i]` to column = angles_info[i, 0].
    n_obs = sphere_data.shape[0]
    n_original_dims = int(np.max(angles_info[:, 0])) + 1
    recovered = np.zeros((n_obs, n_original_dims))

    for i in range(angle_deg.shape[1]):
        original_index = int(angles_info[i, 0])
        recovered[:, original_index] = angle_deg[:, i]

    # 5) Finally, add the “angle_shift” correction (mod 360)
    recovered = (recovered + angle_shift(angles_info)) % 360.0

    return recovered
# def RESHify_1D (data, invert, mode='gap', codata=None):
#     cuts = best_cuts(data) if mode == 'gap' else circular_mean_cuts(data)
#     spread = np.abs(data - cuts[:,1])
#     spread[spread>180] -= 360
#     spread = np.sqrt(np.sum(spread**2, axis=0)/len(data))
#     if invert:
#         """ Greatest spread first, i.e. outermost."""
#         cuts = cuts[spread.argsort()[::-1]]
#     else:
#         """ Greatest spread last, i.e. innermost."""
#         cuts = cuts[spread.argsort()]
#     half = (cuts[:-1,2] < 180.).any() if mode == 'gap' else True
#     if not codata is None:
#         data = np.vstack((data, codata))
#     return to_sphere(sort_data(data, cuts, half) * RAD, half), cuts, half

# def unRESHify_1D (data, angles, half):
#     tmp = data.copy()
#     angle_tmp = np.zeros(data.shape)
#     n = data.shape[1]-1
#     for i in range(n):
#         for j in range(i):
#             tmp[:,i] /= np.sin(angle_tmp[:,j]).clip(EPS, 1)
#         angle_tmp[:,i] = np.arccos(tmp[:,i].clip(-1,1))
#     for j in range(n-1):
#         tmp[:,-1] /= np.sin(angle_tmp[:,j]).clip(EPS, 1)
#     angle_tmp[:,-2] = (2 * PI + np.arctan2(tmp[:,-1], tmp[:,-2])) % (2 * PI)
#     angle_tmp = angle_tmp[:,:-1] * DEG + 270
#     angle_tmp[:,:-1] *= (2 if half else 1)
#     angle_data = np.zeros(angle_tmp.shape)
#     for i in range(n):
#         angle_data[:,int(angles[i][0])] = angle_tmp[:,i]
#     angle_data = (angle_data + angle_shift(angles)) % 360
#     return angle_data

# def RESHify (sphere1, sphere2, colors, with_plots=False):
#     return full_RESHify([sphere1, sphere2], colors, with_plots)

# def full_RESHify_old (spheres, colors, verbose=False, with_plots=False):
#     sphere_list = []
#     var_list = []
#     for s in spheres:
#         mean, var = mean_on_sphere(s, verbose)
#         x_axis = np.zeros(len(mean))
#         x_axis[0] = 1
#         rot = rotation(mean, x_axis)
#         new_s = np.einsum('ij,kj->ik', s, rot)
#         if with_plots:
# #            RESH_plot.sphere_plot(new_s, colors)
#             pass
#         sphere_list.append(new_s)
#         var_list.append(var)
#     sphere_list = np.array(sphere_list)
#     var_list = np.array(var_list)
#     sphere_list = [x for x in sphere_list[var_list.argsort()[::-1]]]
#     n = len(sphere_list)
#     inner = sphere_list[0]
#     for step in range(1,n):
#         outer = sphere_list[step]
#         factor = sqrt(np.mean(outer[:,0]**2))
#         inner = np.hstack((factor * outer[:,1:],
#                            np.einsum('i,ij->ij', outer[:,0], inner)))
#         inner = np.einsum('ij,i->ij', inner, 1/la.norm(inner, axis=1))
# #    for x in sphere_list:
# #        RESH_plot.sphere_plot(x, colors)
#     return inner

def full_RESHify (spheres, colors, with_plots=False):
    return RESHify_with_radii(spheres, np.ones(len(spheres)), colors, with_plots)

def RESHify_with_radii (spheres, radii, colors=None, verbose=False, with_plots=False):
    sphere_list = []
    var_list = []
    rot_list = []
    dimensions = []
    for s in spheres:
        dimensions.append(s.shape[-1])
        mean, var = mean_on_sphere(s, verbose)
        x_axis = np.zeros(len(mean))
        x_axis[0] = 1
        rot = rotation(mean, x_axis)
        new_s = np.einsum('ij,kj->ik', s, rot)
        if with_plots:
#            RESH_plot.sphere_plot([[new_s, colors, 'o', 30]])
            pass
        sphere_list.append(new_s)
        var_list.append(var)
        rot_list.append(rot)
    var_list = np.array([v * radii[i] for i,v in enumerate(var_list)]).argsort()[::-1]
    sphere_list = [sphere_list[i] for i in var_list]
    dimensions = [dimensions[i] for i in var_list]
    radii = radii[var_list] / gmean(radii)
    n = len(sphere_list)
    inner = sphere_list[0]
    inner[:,1:] *= radii[0]
    for step in range(1,n):
        #inner = np.einsum('ij,i->ij', inner, 1/la.norm(inner, axis=1))
        outer = sphere_list[step]
        inner = np.hstack((radii[step] * outer[:,1:],
                           np.einsum('i,ij->ij', outer[:,0], inner)))
    return (np.einsum('ij,i->ij', inner, 1/la.norm(inner, axis=1)), dimensions,
            rot_list, var_list)

def tangent_sphere (spheres, radii, colors=None, verbose=False):
    tangent_list = []
    var_list = []
    rot_list = []
    for j,s in enumerate(spheres):
        mean, var = mean_on_sphere(s, verbose)
#        mean = pns_nested_mean(s, verbose, 'small')[0]
        x_axis = np.zeros(len(mean))
        x_axis[0] = 1
        rot = rotation(mean, x_axis)
        new_s = np.einsum('ij,kj->ik', s, rot)
        tangent_space = new_s[:,1:]
        lengths = la.norm(tangent_space, axis=1)
        geodesic = np.abs(np.arcsin(lengths) / lengths - PI * (new_s[:,0] < 0))
        tangent_space = tangent_space * (geodesic * radii[j]).reshape((-1,1))
        tangent_list.append(tangent_space)
        var_list.append(var)
        rot_list.append(rot)
    var_list = np.array([v * radii[i] for i,v in enumerate(var_list)]).argsort()[::-1]
    tangent_list = [tangent_list[i] for i in var_list]
    tangent = np.hstack(tangent_list)
    print(2*np.max(la.norm(tangent, axis=1))/PI)
    tangent /= max([gmean(radii), 2*np.max(la.norm(tangent, axis=1))/PI])
    phis = la.norm(tangent, axis=1).reshape((-1,1))
    out = np.hstack((np.cos(phis), np.sin(phis) * tangent / phis))
    return (out, rot_list, var_list)

def RESHify_with_template (spheres, radii, rots, sort_list, verbose=False):
    sphere_list = []
    for i,s in enumerate(spheres):
        new_s = np.einsum('ij,kj->ik', s, rots[i])
        sphere_list.append(new_s)
    sphere_list = [sphere_list[i] for i in sort_list]
    radii = radii[sort_list] / gmean(radii)
    n = len(sphere_list)
    inner = sphere_list[0]
    inner[:,1:] *= radii[0]
    for step in range(1,n):
        #inner = np.einsum('ij,i->ij', inner, 1/la.norm(inner, axis=1))
        outer = sphere_list[step]
        inner = np.hstack((radii[step] * outer[:,1:],
                           np.einsum('i,ij->ij', outer[:,0], inner)))
    return np.einsum('ij,i->ij', inner, 1/la.norm(inner, axis=1))

def unRESHify (sphere, dimensions):
    sphere_list = []
    n = 0
    while sphere.shape[1] > dimensions[n]:
        d = dimensions[n]
        scales = la.norm(sphere[:,d-1:], axis=1)
#        factor = sqrt(np.mean(scales**2))
        factor = as_matrix(la.norm(sphere[:,:d-1], axis=1)/np.sqrt(1-scales**2)).T
        small_sphere = np.hstack((as_matrix(scales).T, sphere[:,:d-1]/factor))
        sphere = np.einsum('ij,i->ij', sphere[:,d-1:], 1/scales)
        sphere_list.append(small_sphere)
        n += 1
    sphere_list.append(sphere)
    return sphere_list

def full_unRESHify (sphere, dimensions, var_list, rot_list):
    tmp = [unRESHify(sphere, dimensions)[k] for k in np.argsort(var_list[::-1])]
    return [np.einsum('ij,jk->ik', s, rot) for s, rot in zip(tmp, rot_list)]

def sing_RESH (sphere1, sphere2, colors, with_plots=False):
    f = 0.3
    mean1, _ = mean_on_sphere(sphere1, False)
    mean2 = best_normal(find_singularities(sphere1, sphere2))
    rotation1 = rotation(mean1, np.array([1,0,0]))
    rotation2 = rotation(mean2, np.array([1,0,0]))
    new_sphere1 = np.einsum('ij,kj->ik', sphere1, rotation1)
    new_sphere2 = np.einsum('ij,kj->ik', sphere2, rotation2)
    if with_plots:
#        RESH_plot.sphere_plot(sphere1, colors)
#        RESH_plot.sphere_plot(new_sphere1, colors)
#        RESH_plot.sphere_plot(sphere2, colors)
#        RESH_plot.sphere_plot(new_sphere2, colors)
        pass
    proto_sphere = np.hstack((f * new_sphere2[:,1:], np.einsum('i,ij->ij',
                              new_sphere2[:,0], new_sphere1) / f))
    out_sphere1 = proto_sphere[new_sphere2[:,0]>0]
    colors1 = np.array(colors)[new_sphere2[:,0]>0].tolist()
    out_sphere2 = -proto_sphere[new_sphere2[:,0]<0]
    colors2 = np.array(colors)[new_sphere2[:,0]<0].tolist()
    return out_sphere1, colors1, out_sphere2, colors2

################################################################################
###########################   Auxiliary functions   ############################
################################################################################

def tangent_space (points, normal):
    new_normal = np.zeros(len(normal))
    new_normal[-1] = 1
    tangent_points = apply_matrix(points, rotation(normal, new_normal))[:,:-1]
    lengths = np.arccos(np.einsum('ij,j->i', points, normal))
    return np.einsum('ij,i->ij', tangent_points,
                     lengths / la.norm(tangent_points, axis=1))

def best_cuts (data):
    sorted_data = data.copy()
    sorted_data.sort(axis=0)
    tmp = sorted_data - np.vstack((sorted_data[-1], sorted_data[:-1]))
    tmp[tmp<0] += 360
    tmp = np.argmax(tmp, axis=0)
    means = []; spaces = []
    for j,i in enumerate(tmp):
        if sorted_data[i,j] > sorted_data[i-1,j]:
            means.append((0.5 * (sorted_data[i,j] + sorted_data[i-1,j]) + 180) % 360)
            spaces.append(sorted_data[i,j] - sorted_data[i-1,j])
        else:
            means.append(0.5 * (sorted_data[i,j] + sorted_data[i-1,j]))
            spaces.append(360 + sorted_data[i,j] - sorted_data[i-1,j])
    return np.array([list(range(len(means))), means, spaces]).T

def circular_mean_cuts (data):
    means = []
    for d in (data.T * RAD):
        n = d.size
        mean0 = np.mean(d)
        var0 = np.var(d)
        sorted_points = np.sort(d)
        candidates = variances(mean0, var0, n, sorted_points)
        candidates[:,0] = (candidates[:,0]*DEG + 360) % 360
        candidates[:,1] = 360 - np.sqrt(candidates[:,1]) * DEG
        means.append(candidates[np.argmax(candidates[:,1])])
    return np.vstack((np.array(range(len(means))), np.array(means).T)).T

def variances (mean0, var0, n, points):
    means = [(mean0 + 2 * PI * i / float(n)) for i in range(n)]
    means = [(x if x < PI else x - 2 * PI) for x in means]
    parts = [(sum(points) / n) if means[0] < 0 else 0]
    parts += [((sum(points[:i]) / i) if means[i] >= 0 else (sum(points[i:]) / (n-i)))
             for i in range(1, len(means))]
    #Formula (6) from Hotz, Huckemann
    means = [[means[i],
              var0 + ( 0 if i == 0 else
                       ((4 * PI * i / n) * (PI + parts[i] - mean0) -
                        (2 * PI * i / n)**2) if means[i] >= 0 else
                       ((4 * PI * (n - i) / n) * (PI - parts[i] + mean0) -
                        (2 * PI * (n - i) / n)**2) )]
             for i in range(len(means))]
    return np.array(means)

def euclideanize (data, mode='gap', codata=None):
    cuts = best_cuts(data) if mode == 'gap' else circular_mean_cuts(data)
    if not codata is None:
        return (data - cuts[:,1] + 540.) % 360., (codata - cuts[:,1] + 540.) % 360.
    return (data - cuts[:,1] + 540.) % 360.

def sort_data (data, means, half):
    offset = (180 if half else 90)
    output = data[:, means[:,0].astype(int)] - means[:,1]
    output[:,:-1] += offset
    output[:,-1] += 90
    return (360 + output) % 360

"""
The first angle is outermost, the last is innermost.
"""
def to_sphere (data, half_angles = True):
    tmp = np.hstack((np.copy(data), np.zeros((data.shape[0],1))))
    if (half_angles):
        tmp[:,:-2] /= 2
    return np.array([np.cos(tmp[:,0])] +
                    [np.cos(tmp[:,i+1]) * np.prod(np.sin(tmp)[:,:i+1], axis=1)
                     for i in range(tmp.shape[1] - 1)]).T

def mean_on_sphere (points, verbose):
    warnings.filterwarnings('error')
    d = points.shape[1]
    times = 1
    while points.shape[0] < points.shape[1]:
        if verbose: print('mean_on_sphere: not enough points. Duplicating.')
        points = np.vstack((points, points))
        times *= 2
    """
    Function f: R^d -> R^N for which the Levenberg-Marquardt method is
    applied to identify the argument which minimizes the sum of squares.
    """
    def f (x):
        return np.arccos(np.einsum('ij,j->i', points, x/la.norm(x)).clip(-1, 1))
    tmp = 2 * arandom.rand(d) - 1
    #print('Starting fit...',)
    sys.stdout.flush()
    try:
        tmp, exit_code = opt.leastsq(f, tmp)
    except:
        if verbose: print('Exception in fit! Setting invalid exit code.')
        exit_code = 6
    fails = 0
    while exit_code > 1:
        if verbose: print('failed "mean_on_sphere" with exit code:', exit_code, '\nRestarting...')
        fails += 1
        if fails > 3 or exit_code == 6:
            if verbose: print('with new start value... ')
            tmp = 2 * arandom.rand(d) - 1
            fails = 0
        try:
            tmp, exit_code = opt.leastsq(f, tmp)
        except:
            if verbose: print('Exception in fit! Setting invalid exit code.')
            exit_code = 6
    tmp /= la.norm(tmp)
    #print('done! Exit code:', exit_code, '\n', tmp)
    sys.stdout.flush()
    return tmp, np.sum(f(tmp)**2)/times**2

def apply_matrix (points, matrix):
    if points is None or matrix is None or len(points.shape) > 3:
        return None
    if len(points.shape) == 2:
        return np.einsum('ij,kj->ik', points, matrix)
    return np.einsum('lij,kj->lik', points, matrix)

def rotation (v_from, v_to):
    prod = float(np.einsum('i,i->', v_from, v_to))
    v_aux = v_from - prod * v_to
    if la.norm(v_aux) == 0:
        return np.eye(len(v_from))
    v_aux /= la.norm(v_aux)
    m1 = np.einsum('i,j->ij', v_aux, v_to)
    m1 = m1.T -m1
    m2 = np.einsum('i,j->ij', v_aux, v_aux) + np.einsum('i,j->ij', v_to, v_to)
    return np.eye(len(v_from)) + sqrt(1 - prod**2) * m1 + (prod -1) * m2

def euler2quaternion (data):
    c = np.cos(data).T
    s = np.sin(data).T
    return np.vstack((c[0] * c[1] * c[2] + s[0] * s[1] * s[2],
                      s[0] * c[1] * c[2] + c[0] * s[1] * s[2],
                      c[0] * s[1] * c[2] - s[0] * c[1] * s[2],
                      c[0] * c[1] * s[2] - s[0] * s[1] * c[2])).T

def best_normal (points):
    d = points.shape[1]
    """
    Function f: R^d -> R^N for which the Levenberg-Marquardt method is
    applied to identify the argument which minimizes the sum of squares.
    """
    def f (x):
        return np.einsum('ij,j->i', points, x/la.norm(x))
    tmp = 2 * arandom.rand(d) - 1
    tmp, exit_code = opt.leastsq(f, tmp)
    while exit_code > 1:
        print('failed "best_normal" with exit code:', exit_code, '\nRestarting...')
        tmp, exit_code = opt.leastsq(f, tmp)
    tmp /= la.norm(tmp)
    return tmp

def find_singularities (sphere1, sphere2):
    means = []
    for p in sphere1:
        neighbors = np.zeros(0)
        k = 0.0
        while len(neighbors) < 10:
            k += 0.001
            check = np.arccos(np.einsum('ij,j->i', sphere1, p).clip(-1,1))
            neighbors = sphere2[check<k*PI]
        dist_matrix = np.arccos(np.einsum('ik,jk->ij', neighbors, neighbors).clip(-1,1))
        dists = np.sort(dist_matrix.flatten())
        max_dist, where = np.amax(dists[1:] - dists[:-1]), np.argmax(dists[1:] - dists[:-1])
        if max_dist > 5*k*PI:
            m1, _ = mean_on_sphere(neighbors[(dist_matrix < dists[where] + 0.1 * max_dist)[0]], False)
            m2, _ = mean_on_sphere(neighbors[(dist_matrix > dists[where] + 0.1 * max_dist)[0]], False)
            m = np.mean(np.vstack((m1, m2)), axis=0)
            means.append(m/la.norm(m))
    return np.array(means)

def angle_shift (angles):
    shift = np.zeros(len(angles))
    for line in angles:
        shift[int(line[0])] = line[1]
    #print((shift+ 180e6).astype(int))
    return shift

def torus_distances (p, q):
    d = np.abs(p-q)
    d[d > 180] -= 360
    return la.norm(d, axis=1)


def sphere_distances(p, q):
    return np.arccos(np.sum(p * q, axis=1))

if __name__ == '__main__':
    M = rotation(np.array([1,0,0]), np.array([0,1,0]))

