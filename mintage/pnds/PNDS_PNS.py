# -*- coding: utf-8 -*-
"""
Copyright (C) 2014 Benjamin Eltzner

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
#import scipy.linalg as scila
import scipy.optimize as opt
import scipy.stats as stat
import math, sys, warnings, scipy
from math import sqrt, acos, log, exp, sin
from scipy.integrate import quad

################################################################################
################################   Constants   #################################
################################################################################

PI = np.pi
DEG = math.degrees(1)
EPS = 1e-8
OLD_SCIPY = [int(x) for x in scipy.__version__.split('.')]
OLD_SCIPY = (OLD_SCIPY[0] <= 0 and OLD_SCIPY[1] < 15)

################################################################################
###################################   PNS   ####################################
################################################################################

def fit (f, d, initial, verbose, small=True):
    warnings.filterwarnings('error')
    tol = 1e-8
    if verbose:
        print('Starting fit... ',)
        sys.stdout.flush()
    try:
        initial, exit_code = opt.leastsq(f, initial)
    except Exception as e:
        if verbose: print('Exception in fit! Setting invalid exit code: ' + str(e))
        exit_code = 6
    fails = 0
    counter = 0
    while exit_code > 1 or (small and abs(initial[-1]) > 1):
        if verbose:
            print('failed "get_sphere" with exit code:', exit_code, 'Counter:', counter, '\nRestarting... ')
        fails += 1
        if fails > 3 or exit_code == 6:
            if verbose:
                print('with new start value... ')
            initial = 2 * arandom.rand(d) - 1
            fails = 0
        try:
            initial, exit_code = opt.leastsq(f, initial, ftol=tol, xtol=tol)
        except Exception as e:
            if verbose: print('Exception in fit! Setting invalid exit code: ' + str(e))
            exit_code = 6
        if counter > 20:
            return None
        counter += 1
        tol *= 2
    if small:
        initial[:-1] /= la.norm(initial[:-1])
    else:
        initial /= la.norm(initial)
    if verbose:
        print('done! Exit code:', exit_code, '\n', initial)
        sys.stdout.flush()
    return initial

def __sphere2torus (data, half):
    tmp = data.copy()
    angle_data = np.zeros(data.shape)
    n = data.shape[1]-1
    for i in range(n):
        for j in range(i):
            tmp[:,i] /= np.sin(angle_data[:,j]).clip(EPS, 1)
        angle_data[:,i] = np.arccos(tmp[:,i].clip(-1,1))
    for j in range(n-1):
        tmp[:,-1] /= np.sin(angle_data[:,j]).clip(EPS, 1)
    angle_data[:,-2] = (2 * PI + np.arctan2(tmp[:,-1], tmp[:,-2])) % (2 * PI)
    angle_data = angle_data[:,:-1] * DEG + 270
    angle_data[:,:-1] *= (2 if half else 1)
    return ((angle_data + 180) % 360)

def __torus_dists (p, q, half):
    d = np.abs(__sphere2torus(p, half) - __sphere2torus(q, half))
    d[d > 180] -= 360
    return la.norm(d, axis=1)

def __new_seed (old, d):
    if len(old) <= 0:
        normal = 2 * arandom.rand(d) - 1
        return normal / la.norm(normal)
    out = old[0]
    d = old.shape[1]
    while np.any(np.abs(np.einsum('i,ji->j', out, old)) > 0.7): # cos(45°)
        out = 2 * arandom.rand(d) - 1
        out /= la.norm(out)
    return out

def __get_functions (points, list_spheres, half, verbose):
    def f(x):
        norm_x = la.norm(x)
        angles = np.arcsin(np.einsum('ij,j->i', points, x / norm_x).clip(-1, 1))
        return np.hstack((angles - np.mean(angles), norm_x - 1))
    def f2 (x):
        return Sphere1(x / la.norm(x), 0).distances(points)
    def g (x):
        if abs(x[-1]) > 1:
            if verbose:
                print('Fail:', x[-1])
            return 180 * np.ones(points.shape[0]) * abs(x[-1])
        feet = Sphere1(x[:-1] / la.norm(x[:-1]), x[-1]).foot_points(points)
        if len(list_spheres) > 0:
            return __torus_dists(unfold_points(feet, list_spheres),
                                 unfold_points(points, list_spheres), half)
        return __torus_dists(feet, points, half)
    def g2 (x):
        feet = Sphere1(x / la.norm(x), 0).foot_points(points)
        if len(list_spheres) > 0:
            return __torus_dists(unfold_points(feet, list_spheres),
                                 unfold_points(points, list_spheres), half)
        return __torus_dists(feet, points, half)
    return f, f2, g, g2


def get_sphere (points, orig_points, list_spheres, max_repetitions, half,
                verbose, mode):
    N, d = points.shape
#    if verbose:
#    if N <= d:
#        mode = 'great'
    print('Dimension:', d, 'Mode:', mode, flush=True)
    N2 = len(max_linear_independent_subset(points))
    # Special Case: High dimension low sample size
#    if N <= d or N2 < d:
#        tmp = points.copy()
#        sup_vec = tmp[-1]
#        tmp = gram_schmidt(tmp[:-1] - sup_vec)
#        n = len(tmp)
#        if verbose:
#            print('%d points with dimension %d. Making S^%d' % (N, d, n-1))
#        tmp = np.vstack((tmp, 2 * arandom.rand(d - n, d) - 1))
#        normals = gram_schmidt(tmp)[n:]
#        sup_vec = np.einsum('ij,j->i', normals, sup_vec)
#        if normals.shape[0] == 1 and sup_vec.shape[0] == 1:
#            return Sphere1(normals[0], sup_vec[0])
#        return Sphere(normals, sup_vec)
    if N < d or N2 < d:
        tmp = points.copy()
        tmp = gram_schmidt(tmp)
        n = len(tmp)
        if verbose:
            print('%d points with dimension %d. Making S^%d' % (N, d, n))
        tmp = np.vstack((tmp, 2 * arandom.rand(d - n, d) - 1))
        normals = gram_schmidt(tmp)[n:]
        if normals.shape[0] == 1:
            return Sphere1(normals[0], 0)
        return Sphere(normals, np.zeros(normals.shape[0]))
    N2 = len(max_linear_independent_subset(points))
    def __fit_with_height (func, dim, initial, verbose):
        tmp = fit(func, dim, initial, verbose, False)
        if tmp is None:
            return None
        height = sin(np.mean(np.arcsin(np.einsum('ij,j->i', points, tmp/la.norm(tmp)).clip(-1, 1))))
        return (1 if height >= 0 else -1) * np.hstack((tmp, height))
    def __one_small_circle_run (f, d, starts, verbose):
        out = __fit_with_height(f, d, starts[-1], verbose)
        counter = 0
        while out is None:
            starts[-1] = __new_seed(np.array(starts[:-1]), d)
            out = __fit_with_height(f, d, starts[-1], verbose)
            counter += 1
            if counter > max_repetitions:
                return None
        return out
    f, f2, g, g2 = __get_functions(points, list_spheres, half, verbose)
    def __small_circle_fit ():
        ext_mean = np.mean(points, axis=0)
        starts =  [ext_mean / la.norm(ext_mean)]
        results = [__one_small_circle_run(f, d, starts, verbose)]
        if results[-1] is None:
            scores = [np.inf]
        else:
            scores = [np.sum(f(results[-1][:-1])**2)]
        reruns = min(max_repetitions, d + 1)
        for i in range(reruns):
            starts.append(__new_seed(np.array(starts), d))
            results.append(__one_small_circle_run(f, d, starts, verbose))
            if results[-1] is None:
                scores.append(np.inf)
            else:
                scores.append(np.sum(f(results[-1][:-1])**2))
        scores = np.array(scores)
        return results[np.argmin(scores)]
    tmp = 2 * arandom.rand(d + 1) - 1
    if mode == 'torus':
        tmp = __small_circle_fit()
        if tmp is None:
            tmp = fit(f2, d, 2 * arandom.rand(d) - 1, verbose, False)
        if tmp is None:
            return None
        if verbose: print(la.norm(g(tmp)), la.norm(f(tmp[:-1])))
        tmp = fit(g, d+1, tmp, verbose)
        if tmp is None:
            print('WARNING: Torus fit failed! Falling back to small sphere fit.')
            mode = 'small'
            tmp = __small_circle_fit()
            if tmp is None:
                print('WARNING: Small sphere fit failed! Falling back to great sphere fit.')
                mode = 'great'
                tmp = fit(f2, d, 2 * arandom.rand(d) - 1, verbose, False)
        if verbose: print(la.norm(g(tmp)), la.norm(f(tmp[:-1])))
    elif mode == 'great':
        tmp = fit(f2, d, tmp[:-1], verbose, False)
        if tmp is None:
            return None
        return Sphere1(tmp / la.norm(tmp), 0)
    elif mode == 'small':
        tmp = __small_circle_fit()
        if tmp is None:
            print('WARNING: Small sphere fit failed! Falling back to great sphere fit.')
            mode = 'great'
            tmp = fit(f2, d, 2 * arandom.rand(d) - 1, verbose, False)
        return Sphere1(tmp[:-1] / la.norm(tmp[:-1]), tmp[-1])
    else:
        tmp = __small_circle_fit()
    if tmp is None:
        return None
    if mode in ['scale', 'torus']:
        radii = np.arccos(np.abs(np.einsum('ij,j->i', points, tmp[:-1])))
        #tmp2 = tmp.copy()
        tmp2 = fit((g2 if mode == 'torus' else f2), d, tmp[:-1], verbose, False)

        # log_res_small = np.log(np.sum(g(tmp)**2) if mode is 'torus' else np.sum(f(tmp)**2)) # old code
        log_res_small = np.log(np.sum(g(tmp)**2) if mode == 'torus' else np.sum(f(tmp[:-1])**2))

        log_res_great = np.log(np.sum(g2(tmp2)**2) if mode == 'torus' else np.sum(f2(tmp2)**2))
        chi2 = 1 - stat.chi2.cdf((log_res_great - log_res_small)*N, 1)

        if chi2 > 0.05:
            print('TEST CHI 2')
            return Sphere1(tmp2 / la.norm(tmp2), 0)

        if compare_likelihoods(radii, d-1, verbose):
            ang = np.arccos(np.einsum('i,i->', tmp2, tmp[:-1])) * DEG
            if verbose: print('CHANGED:', ang, abs(tmp[-1]))
            return Sphere1(tmp2 / la.norm(tmp2), 0)
    if mode == 'great':
        return Sphere1(tmp / la.norm(tmp), 0)
    return Sphere1(tmp[:-1] / la.norm(tmp[:-1]), tmp[-1])


def pns_loop(points, great_until_dim,  max_repetitions=10, degenerate=False, verbose=False, mode=None, half=False):
    list_spheres = []
    list_points = []
    list_dists = []
    while points.shape[1] > 2:
        # Do PNS
        orig_points = points.copy()
        this_mode = ('great' if (points.shape[1] > great_until_dim + 1) else mode)
        list_spheres.append(get_sphere(points, orig_points, list_spheres,
                                       max_repetitions, half, verbose, this_mode))
        if list_spheres[-1] == None:
            return None, None, None
        dists, feet = list_spheres[-1].distances(points, True)
        #print(dists)
#        if mode is 'torus':
#            dists = np.sign(dists) * __torus_dists(unfold_points(feet, list_spheres[:-1]),
#                                                   unfold_points(points, list_spheres[:-1]),
#                                                   half) / DEG
        if degenerate:
            if mode == 'torus':
                print('WARNING: Changing distances in torus fit gives bad projections!')
            points_out = unfold_points(points, list_spheres[:-1])
            sphere_out = Sphere.unfold(list_spheres)
            # Determine alternative distances and projections
            dists, feet = alt_dist_loop(points_out, dists, feet, sphere_out,
                                        list_spheres[:-1], verbose)
        points = list_spheres[-1].project(feet)
        list_points.append(points)
        list_dists.append(dists * DEG)
    if points.shape[1] < 2:
        return None, None, None
    mean, residuals = nested_mean(points)
    list_spheres.append(None)
    list_points.append(mean)
    list_dists.append(residuals)
#    if mode is 'torus':
#        p1 = unfold_points(np.tile(mean, (len(points),1)), list_spheres[:-1])
#        p2 = unfold_points(points, list_spheres[:-1])
#        list_dists[-1] = np.sign(residuals) * __torus_dists(p1, p2, half)
    return list_spheres, list_points, list_dists

def pns_nested_mean (sphere, mode, verbose):
    spheres, pro_pts, _ = pns_loop(sphere, 1000, 10, False, verbose, mode=mode, half=False)
    mean = unfold_points(as_matrix(pro_pts[-1]), spheres[:-1])[0]
    variance = np.mean(np.arccos(np.einsum('ij,j->i', sphere, mean).clip(-1,1))**2)
    return mean, variance

################################################################################
######################   Calculation of circular means   #######################
################################################################################
"""
Calculation of circular means is based on the results of
Hotz, T., Huckemann, S.:
"Intrinsic Means on the Circle: Uniqueness, Locus and Asymptotics".
The Annals of the Institute of Mathematical Sciences, arXiv:1108.2141
To appear (2014)
"""

def center_of_mass (points_2d):
    phi, var = circular_mean(np.arctan2(points_2d[:,1], points_2d[:,0]))
    return np.array([math.cos(phi), math.sin(phi)]), sqrt(var) * DEG

def euclideanize (angles):
    return np.array([((5 * PI + a - circular_mean(a)[0]) % (2 * PI) - PI) * DEG
                     for a in angles])

def nested_mean (points_2d):
    phis = np.arctan2(points_2d[:, 1], points_2d[:, 0])
    mean, _ = circular_mean(phis)
    phis = (5 * PI + phis - mean) % (2 * PI) - PI
    return np.array([math.cos(mean), math.sin(mean)]), phis * DEG

def circular_mean (points, perimeter=2*PI):
    data = 2 * PI * points / perimeter
    n = data.size
    mean0 = np.mean(data)
    var0 = np.var(data)
    sorted_points = np.sort(data)
    means = variances(mean0, var0, n, sorted_points)

    tmp = means[np.argmin(means[:, 1])]
    tmp *= 0.5 * perimeter / PI
    tmp[1] *= 0.5 * perimeter / PI
    return tmp

def variances (mean0, var0, n, points):
    means = (mean0 + np.linspace(0, 2 * PI, n, endpoint=False)) % (2 * PI)
    means[means >= PI] -= 2 * PI
    parts = [(sum(points) / n) if means[0] < 0 else 0]
    m_plus = means >= 0
    lo_sums = np.cumsum(points)
    hi_sums = lo_sums[-1] - lo_sums
    i = np.array(range(n))
    j = i[1:]
    p2 = hi_sums[:-1] / (n-j)
    p2[m_plus[1:]] = (lo_sums[:-1] / j)[m_plus[1:]]
    parts = np.hstack([parts, p2])
    #Formula (6) from Hotz, Huckemann
    plus_vec = (4 * PI * i / n) * (PI + parts - mean0) - (2 * PI * i / n)**2
    minus_vec = (4 * PI * (n - i) / n) * (PI - parts + mean0) - (2 * PI * (n - i) / n)**2
    minus_vec[m_plus] = plus_vec[m_plus]
    means = np.vstack([means, var0 + minus_vec]).T
    return np.array(means)


def torus_mean_and_var(data, perimeter=2*PI):
    mean = []
    variance = 0
    for k in range(data.shape[1]):
        tmp = circular_mean(data[:,k], perimeter)
        mean.append(tmp[0])
        variance += tmp[1]
    return np.array(mean), variance

################################################################################
#############################   Likelihood Test   ##############################
################################################################################


def normalization (rho, sigma, d, euclidean=False):
    def f(r): return (exp(-0.5*(r/sigma-rho)**2) + exp(-0.5*(r/sigma+rho)**2))
    try:
        if not euclidean:
            return max(sys.float_info.min, quad(lambda r: sin(r)**(d-1) * f(r), 0, PI)[0])
        else:
            return max(sys.float_info.min, quad(lambda r: r**(d-1) * f(r), 0, (20+rho)*sigma)[0])
    except:
        #print("WARNING: Integral did not converge for sigma ", sigma)
        return max(sys.float_info.min, sqrt(2 * PI) * sigma)


def compare_likelihoods(radii, d, verbose,  euclidean=False):
    mean = radii.mean()
    std = radii.std()
    def likelihood (x):
        penalty = 0.
        scale = 2. * x[0] * x[1] / PI
        if (scale > 1) and not euclidean:
            x[0] = 0.5 * PI / x[1]
            penalty = scale
        out = np.sum(log(normalization(x[0],x[1],d, euclidean= euclidean)) + 0.5*(radii/x[1]-x[0])**2 -
                     np.log(1 + np.exp(- 2. * x[0] * radii / x[1]))) + penalty
        return out
    def likelihood_alt (x):
        penalty = 0.
        scale = 2. * x[0] * x[1] / PI
        if (scale > 1) and not euclidean:
            x[0] = 0.5 * PI / x[1]
            penalty = scale
        out = np.sum(log(x[1]) + 0.5*(radii/x[1]-x[0])**2 -
                     np.log(1 + np.exp(- 2. * x[0] * radii / x[1]))) + penalty
        return out
    def likelihood_null(x):
        # Convert x to scalar if it's an array
        if hasattr(x, '__len__'):
            x = x[0]  # Safely extract the first element
        return likelihood(np.array([1, float(x)]))
    mle = opt.minimize(likelihood, np.array([max(mean/std, 1.), std]),
                       method=('L-BFGS-B' if OLD_SCIPY else None),
                       bounds=((0,PI*1e3),(max(1e-3,0.25*std),max(10*std,1e-2)))).x
    alt_mle = opt.minimize(likelihood_alt, np.array([max(mean/std, 1.), std]),
                           method=('L-BFGS-B' if OLD_SCIPY else None),
                           bounds=((0,PI*1e3),(max(1e-3,0.25*std),max(10*std,1e-2)))).x
    if verbose:
        print('Mean: %.3f %.3f' % (mean, std),
              'MLE: %.3f %.3f' % (mle[0]*mle[1], mle[1]),
              'JMD MLE: %.3f %.3f' % (alt_mle[0], alt_mle[1]))
    if mle[0] < 1:
        return True

    mle_null = opt.minimize(likelihood_null, 1,
                            method=('L-BFGS-B' if OLD_SCIPY else None),
                            bounds=((mle[1], 10*max(std, mle[1])),)).x
    chi2 = 1 - stat.chi2.cdf(2 * (likelihood_null(mle_null[0]) - likelihood(mle)), 1)
    print("chi2 in likelihood", chi2, "mle", mle, "mle_null", mle_null)
    if (verbose and (chi2 > 0)): print('Chi-square: ', chi2)
    return chi2 > 0.05

################################################################################
###########################   Auxiliary functions   ############################
################################################################################

def alt_dist_loop (points, dists, feet, sphere, list_spheres, verbose):
    """
    Determine alternative distances and projections to the PNS sphere for
    points that are sufficiently close to the degenerate sphere.
    """
    dists_deg = deg_distances(points, 0)
    n_close = 0
    n_changed = 0
    for i in range(len(dists)):
        if dists_deg[i] < abs(dists[i]):
            n_close += 1
            tmp = alt_distance(points[i], dists[i], sphere)
            if tmp[0] + EPS < dists[i]:
                n_changed += 1
                if list_spheres:
                    dists[i], foot = tmp
                    feet[i] = fold_points(foot, list_spheres)
                else:
                    dists[i], feet[i] = tmp
    if verbose: print(n_changed, 'of', n_close, 'distances changed.')
    sys.stdout.flush()
    return dists, feet

def alt_distance (p_data, dist, pns_sphere):
    """
    This function is called only for points close to the degenerate sphere.
    p_data     : data point
    dist       : distance of p_data to the PNS sphere
    pns_sphere : the PNS sphere
    """
    n_max = len(p_data) - 2
    min_dist = dist
    best_foot = None
    for i in range(n_max):
        dist_deg, q_data = deg_distance(p_data, i, True)
        if dist_deg > min_dist:
            return min_dist, best_foot
        # Find best passage through degenerate sphere.
        def d (q, with_point = False):
            q /= la.norm(q)
            q = np.hstack((q,np.array([0]*(2+i))))
            q_bar = q.copy()
            q_bar[-3-i] *= -1
            p_pns = pns_sphere.foot_points(as_matrix(q_bar))
            if with_point:
                return bow(p_data, q) + bow(q_bar, as_vector(p_pns)), p_pns
            return bow(p_data, q) + bow(q_bar, as_vector(p_pns))
        try:
            alt_dist, p_pns = d(opt.minimize(d, q_data[:-2-i]).x, True)
        except:
            print('WARNING: Calculating alternative distance failed!')
            return (np.inf, None)
        if alt_dist < min_dist:
            min_dist = alt_dist
            best_foot = p_pns
    return min_dist, best_foot

def fold_points (points, list_spheres):
    out = points.copy()
    for sphere in list_spheres:
        out = sphere.project(out)
    return out

def unfold_points (points, list_spheres):
    out = points.copy()
    for sphere in reversed(list_spheres):
        out = sphere.unproject(out)
    return out

def bow (p, q):
    prod = np.einsum('i,i->', p, q)
    if prod > 1: prod = 1
    if prod < -1: prod = -1
    return acos(prod)

def deg_distances (points, codim, with_feet = False):
    if with_feet:
        tmp = points.copy()
        tmp[:,-2-codim:] = 0
        feet = np.einsum('ij,i->ij', tmp, 1/la.norm(tmp, axis=1))
        dists = np.arccos(np.einsum('ij,ij->i', points, feet).clip(-1, 1))
        return dists, feet
    return np.arccos(la.norm(points[:,:-2-codim], axis=1).clip(-1, 1))

def deg_distance (point, codim, with_foot = False):
    if with_foot:
        foot = point.copy()
        foot[-2-codim:] = 0
        foot /= la.norm(foot)
        dist = acos(max(min(np.einsum('i,i->', point, foot), 1), -1))
        return dist, foot
    return np.acos(la.norm(point[:-2-codim]))

def as_matrix (vector):
    return vector.reshape(1, len(vector))

def as_vector (matrix):
    return matrix.reshape(matrix.shape[1])

def gram_schmidt (vectors):
    if vectors.shape[0] == 1:
        return vectors
    tmp = max_linear_independent_subset(vectors)
    tmp = np.einsum('ij,i->ij', tmp, 1/la.norm(tmp, axis=1))
    if len(tmp) == 0:
        return None
    out = tmp[[0]]
    for i in range(1, len(tmp)):
        new = (tmp[i] - np.einsum('ij,ik,j->k', out, out, tmp[i]))
        out = np.vstack((out, new/la.norm(new)))
    return out


def orthogonal (vector):
    return gram_schmidt(np.vstack((vector, arandom.rand(len(vector)))))[-1]


def max_linear_independent_subset (vectors):
    counter = 0
    out = None
    for i in range(len(vectors)):
        if la.norm(vectors[i]) < EPS:
            pass
        elif counter == 0:
            out = vectors[i][np.newaxis,:]
            counter = 1
        else:
            tmp = np.vstack((out, vectors[i]))
            if safe_rank(tmp) > counter:
                out = tmp
                counter += 1
    return out

# This is a workaround for a weird bug probably in some version of the GNU
# scientific library, which breaks la.matrix_rank (as well as la.svd and la.eig)
# if the number of rows or columns is 129 or 130.
def safe_rank (matrix):
    r, c = matrix.shape
    if (r - 1) % 16 < 2:
        matrix = np.vstack((matrix, np.zeros((2,c))))
    if (c - 1) % 16 < 2:
        matrix = np.hstack((matrix, np.zeros((matrix.shape[0],2))))
    return la.matrix_rank(matrix, EPS)

def array2csv (array, form = '%.6f', s=''):
    if len(array.shape) == 1:
        return s + ','.join(form % item for item in array) + '\n'
    if len(array.shape) == 2:
        return s + ('\n'+s).join(','.join(form % i for i in k) for k in array) + '\n'
    if len(array.shape) > 2:
        marker = ',' * (len(array.shape) - 1) + '\n'
        return s + marker.join([array2csv(m) for m in array])
    return None

################################################################################
#########################   Auxiliary sphere classes   #########################
################################################################################

"""
This class describes a sub-sphere with codimension 1 inside a sphere.
normal : center of the sub-sphere on the sphere
center : center of the sub-sphere in embedding space
"""
class Sphere1 (object):
    def __init__ (self, normal, height):
        self.normal = normal
        self.height = max(-1, min(1, height))
        if self.height < 0:
            self.height *= -1
            self.normal *= -1
        self.center = self.height * self.normal
        self.radius = acos(self.height)
        self.flat_radius = sqrt(1 - self.height**2)

    def on_plane (self, point):
        return np.abs(np.inner((point - self.center), self.normal)) < EPS

    def foot_points (self, points):
        tmp = points - np.einsum('j,k,lj->lk', self.normal, self.normal, points)
        if np.any(la.norm(tmp, axis=1) == 0):
            return None
        return self.center + self.flat_radius * np.einsum('ij,i->ij', tmp,
                                                           1/la.norm(tmp, axis=1))

    def distances (self, points, with_feet = False):
        dists = (self.radius - np.arccos(np.einsum('ij,j->i', points,
                                                   self.normal).clip(-1, 1)))
        if with_feet:
            feet = self.foot_points(points)
            return dists, feet
        return dists

    def calculate_projection_matrix (self):
        self.projection_matrix = np.vstack((self.normal[np.newaxis,:], np.eye(len(self.normal))))
        self.projection_matrix = gram_schmidt(self.projection_matrix)[1:]


    def project (self, points):
        if not hasattr(self, 'projection_matrix'):
            self.calculate_projection_matrix()
        return np.einsum('ik,jk->ij', self.foot_points(points),
                          self.projection_matrix) / self.flat_radius

    def unproject (self, points):
        if not hasattr(self, 'projection_matrix'):
            print('WARNING: This sphere has not been used for projection!')
            self.calculate_projection_matrix()
        return (np.einsum('ik,kj->ij', points, self.projection_matrix) *
                self.flat_radius + self.center)

    def descriptor (self):
        size = len(self.normal) - (abs(self.height) < EPS)
        tmp = np.hstack((self.normal, np.array([self.height])))
        return tmp, size

    def __str__ (self):
        if not hasattr(self, 'projection_matrix'):
            self.calculate_projection_matrix()
        return Sphere.string(self.flat_radius, self.center, self.projection_matrix)


"""
This class describes a sub-sphere with arbitrary codimension c inside a sphere.
As a sub-sphere of codimension c is uniquely defined by a hyperplane of
codimension c, the input data must specify such a hyperplane.
normals  : An array of c orthonormal vectors, that are orthogonal on the
           hyperplane.
position : A c-dimensional vector specifying the position of the hyperplane in
           terms of the normals.
"""
class Sphere (object):
    def __init__ (self, normals, position):
        if normals.shape[0] != len(position):
            print('INVALID HYPERPLANE!\n', normals, '\n', position)
            return
        if (la.norm(normals - gram_schmidt(normals)) > EPS):
            print('NORMALS ARE NOT ORTHONORMAL!',
                  la.norm(normals - gram_schmidt(normals)), '\n', normals)
            return
        self.normals = normals
        height = la.norm(position)
        if height > 1:
            print('INVALID POSITION!\n', position)
            return
        self.center = np.einsum('ij,i->j', self.normals, position)
        self.position = position
        self.flat_radius = sqrt(1 - height**2)

    @classmethod
    def unfold (cls, list_of_spheres):
        """
        Transform a list of n codimension 1 spheres into a codimension n sphere.
        """
        if list_of_spheres == None or len(list_of_spheres) == 0:
            return None
        if len(list_of_spheres) == 1:
            return list_of_spheres[0]
        normals = []
        position = []
        for i, sphere in enumerate(list_of_spheres):
            try:
                normal = sphere.normal.copy()
                height = sphere.height
                for sphere in reversed(list_of_spheres[:i]):
                    normal = np.einsum('k,kj->j', normal, sphere.projection_matrix)
                    height *= sphere.flat_radius
                normals.append(normal)
                position.append(height)
            except:
                if i > 0:
                    return None
                print("Problem unfolding. Trying Sphere instead of Sphere1.")
                normals = [n for n in sphere.normals.copy()]
                position = [p for p in sphere.position.copy()]
        return Sphere(np.array(normals), np.array(position))

    @classmethod
    def string(cls, r, c, m):
        return '%%radius\n%.6f\n%%center\n%s%%matrix\n%s' % (r, array2csv(c),
                                                               array2csv(m))

    def foot_points (self, points):
        tmp = points - np.einsum('ij,ik,lj->lk', self.normals, self.normals, points)
        if np.any(la.norm(tmp, axis=1) == 0):
            return None
        return self.center + self.flat_radius * np.einsum('ij,i->ij', tmp,
                                                           1/la.norm(tmp, axis=1))

    def distances (self, points, with_feet = False):
        feet = self.foot_points(points)
        dists = np.arccos(np.einsum('ij,ij->i', points, feet).clip(-1, 1))
        if with_feet:
            return dists, feet
        return dists

    def calculate_projection_matrix (self):
        self.projection_matrix = np.vstack((self.normals, np.eye(self.normals.shape[1])))
        self.projection_matrix = gram_schmidt(self.projection_matrix)[len(self.normals):]


    def project (self, points):
        if not hasattr(self, 'projection_matrix'):
            self.calculate_projection_matrix()
        return np.einsum('ik,jk->ij', self.foot_points(points),
                          self.projection_matrix) / self.flat_radius

    def unproject (self, points):
        if not hasattr(self, 'projection_matrix'):
            print('WARNING: This sphere has not been used for projection!')
            self.calculate_projection_matrix()
        return (np.einsum('ik,kj->ij', points, self.projection_matrix) *
                self.flat_radius + self.center)

    def descriptor (self):
        n,d = self.normals.shape
        size = n * (d-n) + (la.norm(self.position) > EPS) * n
        return np.hstack((self.normals, self.position[:,np.newaxis])), size

    def __str__ (self):
        if not hasattr(self, 'projection_matrix'):
            self.calculate_projection_matrix()
        return Sphere.string(self.flat_radius, self.center, self.projection_matrix)

"""
This class describes a geodesic inside a sphere.
"""
class Geodesic (object):
    def __init__ (self, span):
        span = gram_schmidt(span.reshape((2,-1)))
        if (la.norm(span - gram_schmidt(span)) > EPS):
            print('SPANNING VECTORS ARE NOT ORTHONORMAL!',
                  la.norm(span - gram_schmidt(span)), '\n', span)
            return
        self.span = span

    @classmethod
    def string(cls, span):
        return span

    def foot_points (self, points):
        tmp = np.einsum('ij,ik,lj->lk', self.span, self.span, points)
        if np.any(la.norm(tmp, axis=1) == 0):
            return None
        return np.einsum('ij,i->ij', tmp, 1/la.norm(tmp, axis=1))

    def distances (self, points):
        feet = self.foot_points(points)
        return np.arccos(np.einsum('ij,ij->i', points, feet).clip(-1, 1))

    def descriptor (self):
        return np.hstack((self.normals, np.einsum('ij,j->i', self.normals,
                                                  self.center)[:,np.newaxis]))

    def __str__ (self):
        if not hasattr(self, 'projection_matrix'):
            self.calculate_projection_matrix()
        return Geodesic.string(self.span)
