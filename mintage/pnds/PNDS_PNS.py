# pns_fixed.py

import numpy as np
import numpy.linalg as la
import numpy.random as arandom
from math import sin, acos, pi as PI, sqrt, log, exp
import sys
from scipy.optimize import leastsq, minimize
import scipy.stats as stat
from scipy.integrate import quad

# ==== Required: your own Sphere class ====
from geometry.hypersphere import Sphere, gram_schmidt, EPS

DEG = np.degrees(1)

################################################################################
###########################   Utility Functions   ##############################
################################################################################

def as_matrix(vector):
    return np.atleast_2d(vector)

def as_vector(matrix):
    return matrix.reshape(-1)

def unfold_points(points, list_spheres):
    out = points.copy()
    for sphere in reversed(list_spheres):
        out = sphere.unproject(out)
    return out

def fold_points(points, list_spheres):
    out = points.copy()
    for sphere in list_spheres:
        out = sphere.project(out)
    return out

################################################################################

def circular_mean(points, perimeter=2*PI):
    data = 2 * PI * points / perimeter
    mean0 = np.mean(data)
    var0 = np.var(data)
    sorted_points = np.sort(data)
    means = _variances(mean0, var0, data.size, sorted_points)
    tmp = means[np.argmin(means[:, 1])]
    tmp *= 0.5 * perimeter / PI
    tmp[1] *= 0.5 * perimeter / PI
    return tmp

def _variances(mean0, var0, n, points):
    means = (mean0 + np.linspace(0, 2 * PI, n, endpoint=False)) % (2 * PI)
    means[means >= PI] -= 2 * PI
    m_plus = means >= 0
    lo_sums = np.cumsum(points)
    hi_sums = lo_sums[-1] - lo_sums
    i = np.arange(n)
    j = i[1:]
    p2 = hi_sums[:-1] / (n-j)
    p2[m_plus[1:]] = (lo_sums[:-1] / j)[m_plus[1:]]
    parts = np.hstack([(np.sum(points) / n) if means[0] < 0 else 0, p2])
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

def normalization(rho, sigma, d, euclidean=False):
    def f(r): return (exp(-0.5*(r/sigma-rho)**2) + exp(-0.5*(r/sigma+rho)**2))
    try:
        if not euclidean:
            return max(sys.float_info.min,
                       __quad_safe(lambda r: sin(r)**(d-1) * f(r), 0, PI))
        else:
            return max(sys.float_info.min,
                       __quad_safe(lambda r: r**(d-1) * f(r), 0, (20+rho)*sigma))
    except:
        return max(sys.float_info.min, sqrt(2 * PI) * sigma)

def __quad_safe(func, a, b):
    return quad(func, a, b)[0]

def compare_likelihoods(radii, d, verbose=False, euclidean=False):
    mean = radii.mean()
    std = radii.std()
    def likelihood(x):
        penalty = 0.
        scale = 2. * x[0] * x[1] / PI
        if (scale > 1) and not euclidean:
            x[0] = 0.5 * PI / x[1]
            penalty = scale
        out = np.sum(log(normalization(x[0], x[1], d, euclidean=euclidean)) +
                     0.5*(radii/x[1]-x[0])**2 -
                     np.log(1 + np.exp(- 2. * x[0] * radii / x[1]))) + penalty
        return out
    def likelihood_alt(x):
        penalty = 0.
        scale = 2. * x[0] * x[1] / PI
        if (scale > 1) and not euclidean:
            x[0] = 0.5 * PI / x[1]
            penalty = scale
        out = np.sum(log(x[1]) + 0.5*(radii/x[1]-x[0])**2 -
                     np.log(1 + np.exp(- 2. * x[0] * radii / x[1]))) + penalty
        return out
    def likelihood_null(x):
        if hasattr(x, '__len__'):
            x = x[0]
        return likelihood(np.array([1, float(x)]))
    mle = minimize(likelihood, np.array([max(mean/std, 1.), std]),
                   method='L-BFGS-B',
                   bounds=((0, PI*1e3), (max(1e-3, 0.25*std), max(10*std, 1e-2)))).x
    alt_mle = minimize(likelihood_alt, np.array([max(mean/std, 1.), std]),
                       method='L-BFGS-B',
                       bounds=((0, PI*1e3), (max(1e-3, 0.25*std), max(10*std, 1e-2)))).x
    if verbose:
        print('Mean: %.3f %.3f' % (mean, std),
              'MLE: %.3f %.3f' % (mle[0]*mle[1], mle[1]),
              'JMD MLE: %.3f %.3f' % (alt_mle[0], alt_mle[1]))
    if mle[0] < 1:
        return True
    mle_null = minimize(likelihood_null, 1,
                        method='L-BFGS-B',
                        bounds=((mle[1], 10*max(std, mle[1])),)).x
    
    likelihood_null_mle_null = likelihood_null(mle_null[0])
    likelihood_mle = likelihood(mle)
    chi2 = 1 - stat.chi2.cdf(2 * (likelihood_null(mle_null[0]) - likelihood(mle)), 1)
    if verbose:
        print("chi2 in likelihood", chi2, "mle", mle, "mle_null", mle_null)
    
    return chi2 > 0.05

################################################################################
#                              Seed / Small sphere helpers                     #
################################################################################

def new_seed(old, d):
    """
    Generate a new direction that is not too close (cosine threshold) to existing ones.
    """
    if len(old) == 0:
        normal = 2 * arandom.rand(d) - 1
        return normal / la.norm(normal)
    out = old[0].copy()
    # ensure angular separation (cosine < 0.7)
    while np.any(np.abs(np.einsum('i,ji->j', out, old)) > 0.7):
        out = 2 * arandom.rand(d) - 1
        norm = la.norm(out)
        if norm < EPS:
            out = np.ones(d) / np.sqrt(d)
        else:
            out /= norm
    return out

################################################################################
#                              Principal Nested Spheres                       #
################################################################################

class PNS:
    """
    Principal Nested Spheres (PNS) estimator with statistical model selection.
    """
    def __init__(self, great_until_dim=2, max_repetitions=10, verbose=False, mode=None, half=False):
        self.great_until_dim = great_until_dim
        self.max_repetitions = max_repetitions
        self.verbose = verbose
        self.mode = mode
        self.half = half

    def fit(self, X):
        points = np.array(X, dtype=np.float64)
        list_spheres = []
        list_points = []
        list_dists = []

        while points.shape[1] > 2:
            orig_points = points.copy()
            dim = points.shape[1]
            this_mode = self._choose_mode(dim)
            sph, current_mode = self._get_sphere(points, orig_points, list_spheres, this_mode)
            if sph is None:
                if self.verbose:
                    print(f"Sphere fitting failed at dimension {dim}, terminating PNS")
                self.spheres_, self.points_, self.dists_ = None, None, None
                return self
            
            # Additional robustness check for the fitted sphere
            try:
                dists, feet = sph.signed_distances(points, with_feet=True)
                mean_dists = np.mean(dists) * DEG
                points = sph.project(feet)
                list_points.append(points)
                list_dists.append(dists * DEG)
            except Exception as e:
                if self.verbose:
                    print(f"Error processing fitted sphere: {e}")
                self.spheres_, self.points_, self.dists_ = None, None, None
                return self
            
            list_spheres.append(sph)

        if points.shape[1] < 2:
            self.spheres_, self.points_, self.dists_ = None, None, None
            return self

        mean, residuals = self._nested_mean(points)
        list_spheres.append(None)
        list_points.append(mean)
        list_dists.append(residuals)
        self.spheres_ = list_spheres
        self.points_ = list_points
        self.dists_ = list_dists

        return self

    def _choose_mode(self, dim):
        if self.mode is not None:
            return self.mode
        # return 'adaptive'
        return 'great' if dim > self.great_until_dim + 1 else 'adaptive'

    # ------------------ great sphere objective ------------------
    def _great_sphere_objective(self, x, points):
        norm_x = la.norm(x)
        if norm_x < EPS:
            return np.ones(len(points)) * 1e6
        normal = x / norm_x
        sph = Sphere(normal.reshape(1, -1), np.array([0.]))
        if len(points) == 0:
            return np.array([])
        # dists = sph.distances(points)
        dists = sph.signed_distances(points)
        # return dists - np.mean(dists)
        return dists
    
    def _fit_great_sphere(self, points):
        d = points.shape[1]
        initial = np.mean(points, axis=0)
        if la.norm(initial) < EPS:
            initial = 2 * arandom.rand(d) - 1
        initial = initial / la.norm(initial)

        result = self._fit_least_squares(
            lambda x: self._great_sphere_objective(x, points),
            d, initial, small=False
        )
        if result is None:
            return None
        
        # Additional robustness check
        if la.norm(result) < EPS:
            if self.verbose:
                print("Great sphere fit resulted in zero norm, trying random initialization")
            # Try with random initialization
            initial = 2 * arandom.rand(d) - 1
            initial = initial / la.norm(initial)
            result = self._fit_least_squares(
                lambda x: self._great_sphere_objective(x, points),
                d, initial, small=False
            )
            if result is None or la.norm(result) < EPS:
                return None
        
        normal = result / la.norm(result)
        return Sphere(normal.reshape(1, -1), np.array([0.]))

    # ------------------ small sphere fitting machinery ------------------
    def _small_sphere_objective(self, x, points):
        """
        Direction with norm constraint.
        """
        norm_x = la.norm(x)
        if norm_x < EPS:
            angles = np.zeros(points.shape[0])
        else:
            angles = np.arcsin(np.einsum('ij,j->i', points, x / norm_x).clip(-1,1))
        residuals = angles - np.mean(angles)
        norm_constraint = norm_x - 1
        return np.hstack((residuals, norm_constraint))

    def _fit_small_sphere(self, points):
        d = points.shape[1]

        def fit_with_height(f, dim, starts):
            tmp = self._fit_least_squares(lambda x: f(x, points), dim, starts[-1], small=False)
            if tmp is None:
                return None
            direction = tmp / la.norm(tmp)
            projections = np.einsum('ij,j->i', points, direction).clip(-1,1)
            try:
                height = sin(np.mean(np.arcsin(projections)))
                # Additional robustness check for height
                if np.isnan(height) or np.isinf(height):
                    if self.verbose:
                        print("Invalid height calculated, skipping this fit")
                    return None
            except (ValueError, RuntimeWarning):
                if self.verbose:
                    print("Error calculating height, skipping this fit")
                return None
            combined = np.hstack((tmp, height))
            if height < 0:
                combined = -combined
            return combined

        def one_small_circle_run(f, dim, starts):
            out = fit_with_height(f, dim, starts)
            attempts = 0
            while out is None and attempts < 10:
                starts[-1] = new_seed(np.array(starts[:-1]), dim)
                out = fit_with_height(f, dim, starts)
                attempts += 1
            return out

        def small_circle_fit():
            ext_mean = np.mean(points, axis=0)
            if la.norm(ext_mean) < EPS:
                ext_mean = 2 * arandom.rand(d) - 1
            starts = [ext_mean / la.norm(ext_mean)]
            results = []
            scores = []

            first = one_small_circle_run(self._small_sphere_objective, d, starts)
            if first is None:
                return None
            results.append(first)
            dir_resid = self._small_sphere_objective(first[:-1], points)[:-1]
            scores.append(np.sum(dir_resid ** 2))

            reruns = min(self.max_repetitions, d + 1)
            for i in range(reruns):
                starts.append(new_seed(np.array(starts), d))
                res = one_small_circle_run(self._small_sphere_objective, d, starts)
                if res is None:
                    continue
                results.append(res)
                dir_resid = self._small_sphere_objective(res[:-1], points)[:-1]
                scores.append(np.sum(dir_resid ** 2))

            if len(scores) == 0:
                return None
            best = results[int(np.argmin(np.array(scores)))]
            return best

        combined = small_circle_fit()
        if combined is None:
            return None
        normal = combined[:-1]
        if la.norm(normal) < EPS:
            return None
        normal = normal / la.norm(normal)
        height = combined[-1]
        height = np.clip(height, -1, 1)
        if height < 0:
            normal = -normal
            height = -height

        if self.verbose:
            print(f"Fitted small sphere with height={height:.3f}")

        return Sphere(normal.reshape(1, -1), np.array([height]))

    def _get_sphere(self, points, orig_points, list_spheres, mode):
        N, d = points.shape

        rank = la.matrix_rank(points, EPS)
        if N < d or rank < d:
            tmp = gram_schmidt(points)
            n = len(tmp)
            if self.verbose:
                print(f'Degenerate case: {N} points in {d}D, rank {rank}. Making S^{n-1}')
            tmp = np.vstack((tmp, 2 * arandom.rand(d - n, d) - 1))
            normals = gram_schmidt(tmp)[n:]
            position = np.zeros(normals.shape[0])
            return Sphere(normals, position), "degenerate"

        if mode == 'great':
            sphere = self._fit_great_sphere(points)
            if sphere is None:
                if self.verbose:
                    print("Great sphere fit failed, trying small sphere")
                sphere = self._fit_small_sphere(points)
                if sphere is None:
                    if self.verbose:
                        print("Both great and small sphere fits failed")
                    return None, "failed"
                return sphere, "small"
            return sphere, "great"
        elif mode == 'small':
            sphere = self._fit_small_sphere(points)
            if sphere is None:
                if self.verbose:
                    print("Small sphere fit failed, trying great sphere")
                sphere = self._fit_great_sphere(points)
                if sphere is None:
                    if self.verbose:
                        print("Both small and great sphere fits failed")
                    return None, "failed"
                return sphere, "great"
            return sphere, "small"
        elif mode == 'adaptive':
            # Try small sphere first
            small_sphere = self._fit_small_sphere(points)
            if small_sphere is None:
                if self.verbose:
                    print("Small sphere fit failed in adaptive mode, trying great sphere")
                great_sphere = self._fit_great_sphere(points)
                if great_sphere is None:
                    if self.verbose:
                        print("Both sphere fits failed in adaptive mode")
                    return None, "failed"
                return great_sphere, "great"

            # Small sphere fit succeeded, now check if we should use it
            normal = small_sphere.normals[0]
            height = small_sphere.position[0]
            projections = np.dot(points, normal)
            radii = np.arccos(np.abs(projections).clip(0, 1))
            prefer_great = compare_likelihoods(radii, d-1, verbose=self.verbose)
            if prefer_great:
                if self.verbose:
                    print(f"Statistical test prefers great sphere (height={height:.3f})")
                great_sphere = self._fit_great_sphere(points)
                if great_sphere is None:
                    if self.verbose:
                        print("Great sphere fit failed after statistical test, using small sphere")
                    return small_sphere, "small"
                return great_sphere, "great"
            else:
                if self.verbose:
                    print(f"Statistical test prefers small sphere (height={height:.3f})")
                return small_sphere, "small"
        else:
            return self._get_sphere(points, orig_points, list_spheres, 'adaptive')


    def _fit_least_squares(self, f, param_dim, initial, small=True):
        tol = 1e-8
        try:
            result, exit_code = leastsq(f, initial)
        except Exception as e:
            if self.verbose:
                print('Exception in fit:', e)
            exit_code = 6
            result = initial.copy()

        fails = 0
        counter = 0

        while exit_code > 1 or (small and param_dim > len(initial) - 1 and abs(result[-1]) > 1):
            fails += 1
            if fails > 3 or exit_code == 6:
                if small and param_dim > len(initial) - 1:
                    initial = np.hstack([
                        2 * arandom.rand(param_dim-1) - 1,
                        0.5 * (2 * arandom.rand() - 1)
                    ])
                else:
                    initial = 2 * arandom.rand(param_dim) - 1
                fails = 0
            try:
                result, exit_code = leastsq(f, initial, ftol=tol, xtol=tol)
            except Exception as e:
                if self.verbose:
                    print('Exception in fit:', e)
                exit_code = 6
            if counter > 20:
                return None
            counter += 1
            tol *= 2

        return result

    def _nested_mean(self, points_2d):
        phis = np.arctan2(points_2d[:, 1], points_2d[:, 0])
        mean_angle = np.arctan2(np.sum(np.sin(phis)), np.sum(np.cos(phis)))
        residuals = ((phis - mean_angle + PI) % (2*PI) - PI) * DEG
        return np.array([np.cos(mean_angle), np.sin(mean_angle)]), residuals


if __name__ == '__main__':
    np.random.seed(0)
    n = 100
    d = 7
    data = np.random.randn(n, d)
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    pns = PNS(great_until_dim=2, max_repetitions=10, verbose=True)
    pns.fit(data)
    print('Finished PNS.\nNumber of spheres:', len(pns.spheres_))
