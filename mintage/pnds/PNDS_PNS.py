# pns.py

import numpy as np
import numpy.linalg as la
import numpy.random as arandom
from math import sin, acos, pi as PI, sqrt, log, exp
import sys
from scipy.optimize import leastsq, minimize
from scipy.integrate import quad
import scipy.stats as stat

# ==== Required: your own Sphere class ====
from geometry.hypersphere import Sphere, gram_schmidt, EPS

DEG = np.degrees(1)

################################################################################
###########################   Utility Functions   ##############################
################################################################################

def as_matrix(vector):
    """
    Reshape a 1D vector to shape (1, len(vector))
    """
    return np.atleast_2d(vector)

def as_vector(matrix):
    """
    Flatten a matrix to 1D vector.
    """
    return matrix.reshape(-1)

def unfold_points(points, list_spheres):
    """
    Unfold points through a list of spheres (inverse of repeated projections).
    """
    out = points.copy()
    for sphere in reversed(list_spheres):
        out = sphere.unproject(out)
    return out

def fold_points(points, list_spheres):
    """
    Fold points through a list of spheres (repeated projections).
    """
    out = points.copy()
    for sphere in list_spheres:
        out = sphere.project(out)
    return out

################################################################################

def circular_mean(points, perimeter=2*PI):
    """
    Compute circular mean and variance for 1D angular data.
    """
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
    """
    Helper for circular_mean.
    """
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
    """
    Circular mean and summed variance for columns of a matrix.
    """
    mean = []
    variance = 0
    for k in range(data.shape[1]):
        tmp = circular_mean(data[:,k], perimeter)
        mean.append(tmp[0])
        variance += tmp[1]
    return np.array(mean), variance

def normalization(rho, sigma, d, euclidean=False):
    """
    Normalization constant for the likelihood ratio test.
    """
    def f(r): return (exp(-0.5*(r/sigma-rho)**2) + exp(-0.5*(r/sigma+rho)**2))
    try:
        if not euclidean:
            return max(sys.float_info.min, quad(lambda r: sin(r)**(d-1) * f(r), 0, PI)[0])
        else:
            return max(sys.float_info.min, quad(lambda r: r**(d-1) * f(r), 0, (20+rho)*sigma)[0])
    except:
        return max(sys.float_info.min, sqrt(2 * PI) * sigma)

def compare_likelihoods(radii, d, verbose=False, euclidean=False):
    """
    Statistical test to determine if a "great" or "small" sphere is preferred.
    """
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
    chi2 = 1 - stat.chi2.cdf(2 * (likelihood_null(mle_null[0]) - likelihood(mle)), 1)
    if verbose: print("chi2 in likelihood", chi2, "mle", mle, "mle_null", mle_null)
    return chi2 > 0.05

################################################################################
#                              Principal Nested Spheres                         #
################################################################################

class PNS:
    """
    Principal Nested Spheres (PNS) estimator with statistical model selection.

    Parameters
    ----------
    great_until_dim : int
        Use "great sphere" until this dimension.
    max_repetitions : int
        Max attempts for sphere fitting.
    verbose : bool
        Print progress.
    mode : str or None
        If not None, force "great" or "torus" fitting mode.

    Attributes after fit
    --------------------
    spheres_ : list
        Sequence of fitted Sphere objects (one per nesting step).
    points_ : list
        Projected points after each nesting.
    dists_ : list
        List of distances at each step.
    """

    def __init__(self, great_until_dim=2, max_repetitions=10, verbose=False, mode=None, half=False):
        self.great_until_dim = great_until_dim
        self.max_repetitions = max_repetitions
        self.verbose = verbose
        self.mode = mode
        self.half = half

    def fit(self, X):
        """
        Fit PNS to data matrix X (n_samples, n_features).

        After fit, access spheres_, points_, dists_.
        """
        points = np.array(X, dtype=np.float64)
        list_spheres = []
        list_points = []
        list_dists = []

        while points.shape[1] > 2:
            orig_points = points.copy()
            dim = points.shape[1]
            this_mode = self._choose_mode(dim)
            sph = self._get_sphere(points, orig_points, list_spheres, this_mode)
            if sph is None:
                self.spheres_, self.points_, self.dists_ = None, None, None
                return self
            list_spheres.append(sph)
            dists, feet = sph.distances(points, with_feet=True)
            points = sph.project(feet)
            list_points.append(points)
            list_dists.append(dists * DEG)
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
        """
        Selects the mode for sphere fitting at the current dimension.
        """
        if self.mode is not None:
            return self.mode
        return 'great' if dim > self.great_until_dim + 1 else 'torus'

    def _get_sphere(self, points, orig_points, list_spheres, mode):
        """
        Fit a sphere (great/small) with statistical model selection.
        """
        N, d = points.shape

        def max_lin_indep(vectors):
            vectors = np.atleast_2d(vectors)
            out = []
            for v in vectors:
                if la.norm(v) < EPS:
                    continue
                if not out:
                    out.append(v)
                    continue
                tmp = np.vstack(out + [v])
                if la.matrix_rank(tmp, EPS) > len(out):
                    out.append(v)
            return np.array(out)
        N2 = len(max_lin_indep(points))
        if N < d or N2 < d:
            tmp = points.copy()
            tmp = gram_schmidt(tmp)
            n = len(tmp)
            tmp = np.vstack((tmp, 2 * arandom.rand(d - n, d) - 1))
            normals = gram_schmidt(tmp)[n:]
            position = np.zeros(normals.shape[0])
            return Sphere(normals, position)
        # --- Standard fitting: try small/great, do likelihood test
        ext_mean = np.mean(points, axis=0)
        initial = ext_mean / la.norm(ext_mean)

        def f(x):
            norm_x = la.norm(x)
            angles = np.arcsin(np.dot(points, x / norm_x).clip(-1, 1))
            return np.hstack((angles - np.mean(angles), norm_x - 1))

        # Small sphere fit
        result = self._fit_least_squares(f, d, initial)
        if result is None:
            return None
        small_sphere = Sphere(result.reshape(1, -1), np.array([0.]))

        # Optionally: Compare with great sphere using likelihoods
        # radii = np.arccos(np.abs(np.dot(points, result[:-1] / la.norm(result[:-1]))))
        radii = np.arccos(np.abs(np.dot(points, result / la.norm(result))))
        prefer_great = compare_likelihoods(radii, d-1, verbose=self.verbose)
        if prefer_great:
            # Fit the great sphere (center at origin, mean direction)
            initial_great = ext_mean / la.norm(ext_mean)
            def f_great(x):
                return np.arcsin(np.dot(points, x / la.norm(x)).clip(-1, 1))
            great_result = self._fit_least_squares(f_great, d, initial_great, small=False)
            if great_result is not None:
                return Sphere(great_result.reshape(1, -1), np.array([0.]))
        return small_sphere

    def _fit_least_squares(self, f, d, initial, small=True):
        """
        Least squares fitting helper, robust against failures.
        """
        tol = 1e-8
        try:
            initial, exit_code = leastsq(f, initial)
        except Exception as e:
            if self.verbose: print('Exception in fit:', e)
            exit_code = 6
        fails = 0
        counter = 0
        while exit_code > 1 or (small and abs(initial[-1]) > 1):
            fails += 1
            if fails > 3 or exit_code == 6:
                initial = 2 * arandom.rand(d) - 1
                fails = 0
            try:
                initial, exit_code = leastsq(f, initial, ftol=tol, xtol=tol)
            except Exception as e:
                if self.verbose: print('Exception in fit:', e)
                exit_code = 6
            if counter > 20:
                return None
            counter += 1
            tol *= 2
        if small:
            initial[:-1] /= la.norm(initial[:-1])
        else:
            initial /= la.norm(initial)
        return initial

    def _nested_mean(self, points_2d):
        """
        Compute circular mean and residuals for 2D points.
        """
        phis = np.arctan2(points_2d[:, 1], points_2d[:, 0])
        mean = np.arctan2(np.sum(np.sin(phis)), np.sum(np.cos(phis)))
        residuals = ((phis - mean + PI) % (2*PI) - PI) * DEG
        return np.array([np.cos(mean), np.sin(mean)]), residuals


if __name__ == '__main__':
    np.random.seed(0)
    n = 100
    d = 7
    data = np.random.randn(n, d)
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    pns = PNS(great_until_dim=2, max_repetitions=10, verbose=True)
    pns.fit(data)
    print('Finished PNS.\nNumber of spheres:', len(pns.spheres_))
