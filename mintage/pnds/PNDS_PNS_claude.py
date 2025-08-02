# pns.py

import numpy as np
import numpy.linalg as la
import numpy.random as arandom
from math import sin, pi as PI, sqrt, log, exp
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

    def fit(self, data_matrix):
        """
        Fit Principal Nested Spheres to data matrix.
        
        Parameters
        ----------
        data_matrix : array-like, shape (n_samples, n_features)
            Input data matrix to fit PNS to.
            
        Returns
        -------
        self : PNS
            Returns self with fitted spheres_, points_, and dists_ attributes.
        """
        current_points = np.array(data_matrix, dtype=np.float64)
        fitted_spheres = []
        projected_points = []
        distances_at_each_level = []

        # Iteratively fit spheres until we reach 2D
        while current_points.shape[1] > 2:
            original_points = current_points.copy()
            current_dimension = current_points.shape[1]
            fitting_mode = self._choose_fitting_mode(current_dimension)
            
            sphere = self._fit_sphere_at_dimension(
                current_points, original_points, fitted_spheres, fitting_mode
            )
            
            if sphere is None:
                self._reset_fitting_results()
                return self
                
            fitted_spheres.append(sphere)
            
            # Project points onto the fitted sphere
            try:
                distances, foot_points = sphere.signed_distances(current_points, with_feet=True)
            except Exception as e:
                print(f"Error in sphere projection: {e}")
                print(f"Points shape: {current_points.shape}")
                print(f"Sphere normals shape: {sphere.normals.shape}")
                self._reset_fitting_results()
                return self
                
            current_points = sphere.project(foot_points)
            projected_points.append(current_points)
            distances_at_each_level.append(distances * DEG)
            
        # Handle final 2D case
        if current_points.shape[1] < 2:
            self._reset_fitting_results()
            return self
            
        final_mean, final_residuals = self._compute_circular_mean_2d(current_points)
        fitted_spheres.append(None)  # No sphere for final 2D projection
        projected_points.append(final_mean)
        distances_at_each_level.append(final_residuals)
        
        # Store results
        self.spheres_ = fitted_spheres
        self.points_ = projected_points
        self.dists_ = distances_at_each_level
        return self

    def _choose_fitting_mode(self, dimension):
        """
        Choose the appropriate fitting mode based on dimension.
        
        Parameters
        ----------
        dimension : int
            Current dimension of the data.
            
        Returns
        -------
        str
            Fitting mode: 'great', 'torus', or 'scale'.
        """
        if self.mode is not None:
            return self.mode
        return 'great' if dimension > self.great_until_dim + 1 else 'torus'

    def _fit_sphere_at_dimension(self, points, original_points, previous_spheres, fitting_mode):
        """
        Fit a sphere at the current dimension using the specified mode.
        
        Parameters
        ----------
        points : array, shape (n_samples, n_features)
            Current points to fit sphere to.
        original_points : array, shape (n_samples, n_features) 
            Original points before any transformations.
        previous_spheres : list
            List of previously fitted spheres.
        fitting_mode : str
            Fitting mode: 'great', 'torus', or 'scale'.
            
        Returns
        -------
        Sphere or None
            Fitted sphere object, or None if fitting failed.
        """
        n_samples, current_dim = points.shape
        if self.verbose:
            print(f'Fitting sphere at dimension {current_dim} using mode: {fitting_mode}', flush=True)
        
        def find_max_linearly_independent_vectors(vectors):
            """Find maximal set of linearly independent vectors."""
            vectors = np.atleast_2d(vectors)
            independent_vectors = []
            for vector in vectors:
                if la.norm(vector) < EPS:
                    continue
                if not independent_vectors:
                    independent_vectors.append(vector)
                    continue
                test_matrix = np.vstack(independent_vectors + [vector])
                if la.matrix_rank(test_matrix, EPS) > len(independent_vectors):
                    independent_vectors.append(vector)
            return np.array(independent_vectors)
        
        n_independent = len(find_max_linearly_independent_vectors(points))
        
        # Special case: insufficient data for full-dimensional fitting
        if n_samples < current_dim or n_independent < current_dim:
            return self._handle_insufficient_data_case(
                points, n_samples, current_dim, n_independent
            )
        
        # Perform sphere fitting based on the chosen mode
        return self._perform_sphere_fitting(
            points, previous_spheres, fitting_mode, current_dim
        )

    def _create_sphere_objective_functions(self, points, previous_spheres):
        """
        Get the optimization functions for different sphere types.
        Adapted from the working older implementation.
        """
        def sphere2torus(data, half=False):
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
        
        def torus_dists(p, q, half=False):
            d = np.abs(sphere2torus(p, half) - sphere2torus(q, half))
            d[d > 180] -= 360
            return la.norm(d, axis=1)
        
        # Small sphere normal, spherical distance, radius not included in output!
        def spherical_distance_objective(x):
            norm_x = la.norm(x)
            angles = np.arcsin(np.dot(points, x / norm_x).clip(-1,1))
            return np.hstack((angles - np.mean(angles), norm_x - 1))
        
        # Great sphere normal, spherical distance, radius 0 not included in output!
        def great_sphere_objective(x):
            return np.arcsin(np.dot(points, x / la.norm(x)).clip(-1,1))
        
        # Small sphere, torus distance, radius included in output
        def torus_distance_objective(x):
            if abs(x[-1]) > 1:
                if self.verbose:
                    print('Fail:', x[-1])
                return 180 * np.ones(points.shape[0]) * abs(x[-1])
            sphere = Sphere(x[:-1].reshape(1, -1), np.array([x[-1]]))
            feet = sphere.foot_points(points)
            if len(previous_spheres) > 0:
                return torus_dists(unfold_points(feet, previous_spheres),
                                 unfold_points(points, previous_spheres), self.half)
            return torus_dists(feet, points, self.half)
        
        # Great sphere normal, torus distance, radius 0 not included in output!
        def great_torus_objective(x):
            sphere = Sphere(x.reshape(1, -1), np.array([0.]))
            feet = sphere.foot_points(points)
            if len(previous_spheres) > 0:
                return torus_dists(unfold_points(feet, previous_spheres),
                                 unfold_points(points, previous_spheres), self.half)
            return torus_dists(feet, points, self.half)
        
        return {
            'spherical_distance': spherical_distance_objective,
            'great_sphere_distance': great_sphere_objective, 
            'torus_distance': torus_distance_objective,
            'great_torus_distance': great_torus_objective
        }
    
    def _generate_well_separated_seed(self, existing_seeds, dimension):
        """
        Generate a random seed that is well-separated from existing seeds.
        
        Parameters
        ----------
        existing_seeds : array
            Previously generated seed vectors.
        dimension : int
            Dimension of the seed vector to generate.
            
        Returns
        -------
        array
            New seed vector that is sufficiently different from existing ones.
        """
        if len(existing_seeds) <= 0:
            random_vector = 2 * arandom.rand(dimension) - 1
            return random_vector / la.norm(random_vector)
        
        candidate = existing_seeds[0]
        # Keep generating until we find a vector with >45° separation from all existing
        while np.any(np.abs(np.dot(candidate, existing_seeds.T)) > 0.7):  # cos(45°)
            candidate = 2 * arandom.rand(dimension) - 1
            candidate /= la.norm(candidate)
        return candidate
    
    def _fit_sphere_normal_then_add_height(self, objective_func, dimension, initial_direction, points):
        """
        Fit sphere normal direction first, then compute and add the height parameter.
        
        This is a two-stage process: first optimize the normal direction, 
        then compute the optimal radius/height for that direction.
        """
        normal_direction = self._fit_objective_function(objective_func, dimension, initial_direction)
        if normal_direction is None:
            return None
        
        # Compute optimal height/radius for this normal direction
        signed_distances = np.dot(points, normal_direction / la.norm(normal_direction)).clip(-1, 1)
        mean_signed_distance = np.mean(np.arcsin(signed_distances))
        height = sin(mean_signed_distance)
        
        # Ensure consistent orientation
        sign = 1 if height >= 0 else -1
        return sign * np.hstack((normal_direction, height))
    
    def _fit_single_small_sphere_attempt(self, objective_func, dimension, seed_list, points):
        """
        Single attempt at fitting a small sphere with automatic seed regeneration.
        
        If the initial seed fails, automatically generates a new well-separated seed.
        """
        result = self._fit_sphere_normal_then_add_height(
            objective_func, dimension, seed_list[-1], points
        )
        while result is None:
            seed_list[-1] = self._generate_well_separated_seed(
                np.array(seed_list[:-1]), dimension
            )
            result = self._fit_sphere_normal_then_add_height(
                objective_func, dimension, seed_list[-1], points
            )
        return result
    
    def _fit_small_sphere_multistart(self, objective_func, dimension, points):
        """
        Fit small sphere using multiple starting points to avoid local minima.
        
        Uses multiple well-separated initial seeds and selects the best result.
        """
        # Start with the data centroid as first seed
        data_centroid = np.mean(points, axis=0)
        seed_list = [data_centroid / la.norm(data_centroid)]
        
        # First attempt
        results = [self._fit_single_small_sphere_attempt(
            objective_func, dimension, seed_list, points
        )]
        scores = [np.sum(objective_func(results[-1][:-1])**2)]
        
        # Additional attempts with well-separated seeds
        n_attempts = min(self.max_repetitions, dimension + 1)
        for _ in range(n_attempts):
            seed_list.append(self._generate_well_separated_seed(
                np.array(seed_list), dimension
            ))
            results.append(self._fit_single_small_sphere_attempt(
                objective_func, dimension, seed_list, points
            ))
            scores.append(np.sum(objective_func(results[-1][:-1])**2))
        
        # Return the best result
        best_index = np.argmin(np.array(scores))
        return results[best_index]
    
    def _fit_objective_function(self, objective_func, dimension, initial_guess):
        """
        Robust least squares optimization with automatic restarts on failure.
        
        Parameters
        ----------
        objective_func : callable
            Objective function to minimize.
        dimension : int  
            Dimension of the parameter space.
        initial_guess : array
            Initial parameter guess.
            
        Returns
        -------
        array or None
            Optimized parameters, or None if optimization failed.
        """
        tolerance = 1e-8
        max_attempts = 20
        
        try:
            result, exit_code = leastsq(objective_func, initial_guess)
        except Exception as e:
            if self.verbose: 
                print(f'Optimization exception: {e}')
            exit_code = 6
            
        failure_count = 0
        attempt_count = 0
        
        while exit_code > 1 and attempt_count < max_attempts:
            failure_count += 1
            
            # After 3 failures or exceptions, try random restart
            if failure_count > 3 or exit_code == 6:
                result = 2 * arandom.rand(dimension) - 1
                failure_count = 0
                
            try:
                result, exit_code = leastsq(
                    objective_func, result, ftol=tolerance, xtol=tolerance
                )
            except Exception as e:
                if self.verbose:
                    print(f'Optimization exception: {e}')
                exit_code = 6
                
            attempt_count += 1
            tolerance *= 2  # Relax tolerance progressively
            
        if attempt_count >= max_attempts:
            return None
            
        # Normalize to unit length
        result /= la.norm(result)
        return result

    def _compute_circular_mean_2d(self, points_2d):
        """
        Compute circular mean and residuals for 2D points on a circle.
        
        Parameters
        ----------
        points_2d : array, shape (n_samples, 2)
            2D points on the unit circle.
            
        Returns
        -------
        mean_direction : array, shape (2,)
            Unit vector representing the circular mean direction.
        residuals : array, shape (n_samples,)
            Angular residuals from the mean direction in degrees.
        """
        angles = np.arctan2(points_2d[:, 1], points_2d[:, 0])
        circular_mean_angle = np.arctan2(np.sum(np.sin(angles)), np.sum(np.cos(angles)))
        residuals = ((angles - circular_mean_angle + PI) % (2*PI) - PI) * DEG
        mean_direction = np.array([np.cos(circular_mean_angle), np.sin(circular_mean_angle)])
        return mean_direction, residuals
    
    def _reset_fitting_results(self):
        """Reset fitting results when an error occurs."""
        self.spheres_, self.points_, self.dists_ = None, None, None
    
    def _handle_insufficient_data_case(self, points, n_samples, current_dim, _):
        """Handle cases where there's insufficient data for full-dimensional fitting."""
        orthonormal_points = gram_schmidt(points.copy())
        n_orthonormal = len(orthonormal_points)
        
        if self.verbose:
            print(f'{n_samples} points with dimension {current_dim}. Making S^{n_orthonormal}')
        
        padding_vectors = 2 * arandom.rand(current_dim - n_orthonormal, current_dim) - 1
        extended_matrix = np.vstack((orthonormal_points, padding_vectors))
        sphere_normals = gram_schmidt(extended_matrix)[n_orthonormal:]
        
        if sphere_normals.shape[0] == 1:
            return Sphere(sphere_normals, np.array([0.]))
        return Sphere(sphere_normals, np.zeros(sphere_normals.shape[0]))
    
    def _perform_sphere_fitting(self, points, previous_spheres, fitting_mode, current_dim):
        """Perform sphere fitting based on the specified mode."""
        # Get optimization functions for different sphere types  
        objective_functions = self._create_sphere_objective_functions(points, previous_spheres)
        
        # Route to appropriate fitting method based on mode
        if fitting_mode == 'great':
            return self._fit_great_sphere(objective_functions, current_dim)
        elif fitting_mode == 'torus':
            return self._fit_torus_sphere(objective_functions, current_dim, points)
        else:
            return self._fit_small_sphere_with_test(objective_functions, current_dim, points, fitting_mode)
    
    def _fit_great_sphere(self, objective_functions, current_dim):
        """Fit a great sphere (passing through origin)."""
        initial_guess = 2 * arandom.rand(current_dim) - 1
        great_sphere_func = objective_functions['great_sphere_distance']
        
        result = self._fit_objective_function(great_sphere_func, current_dim, initial_guess)
        if result is None:
            return None
        return Sphere(result.reshape(1, -1), np.array([0.]))
    
    def _fit_torus_sphere(self, objective_functions, current_dim, points):
        """Fit sphere optimized for torus distance."""
        spherical_func = objective_functions['spherical_distance']
        torus_func = objective_functions['torus_distance']
        
        # First fit using spherical distance
        small_sphere_params = self._fit_small_sphere_multistart(spherical_func, current_dim, points)
        if small_sphere_params is None:
            return None
            
        if self.verbose:
            torus_error = la.norm(torus_func(small_sphere_params))
            spherical_error = la.norm(spherical_func(small_sphere_params[:-1]))
            print(f"Initial fit - Torus error: {torus_error:.3f}, Spherical error: {spherical_error:.3f}")
        
        # Optimize using torus distance
        optimized_params = self._fit_objective_function(
            torus_func, current_dim + 1, small_sphere_params
        )
        
        if optimized_params is None:
            print('WARNING: Torus optimization failed! Using spherical fit.')
            optimized_params = small_sphere_params
        elif self.verbose:
            torus_error = la.norm(torus_func(optimized_params))
            spherical_error = la.norm(spherical_func(optimized_params[:-1]))
            print(f"Final fit - Torus error: {torus_error:.3f}, Spherical error: {spherical_error:.3f}")
        
        return Sphere(optimized_params[:-1].reshape(1, -1), np.array([optimized_params[-1]]))
    
    def _fit_small_sphere_with_test(self, objective_functions, current_dim, points, fitting_mode):
        """Fit small sphere with statistical test for great vs small preference."""  
        spherical_func = objective_functions['spherical_distance']
        
        # Fit small sphere
        small_sphere_params = self._fit_small_sphere_multistart(spherical_func, current_dim, points)
        if small_sphere_params is None:
            return None
        
        # Statistical test: should we prefer a great sphere instead?
        if fitting_mode in ['scale', 'torus']:
            sphere_normal = small_sphere_params[:-1]
            radii_to_sphere = np.arccos(np.abs(np.dot(points, sphere_normal)))
            
            if compare_likelihoods(radii_to_sphere, current_dim - 1, self.verbose):
                # Great sphere is statistically preferred
                if self.verbose:
                    radius_val = abs(small_sphere_params[-1])
                    print(f"Statistical test prefers great sphere (radius={radius_val:.3f})")
                
                if fitting_mode == 'torus':
                    great_func = objective_functions['great_torus_distance']
                else:
                    great_func = objective_functions['great_sphere_distance']
                
                great_result = self._fit_objective_function(great_func, current_dim, sphere_normal)
                if great_result is not None:
                    angle_change = np.arccos(np.dot(great_result, sphere_normal)) * DEG
                    if self.verbose:
                        print(f'Direction changed by {angle_change:.1f}\u00b0')
                    return Sphere(great_result.reshape(1, -1), np.array([0.]))
        
        # Use the small sphere
        return Sphere(small_sphere_params[:-1].reshape(1, -1), np.array([small_sphere_params[-1]]))


if __name__ == '__main__':
    np.random.seed(0)
    n = 100
    d = 7
    data = np.random.randn(n, d)
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    pns = PNS(great_until_dim=2, max_repetitions=10, verbose=True)
    pns.fit(data)
    print('Finished PNS.\nNumber of spheres:', len(pns.spheres_))
