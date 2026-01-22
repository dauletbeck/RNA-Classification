"""
Spherical Data Generation and Projections

This module provides tools to generate synthetic spherical data and apply
exponential map and PNS residual projections. Both projections output
(theta, phi) coordinates in radians for stress calculation.

Projections:
- Exponential Map: Projects to tangent space, outputs [theta, phi] in radians
  where theta is geodesic distance and phi is direction angle
- PNS Residuals: Uses Principal Nested Spheres residual distances directly,
  outputs [theta, phi] in radians from dists_[-2] and dists_[-1]
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from pnds.PNDS_PNS import PNS


def exponential_map(V, p):
    """
    Apply exponential map to project tangent space points onto the sphere.

    Maps points from the tangent space at p back onto the unit sphere.

    Parameters
    ----------
    V : np.ndarray
        Point cloud in tangent space, shape (N, m) where m is tangent space dimension
    p : np.ndarray
        Point of tangency on the sphere, shape (m+1,)

    Returns
    -------
    np.ndarray
        Points on the sphere, shape (N, m+1)
    """
    N, M = V.shape[0], V.shape[1]
    V_mean = V.mean(axis=0)
    # check if points are centered at 0, center them
    V -= V_mean
    V = np.column_stack([V, np.zeros(N)])
    V_norm = np.linalg.norm(V, axis=1)[:, None]

    return np.cos(V_norm) * p + np.sin(V_norm) * (V / V_norm)


class SphericalDataGenerator:
    """Generate synthetic data on unit sphere with anisotropic distribution."""

    def __init__(self, dim: int = 3, random_seed: Optional[int] = None):
        """
        Initialize the spherical data generator.

        Parameters
        ----------
        dim : int
            Dimension of the ambient space (sphere is (dim-1)-dimensional)
        random_seed : int, optional
            Random seed for reproducibility
        """
        self.dim = dim
        if random_seed is not None:
            np.random.seed(random_seed)

    def generate_von_mises_fisher(
        self,
        n_samples: int,
        mean_direction: np.ndarray,
        kappa: float
    ) -> np.ndarray:
        """
        Generate samples from von Mises-Fisher distribution.

        The von Mises-Fisher distribution is a directional distribution on the
        unit sphere, parametrized by a mean direction and concentration parameter.
        Higher kappa means more concentrated around the mean (more anisotropic).

        Parameters
        ----------
        n_samples : int
            Number of samples to generate
        mean_direction : np.ndarray
            Mean direction vector (will be normalized to unit length)
        kappa : float
            Concentration parameter (kappa > 0).
            kappa = 0: uniform on sphere
            kappa >> 0: highly concentrated around mean

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, dim) with points on unit sphere
        """
        # Normalize mean direction
        mu = mean_direction / np.linalg.norm(mean_direction)

        # Generate samples using von Mises-Fisher distribution
        samples = self._sample_vmf(n_samples, mu, kappa)

        return samples

    def _sample_vmf(self, n: int, mu: np.ndarray, kappa: float) -> np.ndarray:
        """
        Sample from von Mises-Fisher distribution using Wood's algorithm.

        Reference: Wood (1994) "Simulation of the von Mises Fisher distribution"
        """
        dim = len(mu)

        # Sample from vMF using rejection sampling
        result = np.zeros((n, dim))

        for i in range(n):
            # Step 1: Sample w from the marginal distribution
            w = self._sample_w(kappa, dim)

            # Step 2: Sample v uniformly from unit sphere in R^(d-1)
            v = np.random.randn(dim - 1)
            v = v / np.linalg.norm(v)

            # Step 3: Construct the sample
            # x = [sqrt(1-w^2) * v; w]
            sample = np.zeros(dim)
            sample[:-1] = np.sqrt(1 - w**2) * v
            sample[-1] = w

            # Step 4: Rotate to align with mu
            # Find rotation matrix that maps e_d to mu
            result[i] = self._rotate_to_mu(sample, mu)

        return result

    def _sample_w(self, kappa: float, dim: int) -> float:
        """Sample w from the marginal distribution using rejection sampling."""
        d = dim
        b = (d - 1) / (2 * kappa + np.sqrt(4 * kappa**2 + (d - 1)**2))
        x0 = (1 - b) / (1 + b)
        c = kappa * x0 + (d - 1) * np.log(1 - x0**2)

        while True:
            z = np.random.beta((d - 1) / 2, (d - 1) / 2)
            w = (1 - (1 + b) * z) / (1 - (1 - b) * z)
            u = np.random.uniform()

            if kappa * w + (d - 1) * np.log(1 - x0 * w) - c >= np.log(u):
                return w

    def _rotate_to_mu(self, x: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """Rotate vector x to align the last basis vector with mu."""
        dim = len(mu)
        e_d = np.zeros(dim)
        e_d[-1] = 1.0

        # If mu is already aligned with e_d, no rotation needed
        if np.allclose(mu, e_d):
            return x
        if np.allclose(mu, -e_d):
            return -x

        # Use Householder reflection
        u = e_d - mu
        u = u / np.linalg.norm(u)

        # Householder transformation: H = I - 2*u*u^T
        return x - 2 * np.dot(u, x) * u

    def generate_anisotropic_mixture(
        self,
        n_samples: int,
        mean_direction: np.ndarray,
        primary_kappa: float,
        secondary_kappa: float = None,
        mixture_weight: float = 0.8
    ) -> np.ndarray:
        """
        Generate anisotropic mixture of von Mises-Fisher distributions.

        Creates a mixture with a dominant mode (ensuring unique Frechet mean)
        and optionally a secondary mode for added complexity.

        Parameters
        ----------
        n_samples : int
            Total number of samples
        mean_direction : np.ndarray
            Mean direction for primary mode
        primary_kappa : float
            Concentration for primary mode (should be high for anisotropy)
        secondary_kappa : float, optional
            Concentration for secondary mode. If None, only primary mode used.
        mixture_weight : float
            Weight for primary mode (0 to 1)

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, dim) with anisotropic data
        """
        n_primary = int(n_samples * mixture_weight)
        n_secondary = n_samples - n_primary

        # Generate primary mode
        samples_primary = self.generate_von_mises_fisher(
            n_primary, mean_direction, primary_kappa
        )

        if secondary_kappa is not None and n_secondary > 0:
            # Generate secondary mode with perpendicular mean
            secondary_mean = self._get_perpendicular_direction(mean_direction)
            samples_secondary = self.generate_von_mises_fisher(
                n_secondary, secondary_mean, secondary_kappa
            )

            # Combine samples
            samples = np.vstack([samples_primary, samples_secondary])

            # Shuffle
            np.random.shuffle(samples)
        else:
            samples = samples_primary

        return samples

    def _get_perpendicular_direction(self, v: np.ndarray) -> np.ndarray:
        """Get a unit vector perpendicular to v."""
        # Create a random vector
        random_vec = np.random.randn(len(v))

        # Gram-Schmidt orthogonalization
        perp = random_vec - np.dot(random_vec, v) * v
        perp = perp / np.linalg.norm(perp)

        return perp

    def compute_frechet_mean(self, data: np.ndarray, max_iter: int = 100) -> np.ndarray:
        """
        Compute Frechet mean (intrinsic mean) on the sphere.

        The Frechet mean on a sphere is the point that minimizes the sum of
        squared geodesic distances to all data points.

        Parameters
        ----------
        data : np.ndarray
            Array of shape (n_samples, dim) with points on unit sphere
        max_iter : int
            Maximum iterations for optimization

        Returns
        -------
        np.ndarray
            Frechet mean as unit vector
        """
        # Initialize with extrinsic mean
        mean = np.mean(data, axis=0)
        mean = mean / np.linalg.norm(mean)

        # Iterative optimization
        for _ in range(max_iter):
            # Compute tangent vectors at current mean
            tangent_sum = np.zeros_like(mean)

            for point in data:
                # Log map from mean to point
                tangent_sum += self._log_map(mean, point)

            tangent_mean = tangent_sum / len(data)

            # Check convergence
            if np.linalg.norm(tangent_mean) < 1e-8:
                break

            # Exponential map to update mean
            mean = self._exp_map(mean, tangent_mean)

        return mean

    def _log_map(self, base: np.ndarray, point: np.ndarray) -> np.ndarray:
        """
        Logarithmic map from base to point on sphere.

        Maps a point on the sphere to the tangent space at base.
        """
        # Compute geodesic distance
        cos_dist = np.clip(np.dot(base, point), -1.0, 1.0)
        dist = np.arccos(cos_dist)

        # Handle case when points coincide
        if dist < 1e-10:
            return np.zeros_like(base)

        # Tangent vector direction
        direction = point - cos_dist * base
        direction = direction / np.linalg.norm(direction)

        return dist * direction

    def _exp_map(self, base: np.ndarray, tangent: np.ndarray) -> np.ndarray:
        """
        Exponential map from tangent space at base to sphere.

        Maps a tangent vector at base to a point on the sphere.
        """
        norm_tangent = np.linalg.norm(tangent)

        # Handle zero tangent vector
        if norm_tangent < 1e-10:
            return base

        # Exponential map formula
        return np.cos(norm_tangent) * base + np.sin(norm_tangent) * (tangent / norm_tangent)


class SphericalProjections:
    """Apply various projections to spherical data."""

    def __init__(self, data: np.ndarray):
        """
        Initialize with spherical data.

        Parameters
        ----------
        data : np.ndarray
            Array of shape (n_samples, dim) with points on unit sphere
        """
        self.data = data
        self.dim = data.shape[1]
        self._geodesic_dists = None  # Cache for geodesic distances

    def _compute_geodesic_distances(self) -> np.ndarray:
        """
        Compute pairwise geodesic distances on the sphere.

        Returns
        -------
        np.ndarray
            Symmetric distance matrix of shape (n_samples, n_samples)
        """
        if self._geodesic_dists is not None:
            return self._geodesic_dists

        n_samples = len(self.data)
        geodesic_dists = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                cos_dist = np.clip(np.dot(self.data[i], self.data[j]), -1.0, 1.0)
                geodesic_dists[i, j] = np.arccos(cos_dist)
                geodesic_dists[j, i] = geodesic_dists[i, j]

        self._geodesic_dists = geodesic_dists
        return geodesic_dists

    def cartesian_to_polar(self, cartesian_coords: np.ndarray) -> np.ndarray:
        """
        Convert Cartesian coordinates to polar coordinates.

        Parameters
        ----------
        cartesian_coords : np.ndarray
            Array of shape (n_samples, dim) with Cartesian coordinates

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, dim) with polar coordinates
            [r, theta] for 2D, [r, theta, phi] for 3D
        """
        n_samples, dim = cartesian_coords.shape
        polar_coords = np.zeros_like(cartesian_coords)

        for i, point in enumerate(cartesian_coords):
            # Radius
            r = np.linalg.norm(point)
            polar_coords[i, 0] = r

            if dim == 2:
                # 2D: (r, theta)
                if r > 1e-10:
                    theta = np.arctan2(point[1], point[0])
                    polar_coords[i, 1] = theta
            elif dim == 3:
                # 3D: (r, theta, phi)
                if r > 1e-10:
                    theta = np.arctan2(point[1], point[0])
                    phi = np.arccos(np.clip(point[2] / r, -1.0, 1.0))
                    polar_coords[i, 1] = theta
                    polar_coords[i, 2] = phi

        return polar_coords

    def compute_stress(
        self,
        projected: np.ndarray,
        coordinate_system: str = 'cartesian'
    ) -> dict:
        """
        Compute stress metric for the projection.

        Stress measures how well distances are preserved:
        Stress = sqrt(Σ(d_geo - d_proj)² / Σ(d_geo)²)

        Parameters
        ----------
        projected : np.ndarray
            Projected coordinates
        coordinate_system : str
            'cartesian' or 'polar' - how to compute distances in projection

        Returns
        -------
        dict
            Dictionary with stress metrics and distance information
        """
        # Get geodesic distances on sphere
        geodesic_dists = self._compute_geodesic_distances()
        n_samples = len(self.data)

        # Convert to appropriate coordinate system
        if coordinate_system == 'polar':
            coords = self.cartesian_to_polar(projected)
        else:
            coords = projected

        # Compute pairwise distances in projected space
        projected_dists = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                # Euclidean distance in the coordinate space
                projected_dists[i, j] = np.linalg.norm(coords[i] - coords[j])
                projected_dists[j, i] = projected_dists[i, j]

        # Compute stress (Kruskal's Stress-1)
        diff = geodesic_dists - projected_dists
        numerator = np.sum(diff ** 2)
        denominator = np.sum(geodesic_dists ** 2)
        stress = np.sqrt(numerator / denominator)

        # Compute Sammon's stress (weighted by original distances)
        # Avoid division by zero
        mask = geodesic_dists > 1e-10
        sammon_numerator = np.sum((diff[mask] ** 2) / geodesic_dists[mask])
        sammon_denominator = np.sum(geodesic_dists[mask])
        sammon_stress = sammon_numerator / sammon_denominator

        # Additional metrics
        relative_errors = np.abs(diff[mask]) / geodesic_dists[mask]
        mean_relative_error = np.mean(relative_errors)
        max_relative_error = np.max(relative_errors)

        return {
            'stress': stress,
            'sammon_stress': sammon_stress,
            'mean_relative_error': mean_relative_error,
            'max_relative_error': max_relative_error,
            'geodesic_dists': geodesic_dists,
            'projected_dists': projected_dists,
            'coordinate_system': coordinate_system
        }

    def exponential_map_projection(self, base_point: np.ndarray) -> np.ndarray:
        """
        Project data using exponential map at base point.

        Returns (theta, phi) coordinates in radians where:
        - theta: geodesic distance from base point (radians)
        - phi: angular direction in tangent plane (radians)

        Parameters
        ----------
        base_point : np.ndarray
            Base point on sphere for exponential map

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, 2) with [theta, phi] in radians
        """
        base_point = base_point / np.linalg.norm(base_point)

        projected = np.zeros_like(self.data)

        for i, point in enumerate(self.data):
            projected[i] = self._log_map(base_point, point)

        # Convert to (theta, phi) in radians
        # theta = geodesic distance from base point
        # phi = angular direction in tangent plane
        theta = np.linalg.norm(projected[:, :2], axis=1)  # geodesic distance in radians
        phi = np.arctan2(projected[:, 1], projected[:, 0])  # direction angle in radians

        return np.column_stack([theta, phi])

    def _log_map(self, base: np.ndarray, point: np.ndarray) -> np.ndarray:
        """Logarithmic map from base to point."""
        cos_dist = np.clip(np.dot(base, point), -1.0, 1.0)
        dist = np.arccos(cos_dist)

        if dist < 1e-10:
            return np.zeros_like(base)

        direction = point - cos_dist * base
        direction = direction / np.linalg.norm(direction)

        return dist * direction

    def pns_residual_projection(self, verbose: bool = False) -> Tuple[np.ndarray, PNS]:
        """
        Project data using PNS residual angles.

        Fits PNS to get residual distances at each level, returns them directly
        as (theta, phi) coordinates in radians.

        Parameters
        ----------
        verbose : bool
            Whether to print PNS fitting details

        Returns
        -------
        coords : np.ndarray
            Array of shape (n_samples, 2) with [theta, phi] in radians
            - theta: residual distance from first sphere (dists_[-2])
            - phi: residual distance from second sphere (dists_[-1])
        pns : PNS
            Fitted PNS object
        """
        # Fit PNS to get residual angles
        pns = PNS(mode='great', verbose=verbose)
        pns.fit(self.data)

        # Extract residual angles (in degrees) - use them directly
        if isinstance(pns.dists_, list) and len(pns.dists_) >= 2:
            theta_deg = pns.dists_[-2]  # Second-to-last dimension
            phi_deg = pns.dists_[-1]     # Last dimension (residuals)
        else:
            # Fallback: return zeros
            return np.zeros((len(self.data), 2)), pns

        # Convert degrees to radians (no flipping, use as-is)
        theta = np.radians(theta_deg)
        phi = np.radians(phi_deg)

        # Return as [theta, phi] in radians
        coords = np.column_stack([theta, phi])

        return coords, pns


    def _align_with_axis(self, data: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Rotate data to align direction with last coordinate axis."""
        dim = len(direction)
        e_d = np.zeros(dim)
        e_d[-1] = 1.0

        direction = direction / np.linalg.norm(direction)

        if np.allclose(direction, e_d):
            return data

        # Householder reflection
        u = e_d - direction
        u = u / np.linalg.norm(u)

        # Apply to all data points
        return data - 2 * np.outer(data @ u, u)

    def apply_exponential_map(
        self,
        tangent_coords: np.ndarray,
        base_point: np.ndarray
    ) -> np.ndarray:
        """
        Apply exponential map to project tangent space coordinates back to sphere.

        Uses the standalone exponential_map function to map 2D tangent space
        coordinates back onto the sphere.

        Parameters
        ----------
        tangent_coords : np.ndarray
            Array of shape (n_samples, 2) with tangent space coordinates
        base_point : np.ndarray
            Point of tangency on the sphere, shape (3,)

        Returns
        -------
        np.ndarray
            Points on the sphere, shape (n_samples, 3)
        """
        base_point = base_point / np.linalg.norm(base_point)
        return exponential_map(tangent_coords, base_point)

    def compute_reconstruction_error(
        self,
        reconstructed: np.ndarray
    ) -> dict:
        """
        Compute reconstruction error after round-trip projection.

        Measures how well the original spherical data is reconstructed
        after projecting to tangent space and back.

        Parameters
        ----------
        reconstructed : np.ndarray
            Reconstructed points on sphere after round-trip

        Returns
        -------
        dict
            Dictionary with reconstruction error metrics
        """
        # Normalize reconstructed points to unit sphere
        reconstructed_norm = reconstructed / np.linalg.norm(reconstructed, axis=1, keepdims=True)

        # Compute geodesic distances between original and reconstructed points
        n_samples = len(self.data)
        geodesic_errors = np.zeros(n_samples)

        for i in range(n_samples):
            cos_dist = np.clip(np.dot(self.data[i], reconstructed_norm[i]), -1.0, 1.0)
            geodesic_errors[i] = np.arccos(cos_dist)

        # Convert to degrees for interpretability
        geodesic_errors_deg = np.degrees(geodesic_errors)

        return {
            'mean_error_rad': np.mean(geodesic_errors),
            'max_error_rad': np.max(geodesic_errors),
            'std_error_rad': np.std(geodesic_errors),
            'mean_error_deg': np.mean(geodesic_errors_deg),
            'max_error_deg': np.max(geodesic_errors_deg),
            'std_error_deg': np.std(geodesic_errors_deg),
            'geodesic_errors': geodesic_errors,
            'reconstructed_points': reconstructed_norm
        }


def example_usage():
    """Demonstrate basic usage of the module."""

    print("="*60)
    print("SPHERICAL DATA GENERATION AND PROJECTIONS")
    print("="*60)

    # Generate synthetic data
    print("\n1. Generating synthetic spherical data...")
    generator = SphericalDataGenerator(dim=3, random_seed=42)

    # Original direction with y-axis flipped (toward viewer)
    mean_direction = np.array([0.5, -0.5, np.sqrt(0.5)])
    mean_direction = mean_direction / np.linalg.norm(mean_direction)

    data = generator.generate_anisotropic_mixture(
        n_samples=200,
        mean_direction=mean_direction,
        primary_kappa=10.0,
        secondary_kappa=2.0,
        mixture_weight=0.85
    )
    print(f"   Generated {len(data)} points on S^2")

    # Compute Frechet mean
    frechet_mean = generator.compute_frechet_mean(data)
    print(f"\n2. Frechet mean: {frechet_mean}")
    print(f"   Angular distance from true mean: "
          f"{np.arccos(np.clip(np.dot(frechet_mean, mean_direction), -1, 1)) * 180/np.pi:.2f}°")

    # Apply projections (both output (theta, phi) in radians)
    print("\n3. Applying projections...")
    projector = SphericalProjections(data)

    # Exponential map - (theta, phi) in radians
    exp_coords = projector.exponential_map_projection(frechet_mean)
    print(f"   Exponential map: {exp_coords.shape} [theta, phi] in radians")

    # PNS residual projection - (theta, phi) in radians
    pns_coords, pns_object = projector.pns_residual_projection(verbose=False)
    print(f"   PNS residuals: {pns_coords.shape} [theta, phi] in radians")

    # Round-trip verification: apply exponential map to go back to sphere
    print("\n4. Round-trip verification (tangent -> sphere via exp map)...")
    reconstructed = projector.apply_exponential_map(exp_coords, frechet_mean)
    reconstruction_error = projector.compute_reconstruction_error(reconstructed)
    print(f"   Mean reconstruction error: {reconstruction_error['mean_error_deg']:.4f}°")
    print(f"   Max reconstruction error:  {reconstruction_error['max_error_deg']:.4f}°")
    print(f"   Std reconstruction error:  {reconstruction_error['std_error_deg']:.4f}°")

    # Compute stress metrics
    print("\n5. Computing stress metrics...")
    print("   " + "="*56)

    # Exponential map stress
    exp_stress = projector.compute_stress(exp_coords, coordinate_system='cartesian')
    print(f"\n   Exponential Map (θ, φ):")
    print(f"      Stress:              {exp_stress['stress']:.4f}")
    print(f"      Sammon Stress:       {exp_stress['sammon_stress']:.4f}")
    print(f"      Mean Rel. Error:     {exp_stress['mean_relative_error']:.4f}")
    print(f"      Max Rel. Error:      {exp_stress['max_relative_error']:.4f}")

    # PNS residuals stress
    pns_stress = projector.compute_stress(pns_coords, coordinate_system='cartesian')
    print(f"\n   PNS Residuals (θ, φ):")
    print(f"      Stress:              {pns_stress['stress']:.4f}")
    print(f"      Sammon Stress:       {pns_stress['sammon_stress']:.4f}")
    print(f"      Mean Rel. Error:     {pns_stress['mean_relative_error']:.4f}")
    print(f"      Max Rel. Error:      {pns_stress['max_relative_error']:.4f}")

    print("\n   " + "="*56)
    print("\n   Interpretation:")
    print("   - Stress = 0:      Perfect distance preservation")
    print("   - Stress < 0.05:   Excellent")
    print("   - Stress < 0.10:   Good")
    print("   - Stress < 0.20:   Fair")
    print("   - Stress ≥ 0.20:   Poor")

    # Determine best method
    stresses = {
        'Exponential Map': exp_stress['stress'],
        'PNS Residuals': pns_stress['stress']
    }
    best_method = min(stresses, key=stresses.get)
    print(f"\n   Best method by stress: {best_method}")
    print("   " + "="*56)

    # Visualize
    print("\n6. Creating visualization...")
    visualize_projections(data, frechet_mean, exp_coords, pns_coords, pns_object,
                         exp_stress, pns_stress, reconstructed, reconstruction_error)

    print("\n" + "="*60)
    print("Example completed successfully!")
    print("="*60)

    return {
        'data': data,
        'frechet_mean': frechet_mean,
        'exp_coords': exp_coords,
        'pns_coords': pns_coords,
        'pns_object': pns_object,
        'exp_stress': exp_stress,
        'pns_stress': pns_stress,
        'reconstructed': reconstructed,
        'reconstruction_error': reconstruction_error
    }


def visualize_projections(data, frechet_mean, exp_coords, pns_coords, pns_object,
                         exp_stress, pns_stress, reconstructed=None, reconstruction_error=None):
    """Visualize the original data, projections, and reconstructed points."""

    fig = plt.figure(figsize=(16, 8))

    # Draw unit sphere helper arrays
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))

    # 1. Original data on sphere
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')

    ax1.plot_surface(x_sphere, y_sphere, z_sphere, color='lightgray',
                     alpha=0.2, linewidth=0, antialiased=True)
    ax1.plot_wireframe(x_sphere, y_sphere, z_sphere, color='gray',
                       alpha=0.1, linewidth=0.5, rstride=5, cstride=5)

    ax1.scatter(data[:, 0], data[:, 1], data[:, 2], c='blue', alpha=0.6, s=20)
    ax1.scatter([frechet_mean[0]], [frechet_mean[1]], [frechet_mean[2]],
                c='red', s=100, marker='*', label='Frechet Mean')

    # Plot the principal great circle from PNS
    if pns_object.spheres_ and pns_object.spheres_[0] is not None:
        normal = pns_object.spheres_[0].normals[0]

        if abs(normal[0]) < 0.9:
            v1 = np.array([1, 0, 0])
        else:
            v1 = np.array([0, 1, 0])

        v1 = v1 - np.dot(v1, normal) * normal
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(normal, v1)
        v2 = v2 / np.linalg.norm(v2)

        t = np.linspace(0, 2*np.pi, 100)
        circle = np.outer(np.cos(t), v1) + np.outer(np.sin(t), v2)

        ax1.plot(circle[:, 0], circle[:, 1], circle[:, 2],
                'g-', linewidth=2, label='Principal Great Circle', alpha=0.7)
        ax1.quiver(0, 0, 0, normal[0], normal[1], normal[2],
                  color='purple', arrow_length_ratio=0.1, linewidth=2,
                  label='PNS Pole (Normal)')

    ax1.set_title('Original Data on Sphere')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.view_init(elev=20, azim=-60)
    ax1.set_xlim([-1.2, 1.2])
    ax1.set_ylim([-1.2, 1.2])
    ax1.set_zlim([-1.2, 1.2])
    ax1.set_box_aspect([1, 1, 1])

    # 2. Exponential Map Chart (theta, phi)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(exp_coords[:, 0], exp_coords[:, 1], c='blue', alpha=0.6, s=20)
    ax2.scatter([0], [0], c='red', s=100, marker='*', label='Origin')
    ax2.set_title(f'Log Map Projection (Sphere -> Tangent)\nStress: {exp_stress["stress"]:.4f}')
    ax2.set_xlabel('θ (radians)')
    ax2.set_ylabel('φ (radians)')
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax2.axvline(x=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')

    # 3. PNS Residuals Chart (theta, phi)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(pns_coords[:, 0], pns_coords[:, 1], c='orange', alpha=0.6, s=20)
    ax3.scatter([0], [0], c='red', s=100, marker='*', label='Origin')
    ax3.set_title(f'PNS Residuals\nStress: {pns_stress["stress"]:.4f}')
    ax3.set_xlabel('θ (radians)')
    ax3.set_ylabel('φ (radians)')
    ax3.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax3.axvline(x=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')

    # 4. Reconstructed data on sphere (after round-trip: log map -> exp map)
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')

    ax4.plot_surface(x_sphere, y_sphere, z_sphere, color='lightgray',
                     alpha=0.2, linewidth=0, antialiased=True)
    ax4.plot_wireframe(x_sphere, y_sphere, z_sphere, color='gray',
                       alpha=0.1, linewidth=0.5, rstride=5, cstride=5)

    if reconstructed is not None:
        # Normalize reconstructed points
        reconstructed_norm = reconstructed / np.linalg.norm(reconstructed, axis=1, keepdims=True)

        # Plot original in light blue, reconstructed in green
        ax4.scatter(data[:, 0], data[:, 1], data[:, 2], c='blue', alpha=0.3, s=15, label='Original')
        ax4.scatter(reconstructed_norm[:, 0], reconstructed_norm[:, 1], reconstructed_norm[:, 2],
                   c='green', alpha=0.6, s=15, label='Reconstructed')
        ax4.scatter([frechet_mean[0]], [frechet_mean[1]], [frechet_mean[2]],
                   c='red', s=100, marker='*', label='Base Point')

        if reconstruction_error is not None:
            title = f'Exp Map Reconstruction (Tangent -> Sphere)\nMean Error: {reconstruction_error["mean_error_deg"]:.2f}°'
        else:
            title = 'Exp Map Reconstruction (Tangent -> Sphere)'
    else:
        ax4.scatter(data[:, 0], data[:, 1], data[:, 2], c='blue', alpha=0.6, s=20)
        title = 'Reconstruction (No Data)'

    ax4.set_title(title)
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.view_init(elev=20, azim=-60)
    ax4.set_xlim([-1.2, 1.2])
    ax4.set_ylim([-1.2, 1.2])
    ax4.set_zlim([-1.2, 1.2])
    ax4.set_box_aspect([1, 1, 1])

    plt.tight_layout()
    plt.savefig('/Users/kaisardauletbek/Documents/GitHub/RNA-Classification/mintage/spherical_projections.png',
                dpi=150, bbox_inches='tight')
    print("   Figure saved to: mintage/spherical_projections.png")
    plt.show()


if __name__ == "__main__":
    results = example_usage()
