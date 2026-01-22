"""Debug script to examine PNS residual angles."""
import numpy as np
from distortion_quantification import SphericalDataGenerator
from pnds.PNDS_PNS import PNS

# Generate same data as in main script
np.random.seed(42)
generator = SphericalDataGenerator(dim=3, random_seed=42)
mean_direction = np.array([0.5, -0.5, np.sqrt(0.5)])
mean_direction = mean_direction / np.linalg.norm(mean_direction)

data = generator.generate_anisotropic_mixture(
    n_samples=200,
    mean_direction=mean_direction,
    primary_kappa=10.0,
    secondary_kappa=2.0,
    mixture_weight=0.85
)

print("Data shape:", data.shape)
print("Data min/max:")
print("  X:", data[:, 0].min(), "to", data[:, 0].max())
print("  Y:", data[:, 1].min(), "to", data[:, 1].max())
print("  Z:", data[:, 2].min(), "to", data[:, 2].max())

# Fit PNS
pns = PNS(mode='great', verbose=False)
pns.fit(data)

print("\n" + "="*60)
print("PNS Residual Angles Analysis")
print("="*60)

print("\nPNS dists_ structure:")
print("  Type:", type(pns.dists_))
print("  Length:", len(pns.dists_) if isinstance(pns.dists_, list) else "N/A")

for i, dist in enumerate(pns.dists_):
    if isinstance(dist, np.ndarray):
        print(f"  dists_[{i}]: shape={dist.shape}, min={dist.min():.2f}°, max={dist.max():.2f}°, mean={dist.mean():.2f}°, std={dist.std():.2f}°")

# Extract residual angles
theta_deg = pns.dists_[-2]
phi_deg = pns.dists_[-1]

print("\nResidual Angles (degrees):")
print(f"  theta: min={theta_deg.min():.2f}°, max={theta_deg.max():.2f}°, mean={theta_deg.mean():.2f}°, std={theta_deg.std():.2f}°")
print(f"  phi:   min={phi_deg.min():.2f}°, max={phi_deg.max():.2f}°, mean={phi_deg.mean():.2f}°, std={phi_deg.std():.2f}°")

# Convert to radians (with 90° offset for theta)
theta_rad = np.radians(theta_deg + 90.0)  # Offset by 90° to center at equator
phi_rad = np.radians(phi_deg)

print("\nAfter 90° theta offset (radians):")
print(f"  theta: min={theta_rad.min():.4f}, max={theta_rad.max():.4f}, mean={theta_rad.mean():.4f}, std={theta_rad.std():.4f}")
print(f"  phi:   min={phi_rad.min():.4f}, max={phi_rad.max():.4f}, mean={phi_rad.mean():.4f}, std={phi_rad.std():.4f}")
print(f"  Note: theta now centered at π/2 = {np.pi/2:.4f} (equator) instead of 0 (pole)")

# Create reparametrized sphere
reparam_sphere = np.zeros((len(theta_rad), 3))
reparam_sphere[:, 0] = np.sin(theta_rad) * np.cos(phi_rad)
reparam_sphere[:, 1] = np.sin(theta_rad) * np.sin(phi_rad)
reparam_sphere[:, 2] = np.cos(theta_rad)

print("\nReparametrized sphere statistics:")
print("  X: min={:.4f}, max={:.4f}, mean={:.4f}, std={:.4f}".format(
    reparam_sphere[:, 0].min(), reparam_sphere[:, 0].max(),
    reparam_sphere[:, 0].mean(), reparam_sphere[:, 0].std()))
print("  Y: min={:.4f}, max={:.4f}, mean={:.4f}, std={:.4f}".format(
    reparam_sphere[:, 1].min(), reparam_sphere[:, 1].max(),
    reparam_sphere[:, 1].mean(), reparam_sphere[:, 1].std()))
print("  Z: min={:.4f}, max={:.4f}, mean={:.4f}, std={:.4f}".format(
    reparam_sphere[:, 2].min(), reparam_sphere[:, 2].max(),
    reparam_sphere[:, 2].mean(), reparam_sphere[:, 2].std()))

# Check if points are on unit sphere
norms = np.linalg.norm(reparam_sphere, axis=1)
print("\nNorms of reparametrized points:")
print(f"  min={norms.min():.6f}, max={norms.max():.6f}, mean={norms.mean():.6f}")

# Compute geodesic distances
from scipy.spatial.distance import pdist, squareform

def geodesic_distance_matrix(points):
    """Compute pairwise geodesic distances on unit sphere."""
    n = len(points)
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            cos_dist = np.clip(np.dot(points[i], points[j]), -1.0, 1.0)
            dists[i, j] = np.arccos(cos_dist)
            dists[j, i] = dists[i, j]
    return dists

original_dists = geodesic_distance_matrix(data)
reparam_dists = geodesic_distance_matrix(reparam_sphere)

print("\nGeodesic distances on original sphere:")
print(f"  min={original_dists[original_dists > 0].min():.4f}, max={original_dists.max():.4f}")
print(f"  mean={original_dists[original_dists > 0].mean():.4f}, std={original_dists[original_dists > 0].std():.4f}")

print("\nGeodesic distances on reparametrized sphere:")
print(f"  min={reparam_dists[reparam_dists > 0].min():.4f}, max={reparam_dists.max():.4f}")
print(f"  mean={reparam_dists[reparam_dists > 0].mean():.4f}, std={reparam_dists[reparam_dists > 0].std():.4f}")

print("\n" + "="*60)
print("Key insight: If theta_deg is small (e.g., 5°), then")
print("theta_rad ≈ 0.087, and cos(theta_rad) ≈ 0.996,")
print("meaning the point is very close to the north pole (0,0,1).")
print("This causes extreme clustering and distortion!")
print("="*60)
