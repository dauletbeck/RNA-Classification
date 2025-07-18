# geometry/hypersphere.py

import numpy as np
import numpy.linalg as la

EPS = 1e-8

# def gram_schmidt(vectors):
#     """Orthonormalize the rows of a matrix using Gram-Schmidt."""
#     vectors = np.atleast_2d(vectors)
#     n = vectors.shape[0]
#     out = []
#     for i in range(n):
#         vec = vectors[i].copy()
#         for j in range(i):
#             vec -= np.dot(out[j], vec) * out[j]
#         norm = la.norm(vec)
#         if norm > EPS:
#             out.append(vec / norm)
#     return np.vstack(out)

def gram_schmidt(vectors):
    """Orthonormalize the rows of a matrix using QR decomposition."""
    vectors = np.atleast_2d(vectors)
    # Get the rank of the matrix to handle linearly dependent vectors
    rank = la.matrix_rank(vectors, tol=EPS)
    if rank == 0:
        return np.empty((0, vectors.shape[1]), dtype=vectors.dtype)
    
    # Use QR decomposition on the transpose to orthonormalize rows
    q, r = la.qr(vectors.T)
    
    # Return the first 'rank' vectors, transposed back to be row vectors
    return q[:, :rank].T

class Sphere:
    """
    General subsphere of codimension c in S^{D-1}.
    - normals: (c, D) orthonormal matrix, each row is a normal vector.
    - position: (c,) vector of "heights" along each normal, |position| <= 1.
    """
    def __init__(self, normals, position):
        normals = np.atleast_2d(normals)
        position = np.atleast_1d(position)
        if normals.shape[0] != len(position):
            raise ValueError("Number of normals and positions must match")
        # Orthonormalize for safety:
        normals = gram_schmidt(normals)
        self.normals = normals
        self.position = position
        self.codim = normals.shape[0]
        self.D = normals.shape[1]
        # Center of the subsphere in ambient space
        self.center = np.dot(self.position, self.normals)
        self.flat_radius = np.sqrt(max(0.0, 1 - la.norm(self.position) ** 2))

    def foot_points(self, points):
        """
        Orthogonally project each point onto the subsphere in ambient space.
        points: (N, D) array
        Returns: (N, D) array
        """
        points = np.atleast_2d(points)
        proj = points.copy()
        for k in range(self.codim):
            nrm = self.normals[k]
            coeffs = np.dot(proj, nrm)
            proj -= np.outer(coeffs, nrm)
        norm_proj = la.norm(proj, axis=1)
        # Avoid division by zero
        zeros = norm_proj < EPS
        proj[zeros] = self.center
        norm_proj[zeros] = 1.0
        return self.center + self.flat_radius * (proj / norm_proj[:, None])
    
    def signed_distances(self, points, with_feet=False):
        """
        Signed geodesic distances from each point to the subsphere (only for codim=1).
        For codim > 1, raises NotImplementedError.
        Returns: (N,) array [and feet if with_feet]
        """
        # get distances from distances func

        # make sure codim 1 (more than 1 normal)
        # if len(self.normal) > 1
        # print(codim more than 1)
        # return distances

        # signs = np.sign(np.sum(points * normal, axis=1))
        # return distances * signs

        if len(self.normals) > 1:
            # raise NotImplementedError("Signed distances only implemented for codimension 1 (hyperspheres)")
            print("WARNING CODIM >1 RETURNING UNSIGNED DISTANCES")
            return self.distances(points, with_feet)
        
        # Get unsigned geodesic distances
        if with_feet:
            dists, feet = self.distances(points, with_feet=True)
        else:
            dists = self.distances(points, with_feet=False)
        normal = self.normals[0]
        # Compute the sign: dot product with normal minus offset (position)
        signs = np.sign(np.sum(points * normal, axis=1))
        signed_dists = dists * signs
        if with_feet:
            return signed_dists, feet
        return signed_dists

    def distances(self, points, with_feet=False):
        """
        Geodesic distances from each point to the subsphere.
        Returns: (N,) array [and feet if with_feet]
        """
        feet = self.foot_points(points)
        dists = np.arccos(np.clip(np.sum(points * feet, axis=1), -1, 1))
        if with_feet:
            return dists, feet
        return dists

    def calculate_projection_matrix(self):
        """
        Returns (D-c, D) projection matrix from ambient space onto the tangent space
        of the subsphere.
        """
        mat = np.eye(self.D)
        for n in self.normals:
            mat = mat - np.outer(n, n)
        # Orthonormal basis for the nullspace (tangent space)
        U, S, Vt = la.svd(mat)
        proj_basis = Vt[S > EPS]
        return proj_basis

    def project(self, points):
        """
        Project points (on sphere) to the lower-dimensional sphere coordinates.
        Result is in the tangent space (i.e., extrinsic, not intrinsic).
        """
        points = np.atleast_2d(points)
        feet = self.foot_points(points)
        tangent_basis = self.calculate_projection_matrix()
        proj = feet @ tangent_basis.T
        return proj / self.flat_radius

    def unproject(self, points):
        """
        Map low-dimensional tangent space coordinates back to ambient space.
        """
        tangent_basis = self.calculate_projection_matrix()
        proj = points @ tangent_basis
        return proj * self.flat_radius + self.center

    def __str__(self):
        return f"Sphere(codim={self.codim}, D={self.D}, center={self.center}, flat_radius={self.flat_radius})"
