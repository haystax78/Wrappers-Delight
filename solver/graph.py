"""Vertex adjacency of the cage, with vectorised neighbourhood operations."""

import numpy as np


class MeshGraph:
    """Edges, triangles and vertex degrees of a mesh, built once per fit."""

    def __init__(self, n, edges, tris):
        self.n = int(n)
        self.a = np.ascontiguousarray(edges[:, 0])
        self.b = np.ascontiguousarray(edges[:, 1])
        self.tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
        deg = (np.bincount(self.a, minlength=self.n)
               + np.bincount(self.b, minlength=self.n)).astype(float)
        self.degree = deg
        self.has_edges = deg > 0

    @classmethod
    def from_mesh(cls, mesh):
        ev = np.empty(len(mesh.edges) * 2, dtype=np.int64)
        mesh.edges.foreach_get("vertices", ev)
        mesh.calc_loop_triangles()
        tv = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int64)
        mesh.loop_triangles.foreach_get("vertices", tv)
        return cls(len(mesh.vertices), ev.reshape(-1, 2), tv)

    # -- neighbourhood sums ---------------------------------------------

    def gather(self, values):
        """Sum of each vertex's neighbours' values; ``values`` is (n,) or
        (n, k)."""
        v = np.asarray(values, dtype=float)
        if v.ndim == 1:
            return (np.bincount(self.a, weights=v[self.b], minlength=self.n)
                    + np.bincount(self.b, weights=v[self.a], minlength=self.n))
        out = np.empty_like(v)
        for c in range(v.shape[1]):
            col = v[:, c]
            out[:, c] = (
                np.bincount(self.a, weights=col[self.b], minlength=self.n)
                + np.bincount(self.b, weights=col[self.a], minlength=self.n))
        return out

    def umbrella(self, P):
        """Vector from each vertex to the centroid of its neighbours."""
        deg = np.maximum(self.degree, 1.0)[:, None]
        U = self.gather(P) / deg - P
        U[~self.has_edges] = 0.0
        return U

    def normals(self, P):
        """Area-weighted unit vertex normals for positions ``P``."""
        t = self.tris
        N = np.zeros((self.n, 3))
        if len(t):
            fn = np.cross(P[t[:, 1]] - P[t[:, 0]], P[t[:, 2]] - P[t[:, 0]])
            for corner in range(3):
                idx = t[:, corner]
                for c in range(3):
                    N[:, c] += np.bincount(idx, weights=fn[:, c],
                                           minlength=self.n)
        length = np.linalg.norm(N, axis=1, keepdims=True)
        length[length < 1e-20] = 1.0
        return N / length

    def blend(self, field, weight, passes):
        """Confidence-weighted diffusion of a per-vertex vector field.

        Values and their weights are blurred together over ``passes`` edge
        rings and divided afterwards, so vertices without data (weight 0)
        take on what their neighbours agree on, and confident vertices are
        smoothed only as much as ``passes`` asks.  Returns (field, weight).
        """
        w = np.asarray(weight, dtype=float)
        num = field * w[:, None]
        den = w.copy()
        inv = 1.0 / (1.0 + self.degree)
        for _ in range(int(passes)):
            num = (num + self.gather(num)) * inv[:, None]
            den = (den + self.gather(den)) * inv
        out = np.zeros_like(field)
        ok = den > 1e-9
        out[ok] = num[ok] / den[ok, None]
        return out, den
