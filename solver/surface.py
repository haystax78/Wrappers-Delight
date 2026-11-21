"""World-space closest-point and ray queries against a target object.

Dense targets (scans, sculpts) can have millions of triangles, so the probe
avoids copying geometry whenever it can:

* Objects whose transform is a similarity (rotation + uniform scale, the
  usual case) are queried through ``Object.closest_point_on_mesh`` and
  ``Object.ray_cast``, which use the BVH Blender already keeps for the
  evaluated mesh.  Distances scale uniformly, so the local-space answer is
  the world-space answer.
* Any other transform (non-uniform scale, shear) gets a world-space
  ``BVHTree`` built from numpy arrays of the evaluated triangles.
"""

import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

UNLIMITED = 1.0e18


def _is_similarity(M3):
    s = np.linalg.svd(M3, compute_uv=False)
    return s.max() - s.min() <= 1e-5 * max(s.max(), 1e-12)


def triangles_local(mesh):
    """(verts (n,3), tris (t,3)) of a mesh in its own space."""
    co = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", co)
    mesh.calc_loop_triangles()
    tv = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int64)
    mesh.loop_triangles.foreach_get("vertices", tv)
    return co.reshape(-1, 3), tv.reshape(-1, 3)


def tree_from_arrays(verts, tris):
    return BVHTree.FromPolygons(np.asarray(verts).tolist(),
                                np.asarray(tris).tolist(),
                                all_triangles=True)


class SurfaceProbe:
    """Batched nearest-point / ray queries in world space."""

    def __init__(self, obj, depsgraph):
        self.obj = obj
        self.depsgraph = depsgraph
        M = np.array(obj.matrix_world)
        self.M = M
        self.Minv = np.linalg.inv(M)
        lin = M[:3, :3]
        self.tree = None
        self.scale = 1.0
        if _is_similarity(lin):
            self.scale = float(np.linalg.svd(lin, compute_uv=False).mean())
            self.R = lin / self.scale
        else:
            ev = obj.evaluated_get(depsgraph)
            mesh = ev.to_mesh()
            try:
                v, t = triangles_local(mesh)
            finally:
                ev.to_mesh_clear()
            vw = v @ lin.T + M[:3, 3]
            self.tree = tree_from_arrays(vw, t)

    # ----------------------------------------------------------- helpers --

    def _to_local(self, p):
        return Vector(self.Minv[:3, :3] @ p + self.Minv[:3, 3])

    def _to_world(self, loc, nor):
        p = self.M[:3, :3] @ np.asarray(loc) + self.M[:3, 3]
        n = self.R @ np.asarray(nor)
        ln = np.linalg.norm(n)
        return p, (n / ln if ln > 1e-20 else n)

    # ----------------------------------------------------------- queries --

    def nearest_one(self, p, reach=UNLIMITED):
        """(location, normal) in world space, or (None, None)."""
        p = np.asarray(p, dtype=float)
        if self.tree is not None:
            loc, nor, _i, _d = self.tree.find_nearest(Vector(p), reach)
            if loc is None:
                return None, None
            return np.array(loc), np.array(nor)
        hit, loc, nor, _i = self.obj.closest_point_on_mesh(
            self._to_local(p), distance=min(reach / self.scale, UNLIMITED),
            depsgraph=self.depsgraph)
        if not hit:
            return None, None
        return self._to_world(loc, nor)

    def ray_one(self, p, d, reach=UNLIMITED):
        """First hit along direction ``d`` within ``reach``: (loc, nor, dist)."""
        p = np.asarray(p, dtype=float)
        d = np.asarray(d, dtype=float)
        if self.tree is not None:
            loc, nor, _i, dist = self.tree.ray_cast(Vector(p), Vector(d),
                                                     reach)
            if loc is None:
                return None, None, None
            return np.array(loc), np.array(nor), dist
        o = self._to_local(p)
        dl = Vector(self.Minv[:3, :3] @ d)
        if dl.length < 1e-20:
            return None, None, None
        dl.normalize()
        hit, loc, nor, _i = self.obj.ray_cast(
            o, dl, distance=min(reach / self.scale, UNLIMITED),
            depsgraph=self.depsgraph)
        if not hit:
            return None, None, None
        wp, wn = self._to_world(loc, nor)
        return wp, wn, float(np.linalg.norm(wp - p))

    def nearest(self, P, reach, active=None):
        """Closest surface point per row of P.

        Returns (T, N, ok): targets, unit surface normals and a hit mask.
        Rows outside ``active`` are skipped (ok False, T = P).
        """
        P = np.asarray(P, dtype=float)
        T = P.copy()
        N = np.zeros_like(P)
        ok = np.zeros(len(P), dtype=bool)
        rows = range(len(P)) if active is None else np.flatnonzero(active)
        for i in rows:
            loc, nor = self.nearest_one(P[i], reach)
            if loc is not None:
                T[i] = loc
                N[i] = nor
                ok[i] = True
        return T, N, ok

    def along(self, P, D, reach, active):
        """Nearest hit along +D or -D for each active row."""
        T = P.copy()
        N = np.zeros_like(P)
        ok = np.zeros(len(P), dtype=bool)
        for i in np.flatnonzero(active):
            best = None
            for sgn in (1.0, -1.0):
                loc, nor, dist = self.ray_one(P[i], sgn * D[i], reach)
                if loc is not None and (best is None or dist < best[2]):
                    best = (loc, nor, dist)
            if best is not None:
                T[i], N[i] = best[0], best[1]
                ok[i] = True
        return T, N, ok
