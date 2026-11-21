"""Mouse-to-surface queries for the editor.

All hits are returned in the object's LOCAL space, which is how the scene
data stores points.  Screen projection of many points at once is done with
numpy through the view's perspective matrix.
"""

import numpy as np
from bpy_extras import view3d_utils
from mathutils import Vector

from ..model import pairs as P


class Hit:
    __slots__ = ("local", "normal", "face", "world")

    def __init__(self, local, normal, face, world):
        self.local = local
        self.normal = normal
        self.face = face
        self.world = world


def mouse_ray(region, rv3d, mouse):
    origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, mouse)
    direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, mouse)
    return origin, direction.normalized()


def surface_hit(obj, depsgraph, region, rv3d, mouse, snap_vertex=False):
    """First surface point under ``mouse`` on ``obj`` (or None)."""
    if obj is None or region is None or rv3d is None:
        return None
    origin, direction = mouse_ray(region, rv3d, mouse)
    inv = obj.matrix_world.inverted_safe()
    o = inv @ origin
    d = (inv.to_3x3() @ direction).normalized()
    ok = False
    # along the view ray, then backwards (the ray origin can sit inside or
    # in front of the mesh, e.g. in orthographic views)
    for ray_dir in (d, -d):
        try:
            ok, loc, nor, face = obj.ray_cast(o, ray_dir, depsgraph=depsgraph)
        except RuntimeError:
            return None
        if ok:
            break
    if not ok:
        return None
    if snap_vertex:
        loc = _nearest_face_vertex(obj, depsgraph, loc, face)
    return Hit(loc, nor, face, obj.matrix_world @ loc)


def _nearest_face_vertex(obj, depsgraph, loc, face):
    mesh = obj.evaluated_get(depsgraph).data
    try:
        poly = mesh.polygons[face]
    except (IndexError, AttributeError):
        return loc
    best = min(poly.vertices, key=lambda v: (mesh.vertices[v].co - loc).length)
    return mesh.vertices[best].co.copy()


def is_visible(obj, depsgraph, region, rv3d, world, tol):
    """True when nothing on ``obj`` sits between the eye and ``world``."""
    if rv3d.is_perspective:
        eye = rv3d.view_matrix.inverted().translation
        direction = world - eye
    else:
        direction = rv3d.view_rotation @ Vector((0.0, 0.0, -1.0))
        eye = world - direction * 1e4
        direction = direction * 1e4
    dist = direction.length
    if dist < 1e-9:
        return True
    inv = obj.matrix_world.inverted_safe()
    o = inv @ eye
    target = inv @ world
    d = target - o
    local_dist = d.length
    try:
        ok, loc, _n, _f = obj.ray_cast(o, d.normalized(),
                                       distance=local_dist,
                                       depsgraph=depsgraph)
    except RuntimeError:
        return True
    return not ok or (loc - target).length <= tol


def project(region, rv3d, world_pts):
    """(n, 2) region coordinates and an in-front mask for world points."""
    W = np.asarray(world_pts, dtype=float).reshape(-1, 3)
    if not len(W):
        return np.zeros((0, 2)), np.zeros(0, bool)
    M = np.array(rv3d.perspective_matrix)
    H = W @ M[:3, :3].T + M[:3, 3]
    w = W @ M[3, :3] + M[3, 3]
    front = w > 1e-6
    w = np.where(front, w, 1.0)
    xy = np.empty((len(W), 2))
    xy[:, 0] = (H[:, 0] / w + 1.0) * 0.5 * region.width
    xy[:, 1] = (H[:, 1] / w + 1.0) * 0.5 * region.height
    return xy, front


def to_world(obj, local_pts):
    M = np.array(obj.matrix_world)
    L = np.asarray(local_pts, dtype=float).reshape(-1, 3)
    return L @ M[:3, :3].T + M[:3, 3]


def segment_distance(xy, q, closed=False):
    """(segment index, t, pixel distance) from q to the 2D polyline xy."""
    if len(xy) < 2:
        if len(xy) == 1:
            return 0, 0.0, float(np.linalg.norm(xy[0] - q))
        return -1, 0.0, float("inf")
    path = np.vstack([xy, xy[:1]]) if closed else xy
    a, b = path[:-1], path[1:]
    ab = b - a
    L = (ab * ab).sum(axis=1)
    L[L < 1e-12] = 1e-12
    t = np.clip(((q - a) * ab).sum(axis=1) / L, 0.0, 1.0)
    d = np.linalg.norm(a + t[:, None] * ab - q, axis=1)
    k = int(np.argmin(d))
    return k, float(t[k]), float(d[k])


# ------------------------------------------------------------ border loops --

def border_loops(mesh):
    """Open boundaries of ``mesh`` as [(closed, [vertex indices])]."""
    if not len(mesh.edges) or not len(mesh.loops):
        return []
    edge_of_loop = np.empty(len(mesh.loops), dtype=np.int64)
    mesh.loops.foreach_get("edge_index", edge_of_loop)
    faces_per_edge = np.bincount(edge_of_loop, minlength=len(mesh.edges))
    ev = np.empty(len(mesh.edges) * 2, dtype=np.int64)
    mesh.edges.foreach_get("vertices", ev)
    border = ev.reshape(-1, 2)[faces_per_edge == 1]

    links = {}
    for a, b in border.tolist():
        links.setdefault(a, []).append(b)
        links.setdefault(b, []).append(a)
    seen_edges = set()
    loops = []
    # walk from chain ends first so open chains come out whole
    starts = [v for v, nb in links.items() if len(nb) != 2]
    starts += [v for v, nb in links.items() if len(nb) == 2]
    for start in starts:
        for nxt in links[start]:
            if (min(start, nxt), max(start, nxt)) in seen_edges:
                continue
            chain = [start]
            prev, cur = start, nxt
            closed = False
            while True:
                seen_edges.add((min(prev, cur), max(prev, cur)))
                if cur == start:
                    closed = True
                    break
                chain.append(cur)
                nbrs = links[cur]
                if len(nbrs) != 2:
                    break
                step = nbrs[0] if nbrs[1] == prev else nbrs[1]
                if (min(cur, step), max(cur, step)) in seen_edges:
                    break
                prev, cur = cur, step
            if len(chain) >= 2:
                loops.append((closed, chain))
    return loops


class BorderCache:
    """Per-side border loops in local space, extracted on first use."""

    def __init__(self):
        self._loops = {}

    def get(self, mw, side, depsgraph):
        obj = P.obj_for(mw, side)
        key = (side, obj.name if obj else None)
        hit = self._loops.get(key)
        if hit is not None:
            return hit
        loops = []
        if obj is not None:
            if side == 'CAGE':
                mesh = obj.data
                co = _rest_positions(obj)
            else:
                ev = obj.evaluated_get(depsgraph)
                mesh = ev.data
                co = np.empty(len(mesh.vertices) * 3)
                mesh.vertices.foreach_get("co", co)
                co = co.reshape(-1, 3)
            for closed, idx in border_loops(mesh):
                loops.append((closed, co[idx]))
        self._loops[key] = loops
        return loops


def _rest_positions(cage):
    from ..solver.pipeline import rest_coords
    return rest_coords(cage)
