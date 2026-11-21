"""Assemble a wrap from scene data and run it stage by stage.

    matched samples -> similarity alignment -> landmark warp -> conform

:class:`WrapJob` gathers everything up front (so errors surface before any
work starts) and exposes :meth:`WrapJob.run`, a generator that yields after
each stage so callers can show progress or stop early.
"""

import numpy as np
from mathutils import Vector

from ..model import pairs as P
from ..model import sampling
from . import conform as conform_mod
from .graph import MeshGraph
from .surface import SurfaceProbe, tree_from_arrays
from .transforms import LandmarkWarp, fit_similarity

MIN_SAMPLES = 3
LOCK_DENSITY = 8     # locked points per curve segment (between samples)



def exclude_ramp(graph, X, frozen, landmarks, smooth_passes=8):
    """How much of the surface fit each vertex receives (0..1), or None.

    0 inside the excluded area (kept as warped), 1 from the nearest
    landmarks outward, easing in between with a smoothstep of the distance
    along the mesh.  The width of the transition is, for each stretch of the
    excluded area's edge, the mesh distance from that edge to its nearest
    landmark; it is smoothed along the mesh so the ramp has no seams.
    """
    frozen = np.asarray(frozen, bool)
    if not frozen.any() or frozen.all() or not len(landmarks):
        return None
    # landmark vertices: the free vertex nearest each matched sample
    free_idx = np.flatnonzero(~frozen)
    lm = set()
    for q in np.asarray(landmarks, float):
        d = np.einsum('ij,ij->i', X[free_idx] - q, X[free_idx] - q)
        lm.add(int(free_idx[int(d.argmin())]))
    d_ex, label = graph.distances(X, np.flatnonzero(frozen))
    d_lm, _ = graph.distances(X, sorted(lm))
    # transition width per excluded-edge vertex, carried to every vertex by
    # its nearest one, then smoothed
    ok = (label >= 0) & np.isfinite(d_ex)
    width = np.zeros(len(X))
    width[ok] = d_lm[label[ok]]
    width[~np.isfinite(width)] = 0.0
    sm, _den = graph.blend(width[:, None], ok.astype(float), smooth_passes)
    width = np.maximum(sm[:, 0], 1e-12)
    t = np.where(ok, np.clip(d_ex / width, 0.0, 1.0), 1.0)
    ramp = t * t * (3.0 - 2.0 * t)
    ramp[frozen] = 0.0
    return ramp


def _barycentric(tri, p):
    """Barycentric weights of ``p`` (on or near) triangle ``tri`` (3, 3)."""
    a, b, c = tri
    v0, v1, v2 = b - a, c - a, p - a
    d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
    d20, d21 = v2 @ v0, v2 @ v1
    den = d00 * d11 - d01 * d01
    if abs(den) < 1e-30:
        return np.array([1.0, 0.0, 0.0])
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    bc = np.clip(np.array([1.0 - v - w, v, w]), 0.0, None)
    return bc / max(bc.sum(), 1e-12)


def _samples(cos, count, closed):
    if len(cos) == count:
        return np.array(cos, dtype=float)
    return sampling.resample(cos, count, closed)

class WrapError(Exception):
    pass


def _affine(M, X):
    return X @ M[:3, :3].T + M[:3, 3]


def rest_coords(cage):
    """Cage vertex positions of the rest shape (Basis key, else the mesh)."""
    me = cage.data
    n = len(me.vertices)
    co = np.empty(n * 3)
    keys = me.shape_keys
    if keys is not None and len(keys.key_blocks):
        keys.key_blocks[0].data.foreach_get("co", co)
    else:
        me.vertices.foreach_get("co", co)
    return co.reshape(n, 3)


def group_mask(obj, name):
    """Bool (n,) of vertices with non-zero weight in vertex group ``name``."""
    n = len(obj.data.vertices)
    mask = np.zeros(n, bool)
    vg = obj.vertex_groups.get(name) if name else None
    if vg is None:
        return mask
    gi = vg.index
    for v in obj.data.vertices:
        for g in v.groups:
            if g.group == gi and g.weight > 0.0:
                mask[v.index] = True
                break
    return mask


def conform_params(mw, reach):
    if mw.fit_preset == 'CUSTOM':
        kw = conform_mod.custom_params(mw.custom_iterations,
                                       mw.custom_stiffness, mw.custom_detail,
                                       mw.custom_relax)
    else:
        kw = dict(conform_mod.PRESETS[mw.fit_preset])
    return conform_mod.ConformParams(reach=reach, **kw)


class WrapJob:
    def __init__(self, context, mw):
        cage, target = mw.cage, mw.target
        if cage is None or target is None:
            raise WrapError("Pick a cage and a target mesh first")
        if cage == target:
            raise WrapError("Cage and target must be different objects")
        self.mw = mw
        self.cage = cage
        self.target = target
        self.depsgraph = context.evaluated_depsgraph_get()

        self.rest_local = rest_coords(cage)
        self.Mc = np.array(cage.matrix_world)
        self.rest_world = _affine(self.Mc, self.rest_local)
        self.probe = SurfaceProbe(target, self.depsgraph)

        self.src, self.dst, self.wts = self._matched_samples()
        if len(self.src) < MIN_SAMPLES:
            raise WrapError(
                f"Need at least {MIN_SAMPLES} matched points "
                f"(have {len(self.src)}): finish more pairs first")
        self.result_world = None
        self.notes = []

    # ------------------------------------------------------- samples --

    def _matched_samples(self):
        mw = self.mw
        verts, tris = rest_triangles(self.cage, self.rest_local)
        cage_tree = tree_from_arrays(verts, tris) if len(tris) else None
        Mt = np.array(self.target.matrix_world)
        src, dst, wts = [], [], []
        lock_tri, lock_bary, lock_dst, lock_w = [], [], [], []
        for pair in mw.pairs:
            if not (pair.enabled and P.is_complete(pair)):
                continue
            count = P.sample_count(pair)
            closed = pair.closed
            # drawn curves store exactly ``count`` points: use them as they are
            a = _samples(P.coords(pair, 'CAGE'), count, closed)
            b = _samples(P.coords(pair, 'TARGET'), count, closed)
            bw = _affine(Mt, b)
            for i, co in enumerate(bw):
                loc, _n = self.probe.nearest_one(co)
                if loc is not None:
                    bw[i] = loc
            if cage_tree is not None:
                for i, co in enumerate(a):
                    loc = cage_tree.find_nearest(Vector(co))[0]
                    if loc is not None:
                        a[i] = loc
                if pair.lock > 0.0:
                    # lock the whole drawn curve, not only its samples, so
                    # the cage curve can't bow away between them
                    for tri, bc, q in self._curve_lock_points(
                            a, b, closed, Mt, verts, tris, cage_tree,
                            self.lock_density):
                        lock_tri.append(tri)
                        lock_bary.append(bc)
                        lock_dst.append(q)
                        lock_w.append(pair.lock)
            src.append(_affine(self.Mc, a))
            dst.append(bw)
            wts.append(np.full(len(a), pair.weight))
        self.locks = conform_mod.CurveLocks(lock_tri, lock_bary, lock_dst,
                                            lock_w)
        if not src:
            return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
        return np.vstack(src), np.vstack(dst), np.concatenate(wts)

    lock_density = LOCK_DENSITY

    def _curve_lock_points(self, a, b, closed, Mt, verts, tris, cage_tree,
                           density):
        """(cage triangle, barycentric, target world point) along a pair.

        Both halves are traced with the same Catmull-Rom the editor draws
        them with, at the same parameters, so point k on the cage curve
        matches point k on the target curve; each is put back on its
        surface (cage rest shape / target)."""
        closed = closed and len(a) > 2
        if len(a) >= 2 and len(a) == len(b) and density > 1:
            da = sampling.catmull_rom(a, closed, density)
            db = sampling.catmull_rom(b, closed, density)
        else:
            da, db = np.asarray(a, float), np.asarray(b, float)
        dbw = _affine(Mt, db)
        for ca, cb in zip(da, dbw):
            loc, _nrm, ti, _d = cage_tree.find_nearest(Vector(ca))
            if loc is None:
                continue
            q, _n = self.probe.nearest_one(cb)
            if q is None:
                q = cb
            tri = tris[ti]
            yield tri, _barycentric(verts[tri], np.array(loc)), np.asarray(q)

    # ----------------------------------------------------------- run --

    def run(self):
        """Generator of (fraction, label, world positions or None)."""
        mw = self.mw
        yield 0.02, "Aligning", None
        sim = fit_similarity(self.src, self.dst, self.wts)
        X = sim.apply(self.rest_world)
        src_a = sim.apply(self.src)
        yield 0.06, "Aligned", X

        if len(self.src) >= 4:
            warp = LandmarkWarp(src_a, self.dst - src_a, relax=mw.warp_relax)
            X = X + warp(X)
        locks = self.locks
        if locks:
            locks.enforce(X, np.ones(len(X), bool))
        yield 0.14, "Warped", X

        if mw.fit_surface:
            graph = MeshGraph.from_mesh(self.cage.data)
            span = float(np.ptp(X, axis=0).max()) or 1.0
            params = conform_params(mw, mw.reach * span)
            # excluded verts keep their warped shape (inner lips, eye
            # sockets) instead of snapping onto interior target surfaces
            frozen = group_mask(self.cage, mw.exclude_group)
            excluded = int(frozen.sum())
            if excluded:
                self.notes.append(f"{excluded} verts kept as warped")
            pins = self._landmark_pins(X, span) if mw.lock_landmarks else None
            if pins:
                # excluded verts stay as warped even when a landmark is near
                pins = {i: q for i, q in pins.items() if not frozen[i]}
            X0 = X.copy()
            ramp = exclude_ramp(graph, X0, frozen, self.dst)
            for k, total, X in conform_mod.conform(X0, graph, self.probe,
                                                   params, pins=pins,
                                                   frozen=frozen,
                                                   locks=locks):
                if ramp is not None:
                    # ease from the warped shape at the excluded area to the
                    # full wrap at the nearest landmarks
                    X = X0 + ramp[:, None] * (X - X0)
                    if locks:
                        locks.enforce(X, ~frozen)
                yield 0.14 + 0.86 * k / total, f"Conforming {k}/{total}", X
        self.result_world = X
        yield 1.0, "Done", X

    def _landmark_pins(self, X, span):
        """Nearest cage vertex to each target sample, if close enough."""
        pins = {}
        limit = 0.03 * span
        for q in self.dst:
            d = np.linalg.norm(X - q, axis=1)
            i = int(d.argmin())
            if d[i] <= limit:
                pins[i] = q
        return pins

    def solve(self):
        for _ in self.run():
            pass
        return self.result_world

    def to_cage_local(self, X):
        return _affine(np.linalg.inv(self.Mc), X)


def rest_triangles(cage, rest_local):
    me = cage.data
    me.calc_loop_triangles()
    tv = np.empty(len(me.loop_triangles) * 3, dtype=np.int64)
    me.loop_triangles.foreach_get("vertices", tv)
    return rest_local, tv.reshape(-1, 3)
