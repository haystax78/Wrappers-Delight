"""Cached, surface-hugging geometry for drawing correspondences.

Each half of a pair is turned into three local-space point sets:

``line``     the control points joined by straight segments,
``controls`` the control points themselves,
``samples``  the matched points the fit will use.

Everything is nudged off the surface along its normal so depth-tested
drawing doesn't z-fight with the mesh.  Results are cached per (pair, side)
and keyed on the point data, so moving an object costs nothing (drawing
applies ``matrix_world`` on the GPU) and editing a pair only rebuilds that
pair.
"""

import numpy as np
from mathutils import Vector

from . import pairs as P
from . import sampling

_CACHE = {}


class HalfGeometry:
    __slots__ = ("line", "controls", "samples", "closed")

    def __init__(self, line, controls, samples, closed):
        self.line = line
        self.controls = controls
        self.samples = samples
        self.closed = closed


def invalidate_all():
    _CACHE.clear()


def invalidate(uid):
    for key in [k for k in _CACHE if k[0] == uid]:
        del _CACHE[key]


def _object_size(obj):
    dims = [abs(d) for d in obj.dimensions]
    scale = max(abs(s) for s in obj.matrix_world.to_scale()) or 1.0
    return max(max(dims) / scale, 1e-6)


def project_local(obj, depsgraph, pts, lift=0.0):
    """Snap local points onto the evaluated surface; returns (pts, normals)."""
    out = np.empty((len(pts), 3))
    nrm = np.zeros((len(pts), 3))
    for i, co in enumerate(pts):
        try:
            ok, loc, no, _ = obj.closest_point_on_mesh(
                Vector(co), depsgraph=depsgraph)
        except (RuntimeError, TypeError):
            ok = False
        if ok:
            out[i] = loc
            nrm[i] = no
        else:
            out[i] = co
    if lift:
        out += nrm * lift
    return out, nrm


def half(obj, depsgraph, pair, side):
    """HalfGeometry for one side of ``pair`` (or None when it has no points)."""
    cos = P.coords(pair, side)
    if not cos or obj is None:
        return None
    count = P.sample_count(pair)
    key = (tuple(cos), pair.closed, count, obj.name)
    slot = (pair.uid, side)
    hit = _CACHE.get(slot)
    if hit is not None and hit[0] == key:
        return hit[1]

    lift = 0.0015 * _object_size(obj)
    closed = pair.closed and len(cos) > 2
    ctrl, _ = project_local(obj, depsgraph, cos, lift)
    if len(cos) > 1:
        line = np.vstack([ctrl, ctrl[:1]]) if closed else ctrl
        if len(cos) == count:
            smp = ctrl
        else:
            smp, _ = project_local(
                obj, depsgraph, sampling.resample(cos, count, closed), lift)
    else:
        line = ctrl
        smp = ctrl
    geo = HalfGeometry(line, ctrl, smp, closed)
    _CACHE[slot] = (key, geo)
    return geo
