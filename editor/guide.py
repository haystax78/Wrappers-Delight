"""Guided placement of template curves.

After a template import every pair has its cage curve but no target curve.
The editor then walks through them in list order: the next waiting pair
becomes active, the cage view frames its curve (looking straight at the
surface) and the target view frames the matching spot on the target, so the
artist only has to trace the same feature there.

The matching spot follows the displacement of the nearest finished curves;
before any are finished it is the same world spot (meshes overlapping, World
Space sync) or the same spot relative to each mesh's bounds.
"""

import numpy as np
from mathutils import Matrix, Vector

from ..model import display, sampling, snapping
from ..model import pairs as P
from . import picking
from .stage import bounds, world_sync

FRAME_MARGIN = 3.8         # view distance per curve radius
MIN_RADIUS = 0.06           # of the mesh size, so tiny curves keep context


def waiting(mw):
    """Pairs with a cage curve still waiting for their target curve."""
    return [p for p in mw.pairs
            if P.has_side(p, 'CAGE') and not P.has_side(p, 'TARGET')]


_D = 0.70710678
# camera directions (from the surface toward the eye): front, 3/4s, sides,
# back and top; snapping to these keeps both views at readable, matching
# angles instead of odd tilts from an averaged normal
_VIEW_DIRS = [Vector(v) for v in (
    (0, -1, 0), (_D, -_D, 0), (-_D, -_D, 0), (1, 0, 0), (-1, 0, 0),
    (_D, _D, 0), (-_D, _D, 0), (0, 1, 0), (0, 0, 1))]


def _look_rotation(normal):
    """View rotation for a surface facing ``normal``: the nearest of the
    standard directions, world Z kept up."""
    n = Vector(normal)
    if n.length < 1e-9:
        return None
    n.normalize()
    z = max(_VIEW_DIRS, key=lambda d: d.dot(n)).copy()
    up = Vector((0.0, 0.0, 1.0))
    if abs(z.dot(up)) > 0.98:
        up = Vector((0.0, 1.0, 0.0))
    x = up.cross(z).normalized()
    y = z.cross(x)
    return Matrix((x, y, z)).transposed().to_quaternion()


def _curve_world(obj, depsgraph, pair, side):
    """(world points, world mean normal) of one half on its surface."""
    cos = P.coords(pair, side)
    if not cos:
        return None, None
    local, nrm = display.project_local(obj, depsgraph, cos)
    world = picking.to_world(obj, local)
    M3 = np.array(obj.matrix_world.to_3x3())
    n = np.asarray(nrm, float) @ np.linalg.inv(M3)   # normals: inverse-T
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return world, n.mean(axis=0)


def _size(obj):
    return max(obj.dimensions) or 1.0


def matched_points(mw):
    """(cage world, target world) matched samples of the finished pairs."""
    src, dst = [], []
    for pair in mw.pairs:
        if not (P.is_complete(pair) and pair.enabled):
            continue
        n = P.sample_count(pair)
        for side, out in (('CAGE', src), ('TARGET', dst)):
            cos = P.coords(pair, side)
            pts = (np.asarray(cos, float) if len(cos) == n
                   else sampling.resample(cos, n, pair.closed))
            out.append(picking.to_world(P.obj_for(mw, side), pts))
    if not src:
        return None, None
    return np.vstack(src), np.vstack(dst)


def _displace(mw, src, dst, pts):
    """Move cage world ``pts`` by the inverse-distance weighted
    displacement of the matched points ``src -> dst``."""
    soft = (0.05 * _size(mw.cage)) ** 2
    d2 = ((pts[:, None, :] - src[None, :, :]) ** 2).sum(axis=2)
    w = 1.0 / (d2 + soft) ** 2
    w /= w.sum(axis=1, keepdims=True)
    return pts + w @ (dst - src)


def map_to_target(mw, centre):
    """(target point, size ratio) for a cage world point, or (None, None).

    The point moves by the displacement of the nearby finished curves
    (inverse-distance weighted), which stays right even when only a couple
    of curves are done; a global fit from a few curves is easily skewed."""
    src, dst = matched_points(mw)
    if src is None or not len(src):
        return None, None
    moved = _displace(mw, src, dst, np.asarray(centre)[None, :])[0]
    ratio = None
    if len(src) >= 3:
        rs = np.sqrt(((src - src.mean(axis=0)) ** 2).sum(axis=1).mean())
        rd = np.sqrt(((dst - dst.mean(axis=0)) ** 2).sum(axis=1).mean())
        if rs > 1e-9:
            ratio = rd / rs
    return moved, ratio


def frame_pair(op, pair):
    """Frame ``pair``'s cage curve in the cage view and the matching spot
    in the target view."""
    mw = op.mw
    stage = op.stage
    _a, _r, cv = stage.view('CAGE')
    _a, _r, tv = stage.view('TARGET')
    if cv is None or tv is None:
        return
    dg = op.depsgraph()
    world, normal = _curve_world(mw.cage, dg, pair, 'CAGE')
    if world is None:
        return
    centre = world.mean(axis=0)
    radius = float(np.linalg.norm(world - centre, axis=1).max())
    radius = max(radius, MIN_RADIUS * _size(mw.cage))
    rot = _look_rotation(normal)
    cv.view_location = Vector(centre)
    cv.view_distance = radius * FRAME_MARGIN
    if rot is not None:
        cv.view_rotation = rot

    link = stage.link
    t_centre, ratio = map_to_target(mw, centre)
    if t_centre is not None:
        # same angle as the cage view, so the two can be compared by eye;
        # the centre isn't snapped to the surface (neither is the cage's:
        # a curved curve's centre sits inside the mesh)
        if ratio is None:
            lo, hi = bounds(mw.target)
            clo, chi = bounds(mw.cage)
            ratio = (hi - lo).length / max((chi - clo).length, 1e-9)
        tv.view_location = Vector(t_centre)
        tv.view_distance = cv.view_distance * ratio
        tv.view_rotation = cv.view_rotation
        tv.view_perspective = cv.view_perspective
    else:
        # nothing matched yet: the same world spot when the meshes overlap
        # (World Space sync), else the same spot relative to each mesh's
        # bounds
        clo, chi = bounds(mw.cage)
        tlo, thi = bounds(mw.target)
        overlap = all(clo[i] < thi[i] and tlo[i] < chi[i] for i in range(3))
        if world_sync() and overlap:
            tv.view_location = Vector(centre)
            tv.view_distance = cv.view_distance
        else:
            csize = np.maximum(np.array(chi - clo), 1e-9)
            rel = (centre - np.array(clo)) / csize
            tv.view_location = Vector(np.array(tlo) + rel * np.array(thi - tlo))
            ratio = (thi - tlo).length / max((chi - clo).length, 1e-9)
            tv.view_distance = cv.view_distance * ratio
        tv.view_rotation = cv.view_rotation
        tv.view_perspective = cv.view_perspective
    # both views moved on purpose: don't let the link copy one onto the other
    if hasattr(link, "_remember"):
        link._remember()
    stage.redraw()


CLEAR_FLIP = 0.7    # reverse only when that fits this much better


def orient(op, pair, hits):
    """``hits`` reordered so the new target curve runs the same way as the
    pair's cage curve.  The cage curve is carried over to the target by the
    finished curves around it (before any are finished: where it sits in
    its own view, the views being framed alike).  An open curve is only
    reversed when that is clearly the better fit, so a drawing that is
    hard to judge keeps the direction it was drawn in."""
    if len(hits) < 2:
        return hits
    mw = op.mw
    cage_pts = picking.to_world(mw.cage, P.coords(pair, 'CAGE'))
    tgt_pts = np.array([tuple(h.world) for h in hits])
    src, dst = matched_points(mw)
    if src is not None and len(src):
        ref = _displace(mw, src, dst, cage_pts)
        cand = tgt_pts
    else:
        _a, cr, cv = op.stage.view('CAGE')
        _a, tr, tv = op.stage.view('TARGET')
        if cv is None or tv is None:
            return hits
        c2, cf = picking.project(cr, cv, cage_pts)
        t2, tf = picking.project(tr, tv, tgt_pts)
        if not (cf.all() and tf.all()):
            return hits
        pad = lambda a: np.hstack([a, np.zeros((len(a), 1))])  # noqa: E731
        ref = pad(c2 / np.array([cr.width, cr.height]))
        cand = pad(t2 / np.array([tr.width, tr.height]))
    if pair.closed and len(hits) > 2:
        order = sampling.matching_order(ref, cand, True)
        return [hits[i] for i in order]
    keep = (np.linalg.norm(cand[0] - ref[0])
            + np.linalg.norm(cand[-1] - ref[-1]))
    flip = (np.linalg.norm(cand[-1] - ref[0])
            + np.linalg.norm(cand[0] - ref[-1]))
    return hits[::-1] if flip < CLEAR_FLIP * keep else hits


def fill_mirror_twin(op, pair):
    """With X symmetry on, a template usually holds both a curve and its
    mirror.  When ``pair`` is finished, give its waiting twin the mirrored
    target curve instead of making a new pair.  Returns the twin or None."""
    mw = op.mw
    cage = P.coords(pair, 'CAGE')
    target = P.coords(pair, 'TARGET')
    if len(cage) < 1 or not target:
        return None
    mc = snapping.mirror_x(cage)
    tol = (0.05 * float(np.ptp(np.asarray(cage), axis=0).max())
           + 0.005 * _size(mw.cage))
    for other in waiting(mw):
        if other == pair:
            continue
        oc = np.asarray(P.coords(other, 'CAGE'), float)
        if len(oc) != len(mc):
            continue
        fwd = float(np.linalg.norm(oc - mc, axis=1).mean())
        rev = float(np.linalg.norm(oc - mc[::-1], axis=1).mean())
        if min(fwd, rev) > tol:
            continue
        mt = snapping.mirror_x(target)
        if rev < fwd:
            mt = mt[::-1]
        mt = snapping.to_side(mw, 'TARGET', op.depsgraph(), mt)
        P.set_coords(other, 'TARGET', [tuple(map(float, p)) for p in mt])
        other.samples = len(mt)
        return other
    return None
