"""Draw correspondences on screen (shared by the editor and the overlay).

Curves are drawn in region pixels (POST_PIXEL): each stored point is
projected to the screen and the curve is a smooth Catmull-Rom line through
those screen points.  Nothing between the points is projected onto the
surface, so a curve is exactly as smooth as the points it passes through.

Occlusion is decided per point with a ray from the eye.  A segment is drawn
as visible when either end is visible, so a curve stays continuous where it
goes round the silhouette; fully hidden stretches are skipped, or drawn
faintly with x-ray on.  Projection and visibility are cached per region and
recomputed only when the view, the object or the points change.
"""



import bpy
import numpy as np
from mathutils import Vector

from ..model import pairs as P
from . import paint, picking

OUTLINE = (0.0, 0.0, 0.0, 0.6)
HIDDEN_ALPHA = 0.3
SMOOTH_SUBDIV = 6

_CACHE = {}       # (region pointer, side) -> (key, [HalfView])


class DrawOptions:
    __slots__ = ("controls", "xray", "hover", "dim_others", "active_uid")

    def __init__(self, controls=False, xray=False, hover=None,
                 dim_others=False, active_uid=None):
        self.controls = controls      # editing: show points larger
        self.xray = xray              # faint pass for occluded parts
        self.hover = hover            # (uid, point index) under the mouse
        self.dim_others = dim_others  # fade pairs other than the active one
        self.active_uid = active_uid


class HalfView:
    __slots__ = ("uid", "color", "enabled", "runs", "markers")

    def __init__(self, uid, color, enabled, runs, markers):
        self.uid = uid
        self.color = color
        self.enabled = enabled
        self.runs = runs          # [(screen points, visible)]
        self.markers = markers    # [(x, y) or None, visible]


def invalidate():
    _CACHE.clear()


# ---------------------------------------------------------------- geometry --

def smooth_polyline(pts, closed=False, subdiv=SMOOTH_SUBDIV):
    """Catmull-Rom curve through 2D points (display smoothing only)."""
    n = len(pts)
    if n < 3 or subdiv < 2:
        out = [tuple(p) for p in pts]
        if closed and n > 1:
            out.append(out[0])
        return out
    P2 = np.asarray(pts, dtype=float)
    if closed:
        ext = np.vstack([P2[-1:], P2, P2[:2]])
        spans = n
    else:
        ext = np.vstack([P2[:1], P2, P2[-1:]])
        spans = n - 1
    t = np.arange(subdiv, dtype=float)[:, None] / subdiv
    t2, t3 = t * t, t * t * t
    out = []
    for i in range(spans):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        seg = 0.5 * (2.0 * p1 + (p2 - p0) * t
                     + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                     + (3.0 * p1 - p0 - 3.0 * p2 + p3) * t3)
        out.extend(map(tuple, seg))
    out.append(tuple(P2[0] if closed else P2[-1]))
    return out


def visibility(obj, depsgraph, region, rv3d, world_pts):
    """Per point: True when nothing on ``obj`` hides it from the eye."""
    tol = 0.002 * _size(obj)
    return [picking.is_visible(obj, depsgraph, region, rv3d, Vector(w), tol)
            for w in world_pts]


def visible_runs(xy, front, vis, closed):
    """Split a projected curve into runs of equal visibility.

    A segment counts as visible when either end point is visible, which
    keeps the curve whole where it crosses the silhouette.  Returns
    [(points, loop, visible)]; ``loop`` marks a closed curve drawn whole.
    """
    n = len(xy)
    if n < 2:
        return []
    segs = [(i, (i + 1) % n) for i in range(n if closed else n - 1)]
    runs, cur, cur_vis = [], [], None
    for a, b in segs:
        if not (front[a] and front[b]):
            if len(cur) >= 2:
                runs.append((cur, False, cur_vis))
            cur, cur_vis = [], None
            continue
        sv = bool(vis[a] or vis[b])
        if cur_vis is None:
            cur, cur_vis = [tuple(xy[a]), tuple(xy[b])], sv
        elif sv == cur_vis:
            cur.append(tuple(xy[b]))
        else:
            runs.append((cur, False, cur_vis))
            cur, cur_vis = [tuple(xy[a]), tuple(xy[b])], sv
    if len(cur) >= 2:
        runs.append((cur, False, cur_vis))
    if closed and len(runs) == 1 and len(runs[0][0]) == n + 1:
        pts, _loop, v = runs[0]
        return [(pts[:-1], True, v)]
    return runs


def _size(obj):
    scale = max(abs(s) for s in obj.matrix_world.to_scale()) or 1.0
    return max(max(obj.dimensions) / scale, 1e-6)


def _view_key(region, rv3d):
    m = rv3d.perspective_matrix
    return (region.width, region.height,
            tuple(round(m[i][j], 5) for i in range(4) for j in range(4)))


def _matrix_key(obj):
    m = obj.matrix_world
    return tuple(round(m[i][j], 6) for i in range(4) for j in range(4))


# -------------------------------------------------------------------- draw --

def _halves(mw, side, obj, region, rv3d):
    items = []
    for pair in mw.pairs:
        if pair.hidden:
            continue
        cos = P.coords(pair, side)
        if not cos:
            continue
        items.append((pair.uid, tuple(cos), bool(pair.closed and len(cos) > 2),
                      tuple(pair.color), bool(pair.enabled)))
    key = (_view_key(region, rv3d), _matrix_key(obj), tuple(items))
    slot = (region.as_pointer(), side)
    hit = _CACHE.get(slot)
    if hit is not None and hit[0] == key:
        return hit[1]
    dg = bpy.context.evaluated_depsgraph_get()
    out = []
    for uid, cos, closed, color, enabled in items:
        world = picking.to_world(obj, cos)
        xy, front = picking.project(region, rv3d, world)
        vis = visibility(obj, dg, region, rv3d, world)
        runs = [(smooth_polyline(run, loop), v)
                for run, loop, v in visible_runs(xy, front, vis, closed)]
        markers = [((float(p[0]), float(p[1])) if f else None, bool(v))
                   for p, f, v in zip(xy, front, vis)]
        out.append(HalfView(uid, color, enabled, runs, markers))
    _CACHE[slot] = (key, out)
    return out


def draw_side(mw, side, obj, region, rv3d, opts):
    """Draw every visible pair's ``side`` half in this region."""
    if obj is None or region is None or rv3d is None:
        return
    halves = _halves(mw, side, obj, region, rv3d)
    if not halves:
        return
    # active pair last so it sits on top
    halves = sorted(halves, key=lambda h: h.uid == opts.active_uid)
    for h in halves:
        _draw_half(h, side, opts)


WHITE = (1.0, 1.0, 1.0, 1.0)


def _draw_half(h, side, opts):
    """Pair-coloured line with a dark outline and light dots at its points;
    a single-point pair is one larger dot.  Editing shows the points larger,
    the hovered one ringed in white."""
    active = h.uid == opts.active_uid
    alpha = 1.0
    if not h.enabled:
        alpha = 0.35
    elif opts.dim_others and opts.active_uid is not None and not active:
        alpha = 0.55
    base = h.color + (1.0,)
    width = paint.px(4.0 if active else 2.5)
    for pts, visible in h.runs:
        if not (visible or opts.xray):
            continue
        a = alpha * (1.0 if visible else HIDDEN_ALPHA)
        if visible:
            paint.polyline_2d(pts, paint.with_alpha(OUTLINE, OUTLINE[3] * a),
                              width + paint.px(2.5))
        paint.polyline_2d(pts, paint.with_alpha(base, a), width)

    single = len(h.markers) == 1
    if single:
        size, edge, col = (13.0 if active else 10.0), 4.0, base
    elif opts.controls:
        size, edge = (9.0 if active else 6.5), 3.5
        col = paint.mix(base, WHITE, 0.15)
    else:
        size, edge = (6.0 if active else 4.5), 0.0
        col = paint.mix(base, WHITE, 0.55)
    for i, (xy, visible) in enumerate(h.markers):
        if xy is None or not (visible or opts.xray):
            continue
        a = alpha * (1.0 if visible else HIDDEN_ALPHA)
        x, y = xy
        if opts.hover is not None and opts.hover == (h.uid, i):
            _dot(x, y, 17.0, (0.0, 0.0, 0.0, 0.8))
            _dot(x, y, 13.0, WHITE)
            _dot(x, y, 8.0, base)
            continue
        if edge and visible:
            _dot(x, y, size + edge, paint.with_alpha(OUTLINE, OUTLINE[3] * a))
        _dot(x, y, size, paint.with_alpha(col, a))


def _dot(x, y, diameter, color):
    paint.circle(x, y, paint.px(diameter) / 2.0, color, seg=14)


def dense_spline(world_pts, per_segment=10):
    """Catmull-Rom through world points (clamped ends), for live drawing."""
    from ..model import sampling
    return sampling.catmull_rom(world_pts, False, per_segment)


def ring_marker(x, y, r, color, width=1.4, fill=False):
    paint.ring(x, y, r, paint.with_alpha(OUTLINE, OUTLINE[3] * color[3]),
               width + paint.px(2.0), seg=16)
    paint.ring(x, y, r, color, width, seg=16)
    if fill:
        paint.circle(x, y, max(r - paint.px(2.5), 1.0), color, seg=12)



