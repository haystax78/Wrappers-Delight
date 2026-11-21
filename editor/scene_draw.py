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
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

from ..model import pairs as P
from . import paint, picking

OUTLINE = (0.0, 0.0, 0.0, 0.6)
HIDDEN_ALPHA = 0.3
SMOOTH_SUBDIV = 6

_CACHE = {}       # (region pointer, side) -> (key, generation, [HalfView])
_BATCHES = {}     # (region pointer, side) -> ((generation, style), batches)
_GEN = [0]


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
    _BATCHES.clear()


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
    t = (np.arange(subdiv, dtype=float) / subdiv)[None, :, None]
    t2, t3 = t * t, t * t * t
    p0 = ext[0:spans][:, None, :]
    p1 = ext[1:spans + 1][:, None, :]
    p2 = ext[2:spans + 2][:, None, :]
    p3 = ext[3:spans + 3][:, None, :]
    seg = 0.5 * (2.0 * p1 + (p2 - p0) * t
                 + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                 + (3.0 * p1 - p0 - 3.0 * p2 + p3) * t3)
    out = seg.reshape(-1, 2)
    return np.vstack([out, (P2[0] if closed else P2[-1])[None, :]])


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
        return hit[1], hit[2]
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
    _GEN[0] += 1
    _CACHE[slot] = (key, _GEN[0], out)
    return _GEN[0], out


def draw_side(mw, side, obj, region, rv3d, opts):
    """Draw every visible pair's ``side`` half in this region."""
    if obj is None or region is None or rv3d is None:
        return
    gen, halves = _halves(mw, side, obj, region, rv3d)
    if not halves:
        return
    style = (opts.active_uid, opts.hover, bool(opts.xray), bool(opts.controls),
             bool(opts.dim_others), paint.px(1.0))
    slot = (region.as_pointer(), side)
    hit = _BATCHES.get(slot)
    if hit is None or hit[0] != (gen, style):
        # active pair last so it sits on top
        ordered = sorted(halves, key=lambda h: h.uid == opts.active_uid)
        items = []
        for h in ordered:
            items.extend(_build_half(h, opts))
        hit = ((gen, style), items)
        _BATCHES[slot] = hit
    _issue(hit[1])


WHITE = (1.0, 1.0, 1.0, 1.0)
DOT_SEGS = 14


def _build_half(h, opts):
    """GPU batches for one half, in drawing order: outline, line, dots.

    Same look as drawing each piece on its own: pair-coloured line with a
    dark outline and light dots at its points; a single-point pair is one
    larger dot; editing shows the points larger, the hovered one ringed in
    white.
    """
    active = h.uid == opts.active_uid
    alpha = 1.0
    if not h.enabled:
        alpha = 0.35
    elif opts.dim_others and opts.active_uid is not None and not active:
        alpha = 0.55
    base = h.color + (1.0,)
    width = paint.px(4.0 if active else 2.5)
    out = []

    seg_o, col_o, seg_l, col_l = [], [], [], []
    for pts, visible in h.runs:
        if not (visible or opts.xray) or len(pts) < 2:
            continue
        a = alpha * (1.0 if visible else HIDDEN_ALPHA)
        P2 = np.asarray(pts, dtype=np.float32)
        segs = np.empty((2 * (len(P2) - 1), 2), np.float32)
        segs[0::2] = P2[:-1]
        segs[1::2] = P2[1:]
        if visible:
            seg_o.append(segs)
            col_o.append(np.tile(paint.with_alpha(OUTLINE, OUTLINE[3] * a),
                                 (len(segs), 1)))
        seg_l.append(segs)
        col_l.append(np.tile(paint.with_alpha(base, a), (len(segs), 1)))
    if seg_o:
        out.append(_lines(seg_o, col_o, width + paint.px(2.5)))
    if seg_l:
        out.append(_lines(seg_l, col_l, width))

    single = len(h.markers) == 1
    if single:
        size, edge, col = (13.0 if active else 10.0), 4.0, base
    elif opts.controls:
        size, edge = (9.0 if active else 6.5), 3.5
        col = paint.mix(base, WHITE, 0.15)
    else:
        size, edge = (6.0 if active else 4.5), 0.0
        col = paint.mix(base, WHITE, 0.55)
    centres, radii, colors = [], [], []
    for i, (xy, visible) in enumerate(h.markers):
        if xy is None or not (visible or opts.xray):
            continue
        a = alpha * (1.0 if visible else HIDDEN_ALPHA)
        if opts.hover is not None and opts.hover == (h.uid, i):
            for d, c in ((17.0, (0.0, 0.0, 0.0, 0.8)), (13.0, WHITE),
                         (8.0, base)):
                centres.append(xy)
                radii.append(paint.px(d) / 2.0)
                colors.append(c)
            continue
        if edge and visible:
            centres.append(xy)
            radii.append(paint.px(size + edge) / 2.0)
            colors.append(paint.with_alpha(OUTLINE, OUTLINE[3] * a))
        centres.append(xy)
        radii.append(paint.px(size) / 2.0)
        colors.append(paint.with_alpha(col, a))
    if centres:
        out.append(_discs(centres, radii, colors))
    return out


def _lines(seg_list, col_list, width):
    pos = np.vstack(seg_list)
    pos3 = np.zeros((len(pos), 3), np.float32)
    pos3[:, :2] = pos
    sh = paint.shader('POLYLINE_FLAT_COLOR')
    batch = batch_for_shader(sh, 'LINES', {"pos": pos3,
                                           "color": np.vstack(col_list)
                                           .astype(np.float32)})
    return ('line', sh, batch, width)


_RING = np.stack([np.cos(np.linspace(0.0, 2.0 * np.pi, DOT_SEGS + 1)),
                  np.sin(np.linspace(0.0, 2.0 * np.pi, DOT_SEGS + 1))], 1)


def _discs(centres, radii, colors):
    """Filled circles as one triangle list (drawn in the given order)."""
    C = np.asarray(centres, np.float32)
    R = np.asarray(radii, np.float32)
    n = len(C)
    rim = C[:, None, :] + R[:, None, None] * _RING[None, :, :]   # n, s+1, 2
    tris = np.empty((n, DOT_SEGS, 3, 3), np.float32)
    tris[..., 2] = 0.0
    tris[:, :, 0, :2] = C[:, None, :]
    tris[:, :, 1, :2] = rim[:, :-1]
    tris[:, :, 2, :2] = rim[:, 1:]
    col = np.repeat(np.asarray(colors, np.float32), DOT_SEGS * 3, axis=0)
    sh = paint.shader('FLAT_COLOR')
    batch = batch_for_shader(sh, 'TRIS', {"pos": tris.reshape(-1, 3),
                                          "color": col})
    return ('tris', sh, batch, 0.0)


def _issue(items):
    if not items:
        return
    gpu.state.blend_set('ALPHA')
    vp = paint._viewport()
    for kind, sh, batch, width in items:
        sh.bind()
        if kind == 'line':
            sh.uniform_float("lineWidth", width)
            sh.uniform_float("viewportSize", vp)
        batch.draw(sh)
    gpu.state.blend_set('NONE')


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



