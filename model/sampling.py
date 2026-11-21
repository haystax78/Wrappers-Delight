"""Polyline maths shared by the editor and the solver (numpy only)."""

import numpy as np


def as_array(pts):
    return np.asarray(pts, dtype=float).reshape(-1, 3)


def cumulative_length(P, closed=False):
    """Arc length at each vertex (and, when closed, at the return to P[0])."""
    if closed and len(P) > 1:
        P = np.vstack([P, P[:1]])
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def resample(pts, count, closed=False):
    """``count`` points evenly spaced by arc length along the polyline.

    Open curves keep both end points.  Closed curves start at ``pts[0]`` and
    leave out the duplicate end point.
    """
    P = as_array(pts)
    if len(P) == 0 or count <= 0:
        return np.zeros((0, 3))
    if len(P) == 1 or count == 1:
        return np.repeat(P[:1], max(count, 1), axis=0)
    path = np.vstack([P, P[:1]]) if closed else P
    s = cumulative_length(P, closed)
    total = s[-1]
    if total <= 1e-12:
        return np.repeat(P[:1], count, axis=0)
    if closed:
        t = np.linspace(0.0, total, count, endpoint=False)
    else:
        t = np.linspace(0.0, total, count)
    out = np.empty((count, 3))
    for ax in range(3):
        out[:, ax] = np.interp(t, s, path[:, ax])
    return out


def densify(pts, closed=False, per_segment=6):
    """Insert evenly spaced points inside every segment (for drawing)."""
    P = as_array(pts)
    if len(P) < 2:
        return P
    path = np.vstack([P, P[:1]]) if closed else P
    t = np.linspace(0.0, 1.0, per_segment, endpoint=False)[:, None]
    a, b = path[:-1], path[1:]
    out = (a[:, None, :] * (1.0 - t) + b[:, None, :] * t).reshape(-1, 3)
    if not closed:
        out = np.vstack([out, path[-1:]])
    return out


def smooth(pts, passes=2, closed=False):
    """Light Laplacian smoothing that keeps open-curve end points fixed."""
    P = as_array(pts).copy()
    if len(P) < 3:
        return P
    for _ in range(passes):
        if closed:
            avg = 0.5 * (np.roll(P, 1, axis=0) + np.roll(P, -1, axis=0))
            P = 0.5 * P + 0.5 * avg
        else:
            P[1:-1] = 0.5 * P[1:-1] + 0.25 * (P[:-2] + P[2:])
    return P


def simplify(pts, tolerance, closed=False, max_points=64):
    """Ramer-Douglas-Peucker reduction of a dense stroke.

    The tolerance is relaxed until the result has at most ``max_points``.
    """
    P = as_array(pts)
    if len(P) <= 2:
        return P
    tol = max(float(tolerance), 1e-9)
    for _ in range(12):
        keep = _rdp_mask(P, tol, closed)
        if keep.sum() <= max_points:
            return P[keep]
        tol *= 1.6
    return resample(P, max_points, closed)


def _rdp_mask(P, tol, closed):
    n = len(P)
    keep = np.zeros(n, dtype=bool)
    if closed:
        # split the loop at the point farthest from the start
        far = int(np.argmax(np.linalg.norm(P - P[0], axis=1)))
        spans = [(0, far), (far, n - 1)]
        keep[[0, far, n - 1]] = True
    else:
        spans = [(0, n - 1)]
        keep[[0, n - 1]] = True
    stack = list(spans)
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = P[i], P[j]
        ab = b - a
        L = float(ab @ ab)
        seg = P[i + 1:j] - a
        if L <= 1e-18:
            d = np.linalg.norm(seg, axis=1)
        else:
            t = np.clip(seg @ ab / L, 0.0, 1.0)
            d = np.linalg.norm(seg - t[:, None] * ab, axis=1)
        k = int(np.argmax(d))
        if d[k] > tol:
            m = i + 1 + k
            keep[m] = True
            stack.append((i, m))
            stack.append((m, j))
    if closed and n > 2 and np.linalg.norm(P[-1] - P[0]) < tol:
        keep[-1] = False
    return keep


def nearest_segment(P, q, closed=False):
    """(segment index, parameter t, distance) of point ``q`` to polyline P."""
    P = as_array(P)
    path = np.vstack([P, P[:1]]) if closed else P
    a, b = path[:-1], path[1:]
    ab = b - a
    L = (ab * ab).sum(axis=1)
    L[L < 1e-18] = 1e-18
    t = np.clip(((q - a) * ab).sum(axis=1) / L, 0.0, 1.0)
    d = np.linalg.norm(a + t[:, None] * ab - q, axis=1)
    k = int(np.argmin(d))
    return k, float(t[k]), float(d[k])


# ------------------------------------------------- matching orientation ---

def normalized(P, lo, hi):
    """Map points into the unit box spanned by ``lo``/``hi``."""
    size = np.maximum(np.asarray(hi) - np.asarray(lo), 1e-9)
    return (as_array(P) - lo) / size.max()


def matching_order(reference, candidate, closed):
    """Index order that makes ``candidate`` run the same way as
    ``reference``.  Both must already be expressed in a comparable frame.

    Open curves may be reversed; closed loops are also rotated so they start
    next to the reference's first point.
    """
    R = as_array(reference)
    C = as_array(candidate)
    n = len(C)
    idx = np.arange(n)
    if len(R) < 2 or n < 2:
        return idx
    if not closed:
        keep = (np.linalg.norm(C[0] - R[0]) + np.linalg.norm(C[-1] - R[-1]))
        flip = (np.linalg.norm(C[-1] - R[0]) + np.linalg.norm(C[0] - R[-1]))
        return idx[::-1].copy() if flip < keep else idx
    best = None
    rs = resample(R, 16, closed=True)
    for order in (idx, idx[::-1]):
        start = int(np.argmin(np.linalg.norm(C[order] - R[0], axis=1)))
        rolled = np.roll(order, -start)
        cost = np.linalg.norm(resample(C[rolled], 16, closed=True) - rs,
                              axis=1).sum()
        if best is None or cost < best[0]:
            best = (cost, rolled)
    return best[1].copy()


# -------------------------------------------------------------- splines ---

def catmull_rom(pts, closed=False, per_segment=10):
    """Dense uniform Catmull-Rom curve passing through every point of ``pts``.

    Used for click-placed points, so the curve threads through exactly the
    points the artist clicked.  Fewer than three points are returned as is.
    """
    P = as_array(pts)
    n = len(P)
    if n < 3:
        return P.copy()
    if closed:
        def at(i):
            return P[i % n]
        segs = n
    else:
        def at(i):
            return P[min(max(i, 0), n - 1)]
        segs = n - 1
    t = np.linspace(0.0, 1.0, per_segment, endpoint=False)[:, None]
    t2, t3 = t * t, t * t * t
    out = []
    for i in range(segs):
        p0, p1, p2, p3 = at(i - 1), at(i), at(i + 1), at(i + 2)
        out.append(0.5 * (2.0 * p1 + (p2 - p0) * t
                          + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                          + (3.0 * p1 - p0 - 3.0 * p2 + p3) * t3))
    out = np.vstack(out)
    if not closed:
        out = np.vstack([out, P[-1:]])
    return out


def resample_stroke(pts, kind, closed, count):
    """``count`` evenly spaced points along a drawn stroke.

    'POINTS' strokes (clicked points) are threaded with a Catmull-Rom spline
    so the curve passes through every click; 'FREE' strokes (dragged) get
    light smoothing for hand tremor; 'BORDER' loops are used as they are.
    A closed path is resampled around the whole loop, seam not duplicated.
    """
    P = as_array(pts)
    if kind == 'POINTS':
        dense = catmull_rom(P, False, 10)
    elif kind == 'BORDER':
        dense = P
    else:
        dense = smooth(P, passes=2, closed=False)
    if closed and len(dense) >= 2:
        loop = np.vstack([dense, dense[:1]])
        return resample(loop, count + 1, False)[:count]
    return resample(dense, count, False)
