"""Pull a deformed cage onto a target surface.

Coarse to fine over a number of stages.  Each stage finds where every cage
vertex would like to go on the target (closest point, filtered so a vertex
only takes a surface that faces the same way it does; failing that, the
surface straight along its normal), then runs a few inner steps that move
vertices part of the way there while smoothing their POSITIONS with an
umbrella operator.  The smoothing is strong at first, so the cage moves as
a whole, and eases off stage by stage so it can follow finer detail.  From
the third stage on the smoothing only acts along the surface (its normal
component is removed): it keeps edge rows evenly spread over the target
without shrinking the cage.  That spreading is what keeps rows from piling
up in creases such as the fold above an eyelid.

A short finish snaps vertices exactly onto the surface, with tangential
smoothing passes in between to even out the spacing again.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class ConformParams:
    iterations: int = 8         # correspondence stages
    inner: int = 10             # pull + smooth steps per stage
    stiff_start: float = 0.92   # umbrella smoothing weight, first stage
    stiff_end: float = 0.18     # ... last stage (geometric in between)
    step: float = 0.6           # fraction of the way to the surface per step
    facing: float = 0.10        # min cos(cage normal, surface normal)
    relax: int = 2              # tangential passes between the final snaps
    reach: float = 0.1          # world distance searched for the surface
    reach_end: float = 0.45     # reach multiplier at the last stage
    facing_snap: float = -0.35  # facing still required by the final snaps
    fill: int = 4               # edge rings an unmatched vertex borrows from

    def schedule(self, k):
        """(smoothing weight, reach) for stage k."""
        t = k / max(self.iterations - 1, 1)
        stiff = self.stiff_start * (self.stiff_end / self.stiff_start) ** t
        reach = self.reach * (1.0 - (1.0 - self.reach_end) * t)
        return stiff, reach


PRESETS = {
    'QUICK': dict(iterations=4, inner=6, stiff_start=0.85, stiff_end=0.30,
                  step=0.70, facing=0.00, relax=2),
    'STANDARD': dict(iterations=8, inner=10, stiff_start=0.92,
                     stiff_end=0.18, step=0.60, facing=0.10, relax=2),
    'PRECISE': dict(iterations=14, inner=12, stiff_start=0.95,
                    stiff_end=0.10, step=0.50, facing=0.15, relax=2),
}


def custom_params(iterations, stiffness, detail, relax):
    """Map the panel's Custom settings onto the solver's parameters."""
    return dict(iterations=max(2, int(iterations)), inner=10,
                stiff_start=0.80 + 0.15 * stiffness,
                stiff_end=max(0.05, 0.45 - 0.37 * detail),
                step=0.60, facing=0.10, relax=int(relax))


def _robust(P, T, ok, reach):
    """Drop correspondences far beyond what the rest of the cage agreed on
    (a jaw vertex that grabbed an ear across a gap).  Median + MAD, with a
    floor so nothing close is ever thrown out."""
    idx = np.flatnonzero(ok)
    if len(idx) < 32:
        return ok
    d = np.linalg.norm(T[idx] - P[idx], axis=1)
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    limit = max(med + 6.0 * 1.4826 * mad, 3.0 * med, 0.05 * reach)
    ok = ok.copy()
    ok[idx[d > limit]] = False
    return ok


def _targets(P, N, probe, graph, reach, facing, active, use_rays, fill):
    """Surface target per active vertex, plus which ones are usable."""
    T, TN, ok = probe.nearest(P, reach, active)
    if facing > -1.0:
        ok &= (TN * N).sum(axis=1) >= facing
    if use_rays:
        retry = active & ~ok
        if retry.any():
            RT, RN, rok = probe.along(P, N, reach, retry)
            if facing > -1.0:
                rok &= (RN * N).sum(axis=1) >= facing
            T[rok] = RT[rok]
            ok |= rok
    ok = _robust(P, T, ok, reach)
    if fill and ok.any() and not ok.all():
        # an unmatched vertex takes the move its matched neighbours agree on
        # (within ``fill`` rings), instead of lagging behind as a spike
        D = np.where(ok[:, None], T - P, 0.0)
        moved, den = graph.blend(D, ok.astype(float), fill)
        new = active & ~ok & (den > 1e-9)
        T[new] = P[new] + moved[new]
        ok |= new
    return T, ok


class CurveLocks:
    """Points on the cage surface held on target positions.

    Each locked sample is a point inside a cage triangle (vertex indices +
    barycentric weights) with the world position it should sit at and a
    strength 0..1.  :meth:`enforce` moves the triangle's vertices so the
    point moves that fraction of the way to its target; repeated every
    solver step, strength 1 holds the point exactly while weaker locks
    settle where they balance against the smoothing.
    """

    def __init__(self, tris, bary, targets, strength):
        keep = np.asarray(strength, float) > 0.0
        self.tris = np.asarray(tris, np.int64).reshape(-1, 3)[keep]
        self.bary = np.asarray(bary, float).reshape(-1, 3)[keep]
        self.targets = np.asarray(targets, float).reshape(-1, 3)[keep]
        self.strength = np.asarray(strength, float)[keep]

    def __bool__(self):
        return len(self.strength) > 0

    def error(self, P):
        p = (P[self.tris] * self.bary[:, :, None]).sum(axis=1)
        return np.linalg.norm(self.targets - p, axis=1)

    def enforce(self, P, movable, passes=12):
        """Move ``movable`` vertices of ``P`` (in place) toward the locks."""
        if not self:
            return
        m = movable[self.tris]                       # (k, 3)
        b = np.where(m, self.bary, 0.0)
        den = (b * b).sum(axis=1)
        live = den > 1e-12
        if not live.any():
            return
        tris, b, bary = self.tris[live], b[live], self.bary[live]
        gain = (1.0 / den[live])[:, None]
        # this call moves each point ``strength`` of the way to its target
        p = (P[tris] * bary[:, :, None]).sum(axis=1)
        s = self.strength[live][:, None]
        goal = p + s * (self.targets[live] - p)
        flat = tris.ravel()
        n = len(P)
        cnt = np.bincount(flat, weights=b.ravel(), minlength=n)
        touched = cnt > 0
        for _ in range(passes):
            p = (P[tris] * bary[:, :, None]).sum(axis=1)
            e = (goal - p) * gain
            # smallest move of one triangle that carries its point by
            # strength * error is b_v * error / sum(b^2) per vertex; a vertex
            # shared by several locks takes the b-weighted mean of its moves
            mv = (b[:, :, None] * b[:, :, None] * e[:, None, :]).reshape(-1, 3)
            acc = np.stack([np.bincount(flat, weights=mv[:, c], minlength=n)
                            for c in range(3)], axis=1)
            P[touched] += acc[touched] / cnt[touched, None]


def _tangential(U, N):
    return U - (U * N).sum(axis=1, keepdims=True) * N


def conform(P0, graph, probe, params, attract=None, pins=None, frozen=None,
            locks=None):
    """Generator: yields ``(step, total, P)`` after every stage.

    attract  bool (n,) vertices that seek the surface (default: all);
             the rest are carried along by the smoothing.
    pins     {vertex index: world position} held fixed throughout.
    frozen   bool (n,) vertices that keep their starting position; their
             neighbours still see them, so the cage stays continuous.
    locks    :class:`CurveLocks` re-applied after every step.
    """
    P = np.array(P0, dtype=float)
    n = len(P)
    attract = np.ones(n, bool) if attract is None else attract.copy()
    pin_idx = (np.fromiter(pins.keys(), np.int64) if pins
               else np.zeros(0, np.int64))
    pin_pos = (np.array([pins[i] for i in pin_idx.tolist()])
               if len(pin_idx) else np.zeros((0, 3)))
    if len(pin_idx):
        P[pin_idx] = pin_pos
    free = np.ones(n, bool)
    free[pin_idx] = False
    if frozen is not None:
        free &= ~frozen
    attract &= free
    snaps = max(1, int(params.relax) + 1)
    total = params.iterations + snaps

    for k in range(params.iterations):
        stiff, reach = params.schedule(k)
        N = graph.normals(P)
        T, ok = _targets(P, N, probe, graph, reach, params.facing, attract,
                         use_rays=k >= 1, fill=params.fill)
        pull = ok & attract
        # the first stages may shrink a little to untangle folds; after
        # that the smoothing slides along the surface only
        along = k >= 2
        for _ in range(max(1, int(params.inner))):
            P[pull] += params.step * (T[pull] - P[pull])
            U = graph.umbrella(P)
            if along:
                U = _tangential(U, N)
            P[free] += stiff * U[free]
            if locks:
                locks.enforce(P, free, passes=3)
        yield k + 1, total, P

    # exact snaps with tangential evening-out in between
    _s, reach = params.schedule(params.iterations - 1)
    for j in range(snaps):
        N = graph.normals(P)
        T, ok = _targets(P, N, probe, graph, reach, params.facing_snap,
                         attract, use_rays=True, fill=params.fill)
        snap = ok & attract
        P[snap] = T[snap]
        if j < snaps - 1:
            U = _tangential(graph.umbrella(P), N)
            P[free] += 0.5 * U[free]
        if locks:
            locks.enforce(P, free)
        yield params.iterations + j + 1, total, P
