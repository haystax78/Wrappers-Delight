"""Pull a deformed cage onto a target surface.

Each iteration finds where every cage vertex would like to go (its closest
point on the target, or failing that the surface straight along its normal),
turns those wishes into a displacement field, and smooths the FIELD rather
than the positions.  Smoothing displacements keeps the cage's own shape and
edge flow intact while it moves; smoothing positions would shrink it.

The amount of smoothing falls from iteration to iteration: early on the
field is nearly rigid and the cage moves as a whole, later it is allowed to
follow finer and finer detail.  A last pass snaps vertices exactly onto the
surface and evens out edge lengths along it.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class ConformParams:
    iterations: int = 10
    stiffness: float = 0.5      # 0..1, smoothing of the early field
    detail: float = 0.6         # 0..1, how little the late field is smoothed
    step: float = 0.65          # fraction of the field applied per iteration
    relax: int = 2              # tangential relax passes after the snap
    reach: float = 0.1          # world distance searched for the surface
    reach_end: float = 0.35     # reach multiplier at the last iteration
    facing_start: float = 0.0   # min cos(cage normal, surface normal)
    facing_end: float = 0.4
    facing_snap: float = -0.3

    def schedule(self, k):
        """(smoothing passes, reach, facing) for iteration k."""
        t = k / max(self.iterations - 1, 1)
        first = 8.0 + 72.0 * self.stiffness
        last = 1.0 + 6.0 * (1.0 - self.detail)
        passes = int(round(first * (last / first) ** t))
        reach = self.reach * (1.0 + (self.reach_end - 1.0) * t)
        facing = self.facing_start + (self.facing_end - self.facing_start) * t
        return passes, reach, facing


PRESETS = {
    'QUICK': dict(iterations=5, stiffness=0.5, detail=0.4, step=0.75,
                  relax=1),
    'STANDARD': dict(iterations=10, stiffness=0.5, detail=0.6, step=0.65,
                     relax=2),
    'PRECISE': dict(iterations=16, stiffness=0.6, detail=0.85, step=0.55,
                    relax=3),
}


def _targets(P, N, probe, reach, facing, active, use_rays, robust=True):
    """Surface targets for the active vertices, filtered by facing."""
    T, TN, ok = probe.nearest(P, reach, active)
    ok &= (TN * N).sum(axis=1) >= facing
    if use_rays:
        retry = active & ~ok
        if retry.any():
            RT, RN, rok = probe.along(P, N, reach, retry)
            rok &= (RN * N).sum(axis=1) >= facing
            T[rok] = RT[rok]
            ok |= rok
    # drop wild outliers: a vertex that found a surface far further away
    # than its peers probably grabbed the wrong part of the target.  Never
    # below half the reach, so stragglers of a converged fit still snap.
    if robust and ok.sum() > 16:
        d = np.linalg.norm(T[ok] - P[ok], axis=1)
        limit = max(4.0 * float(np.median(d)), 0.5 * reach)
        bad = np.flatnonzero(ok)[d > limit]
        ok[bad] = False
    return T, ok


def conform(P0, graph, probe, params, attract=None, pins=None, frozen=None):
    """Generator: yields ``(step, total, P)`` after every stage.

    attract  bool (n,) vertices that seek the surface (default: all);
             the rest are carried along by their neighbours.
    pins     {vertex index: world position} held fixed throughout.
    frozen   bool (n,) vertices that keep their starting position and
             neither seek the surface nor influence their neighbours.
    """
    P = np.array(P0, dtype=float)
    n = len(P)
    attract = np.ones(n, bool) if attract is None else attract.copy()
    pin_idx = np.fromiter(pins.keys(), np.int64) if pins else np.zeros(0, np.int64)
    pin_pos = (np.array([pins[i] for i in pin_idx.tolist()])
               if len(pin_idx) else np.zeros((0, 3)))
    free = np.ones(n, bool)
    free[pin_idx] = False
    if frozen is not None:
        free &= ~frozen
    attract &= free
    total = params.iterations + 1 + params.relax

    for k in range(params.iterations):
        passes, reach, facing = params.schedule(k)
        N = graph.normals(P)
        T, ok = _targets(P, N, probe, reach, facing, attract, use_rays=k > 0)
        D = np.where(ok[:, None], T - P, 0.0)
        w = ok.astype(float)
        if len(pin_idx):
            D[pin_idx] = pin_pos - P[pin_idx]
            w[pin_idx] = 4.0
        field, _den = graph.blend(D, w, passes)
        P[free] += params.step * field[free]
        P[pin_idx] = pin_pos
        yield k + 1, total, P

    # exact snap; vertices that could not snap inherit their neighbours' move
    _p, reach, _f = params.schedule(params.iterations - 1)
    N = graph.normals(P)
    T, ok = _targets(P, N, probe, reach, params.facing_snap, attract, True,
                     robust=False)
    D = np.where(ok[:, None], T - P, 0.0)
    fill, den = graph.blend(D, ok.astype(float), 3)
    carry = free & ~ok & (den > 1e-6)
    P[ok] = T[ok]
    P[carry] += fill[carry]
    P[pin_idx] = pin_pos
    yield params.iterations + 1, total, P

    # even out edge lengths along the surface, then re-snap
    for r in range(params.relax):
        N = graph.normals(P)
        U = graph.umbrella(P)
        U -= (U * N).sum(axis=1, keepdims=True) * N
        P[ok] += 0.5 * U[ok]
        T, _TN, hit = probe.nearest(P, reach, ok)
        P[hit] = T[hit]
        yield params.iterations + 2 + r, total, P
