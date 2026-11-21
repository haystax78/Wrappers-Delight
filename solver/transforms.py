"""Global alignment and smooth landmark warping (numpy only)."""

import numpy as np


# ------------------------------------------------------------ similarity ---

class Similarity:
    """x -> scale * R x + offset."""

    __slots__ = ("scale", "rotation", "offset")

    def __init__(self, scale=1.0, rotation=None, offset=None):
        self.scale = float(scale)
        self.rotation = np.eye(3) if rotation is None else np.asarray(rotation)
        self.offset = np.zeros(3) if offset is None else np.asarray(offset)

    def apply(self, pts):
        pts = np.asarray(pts, dtype=float)
        return self.scale * (pts @ self.rotation.T) + self.offset


def fit_similarity(src, dst, weights=None, allow_scale=True):
    """Weighted least-squares similarity taking ``src`` onto ``dst``.

    Closed-form solution from the SVD of the weighted cross-covariance
    (Kabsch, extended with a uniform scale).  Reflections are excluded.
    """
    X = np.asarray(src, dtype=float)
    Y = np.asarray(dst, dtype=float)
    if len(X) == 0:
        return Similarity()
    w = np.ones(len(X)) if weights is None else np.asarray(weights, float)
    w = w / max(w.sum(), 1e-12)

    cx = w @ X
    cy = w @ Y
    Xc = X - cx
    Yc = Y - cy
    if len(X) == 1:
        return Similarity(1.0, np.eye(3), cy - cx)

    H = (Xc * w[:, None]).T @ Yc
    U, S, Vt = np.linalg.svd(H)
    flip = np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0
    D = np.array([1.0, 1.0, flip])
    R = (Vt.T * D) @ U.T

    scale = 1.0
    if allow_scale:
        spread = float(w @ (Xc * Xc).sum(axis=1))
        if spread > 1e-18:
            scale = float((S * D).sum()) / spread
    return Similarity(scale, R, cy - scale * (R @ cx))


# ----------------------------------------------------------------- warp ---

def _distances(A, B):
    """Euclidean distance matrix |A_i - B_j| without a (n, m, 3) temporary."""
    aa = (A * A).sum(axis=1)[:, None]
    bb = (B * B).sum(axis=1)[None, :]
    d2 = aa + bb - 2.0 * (A @ B.T)
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2)


def _merge_duplicates(C, V, tol):
    """Average the values of centres closer together than ``tol``."""
    keys = np.round(C / max(tol, 1e-12)).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True,
                                   return_counts=True)
    inverse = inverse.reshape(-1)
    m = len(counts)
    Cm = np.zeros((m, 3))
    Vm = np.zeros((m, 3))
    np.add.at(Cm, inverse, C)
    np.add.at(Vm, inverse, V)
    return Cm / counts[:, None], Vm / counts[:, None]


class LandmarkWarp:
    """Smooth 3D displacement field through a set of landmark offsets.

    Uses the polyharmonic radial basis ``phi(r) = r`` (the minimum-bending
    interpolant in 3D) plus an affine part.  ``relax`` trades exactness at
    the landmarks for smoothness; 0 interpolates them exactly.
    """

    def __init__(self, centers, offsets, relax=0.0):
        C = np.asarray(centers, dtype=float)
        V = np.asarray(offsets, dtype=float)
        span = float(np.ptp(C, axis=0).max()) if len(C) else 1.0
        span = span or 1.0
        C, V = _merge_duplicates(C, V, 1e-7 * span)
        n = len(C)
        self.centers = C

        K = _distances(C, C)
        lam = (relax ** 2) * span * 4.0
        K[np.diag_indices(n)] += lam + 1e-9 * span
        A = np.zeros((n + 4, n + 4))
        A[:n, :n] = K
        A[:n, n] = 1.0
        A[n, :n] = 1.0
        A[:n, n + 1:] = C
        A[n + 1:, :n] = C.T
        rhs = np.zeros((n + 4, 3))
        rhs[:n] = V
        try:
            sol = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
        self.weights = sol[:n]
        self.affine = sol[n:]          # row 0 constant, rows 1..3 linear

    def __call__(self, pts, block=4096):
        P = np.asarray(pts, dtype=float)
        out = np.empty_like(P)
        for i in range(0, len(P), block):
            Q = P[i:i + block]
            out[i:i + block] = (_distances(Q, self.centers) @ self.weights
                                + self.affine[0] + Q @ self.affine[1:])
        return out
