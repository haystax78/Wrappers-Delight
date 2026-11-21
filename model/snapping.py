"""Snap local-space points onto a mesh surface.

Cage points always refer to the cage's rest shape, so cage snapping uses a
tree built from the Basis coordinates even while the cage shows a wrap.
"""

import numpy as np

from ..solver.pipeline import rest_coords, rest_triangles
from ..solver.surface import tree_from_arrays
from . import display


def to_cage_rest(cage, pts):
    rest = rest_coords(cage)
    verts, tris = rest_triangles(cage, rest)
    if not len(tris):
        return np.asarray(pts, float)
    tree = tree_from_arrays(verts, tris)
    out = np.array(pts, dtype=float)
    for i, co in enumerate(out):
        loc = tree.find_nearest(co)[0]
        if loc is not None:
            out[i] = loc
    return out


def to_side(mw, side, depsgraph, pts):
    if side == 'CAGE':
        return to_cage_rest(mw.cage, pts)
    out, _n = display.project_local(mw.target, depsgraph, pts)
    return out


def mirror_x(pts):
    out = np.array(pts, dtype=float)
    out[:, 0] *= -1.0
    return out
