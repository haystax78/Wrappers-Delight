"""The X symmetry line: where a mesh crosses the world X=0 plane.

Drawn on the meshes in the editor while X symmetry is on, so the artist can
see the centre line strokes snap to and can't cross.  The slice is computed
once per mesh (vectorised, on native-typed arrays so it stays quick on
multi-million-face targets) and kept as a GPU batch until the mesh or its
transform changes.
"""

import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix

from ..model import meshdata
from . import paint

COLOR = (1.0, 0.24, 0.32, 0.85)     # Blender's X-axis red
WIDTH = 1.75
DEPTH_BIAS = 2e-4                   # pull the line just in front of the surface

_CACHE = {}       # object name -> (key, batch or None)


def _key(obj, mesh):
    m = obj.matrix_world
    return (len(mesh.vertices), len(mesh.edges), len(mesh.polygons),
            tuple(round(m[i][j], 6) for i in range(4) for j in range(4)))


def plane_segments(obj, mesh):
    """World-space line segments (2k, 3) where ``mesh`` crosses X=0."""
    nv, ne, nl = len(mesh.vertices), len(mesh.edges), len(mesh.loops)
    if not nv or not ne or not nl:
        return np.zeros((0, 3), np.float32)
    M = np.array(obj.matrix_world, dtype=np.float64)
    co = meshdata.positions(mesh)
    x = co @ M[0, :3].astype(np.float32) + np.float32(M[0, 3])
    pos = x > 0.0

    ev = meshdata.edge_vertices(mesh)
    cross = pos[ev[:, 0]] != pos[ev[:, 1]]
    cidx = np.flatnonzero(cross)
    if not len(cidx):
        return np.zeros((0, 3), np.float32)
    a, b = ev[cidx, 0], ev[cidx, 1]
    xa, xb = x[a].astype(np.float64), x[b].astype(np.float64)
    t = (xa / (xa - xb))[:, None]
    local = co[a] + t * (co[b] - co[a])
    world = local @ M[:3, :3].T + M[:3, 3]
    world[:, 0] = 0.0
    point_of_edge = np.full(ne, -1, np.int64)
    point_of_edge[cidx] = np.arange(len(cidx))
    del ev, co, x, pos

    # pair up the crossing edges face by face
    edge_of_loop = meshdata.corner_edges(mesh)
    lidx = np.flatnonzero(cross[edge_of_loop])
    crossing_edges = edge_of_loop[lidx]
    del edge_of_loop
    starts = meshdata.face_starts(mesh)
    face = np.searchsorted(starts, lidx, side='right') - 1
    # faces crossed an even number of times: consecutive crossings pair up
    first = np.r_[True, face[1:] != face[:-1]]
    rank = np.arange(len(face)) - np.maximum.accumulate(
        np.where(first, np.arange(len(face)), 0))
    count = np.bincount(face, minlength=len(starts))[face]
    keep = (count % 2 == 0)
    sel = np.flatnonzero(keep & (rank % 2 == 0))
    sel = sel[sel + 1 < len(face)]
    sel = sel[face[sel + 1] == face[sel]]
    p0 = point_of_edge[crossing_edges[sel]]
    p1 = point_of_edge[crossing_edges[sel + 1]]
    segs = np.empty((2 * len(sel), 3), np.float32)
    segs[0::2] = world[p0]
    segs[1::2] = world[p1]
    return segs


def batch_for(obj, mesh):
    """Cached GPU batch of the symmetry line on ``obj`` (or None)."""
    key = _key(obj, mesh)
    hit = _CACHE.get(obj.name)
    if hit is not None and hit[0] == key:
        return hit[1]
    segs = plane_segments(obj, mesh)
    batch = None
    if len(segs):
        batch = batch_for_shader(paint.shader('POLYLINE_UNIFORM_COLOR'),
                                 'LINES', {"pos": segs})
    _CACHE[obj.name] = (key, batch)
    return batch


def draw(obj, mesh):
    """POST_VIEW: the symmetry line on ``obj``, depth-tested."""
    batch = batch_for(obj, mesh)
    if batch is None:
        return
    proj = gpu.matrix.get_projection_matrix()
    bias = Matrix.Identity(4)
    bias[2][2] = 1.0 - DEPTH_BIAS
    sh = paint.shader('POLYLINE_UNIFORM_COLOR')
    with gpu.matrix.push_pop_projection():
        gpu.matrix.load_projection_matrix(bias @ proj)
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('LESS_EQUAL')
        gpu.state.depth_mask_set(False)
        sh.bind()
        sh.uniform_float("color", COLOR)
        sh.uniform_float("lineWidth", WIDTH)
        sh.uniform_float("viewportSize", paint._viewport())
        batch.draw(sh)
        gpu.state.depth_test_set('NONE')
        gpu.state.blend_set('NONE')


def invalidate():
    _CACHE.clear()
