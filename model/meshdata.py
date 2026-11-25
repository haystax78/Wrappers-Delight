"""Fast bulk reads of mesh arrays.

``foreach_get`` copies integer mesh data (edge vertices, corner edges) one
element at a time through RNA, which takes seconds on a multi-million-face
scan.  Mesh attributes are stored as plain contiguous arrays, so the
integer ones are copied straight from memory here instead: the layout is
checked first (element stride and total span) and anything unexpected falls
back to ``foreach_get``.  The data is copied immediately, never kept as a
view into Blender's memory.
"""

import ctypes

import numpy as np


def _raw(collection, count, comps, ctype, dtype):
    if count == 0:
        return np.zeros((0, comps), dtype)
    try:
        p0 = collection[0].as_pointer()
        item = ctypes.sizeof(ctype) * comps
        if count > 1:
            if collection[1].as_pointer() - p0 != item:
                return None
            if collection[count - 1].as_pointer() - p0 != item * (count - 1):
                return None
        buf = (ctype * (count * comps)).from_address(p0)
        return np.ctypeslib.as_array(buf).reshape(count, comps).astype(
            dtype, copy=True)
    except (AttributeError, IndexError, ValueError, TypeError):
        return None


def int_attribute(mesh, name, count, comps=1):
    """(count, comps) int32 copy of an integer attribute such as
    ``.edge_verts`` (comps=2) or ``.corner_edge``."""
    attr = mesh.attributes.get(name)
    if attr is None or len(attr.data) != count:
        return None
    out = _raw(attr.data, count, comps, ctypes.c_int32, np.int32)
    if out is None:
        flat = np.empty(count * comps, np.int32)
        attr.data.foreach_get("value", flat)
        out = flat.reshape(count, comps)
    return out


def edge_vertices(mesh):
    """(edges, 2) int32 vertex indices of every edge."""
    n = len(mesh.edges)
    out = int_attribute(mesh, ".edge_verts", n, 2)
    if out is None:
        flat = np.empty(n * 2, np.int32)
        mesh.edges.foreach_get("vertices", flat)
        out = flat.reshape(n, 2)
    return out


def corner_edges(mesh):
    """(corners,) int32 edge index of every face corner."""
    n = len(mesh.loops)
    out = int_attribute(mesh, ".corner_edge", n, 1)
    if out is None:
        out = np.empty(n, np.int32)
        mesh.loops.foreach_get("edge_index", out)
        return out
    return out[:, 0]


def positions(mesh):
    """(verts, 3) float32 vertex positions (bulk copy of ``position``)."""
    n = len(mesh.vertices)
    out = np.empty(n * 3, np.float32)
    attr = mesh.attributes.get("position")
    if attr is not None and len(attr.data) == n:
        attr.data.foreach_get("vector", out)
    else:
        mesh.vertices.foreach_get("co", out)
    return out.reshape(n, 3)


def face_starts(mesh):
    """(faces,) int32 first corner of every face."""
    out = np.empty(len(mesh.polygons), np.int32)
    mesh.polygons.foreach_get("loop_start", out)
    return out


def corner_vertices(mesh):
    """(corners,) int32 vertex index of every face corner."""
    n = len(mesh.loops)
    out = int_attribute(mesh, ".corner_vert", n, 1)
    if out is None:
        out = np.empty(n, np.int32)
        mesh.loops.foreach_get("vertex_index", out)
        return out
    return out[:, 0]


def triangle_corners(mesh):
    """(triangles, 3) int32 corner indices (after calc_loop_triangles)."""
    n = len(mesh.loop_triangles)
    out = _raw(mesh.loop_triangles, n, 3, ctypes.c_int32, np.int32)
    # the raw layout is only trusted when it agrees with RNA
    if out is not None and n and all(
            tuple(out[i]) == tuple(mesh.loop_triangles[i].loops)
            for i in (0, n // 2, n - 1)):
        return out
    flat = np.empty(n * 3, np.int32)
    mesh.loop_triangles.foreach_get("loops", flat)
    return flat.reshape(n, 3)
