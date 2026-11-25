"""GPU side of Texture Bake's Resample mode (compute shaders).

Two independent parts, so each can fall back on its own:

* :class:`Tracer` builds the lookup: the destination triangles are drawn in
  UV space (world position and normal per texel), then a compute shader
  casts each texel's ray against a BVH of the source triangles, passing
  through faces whose normal is off by more than the limit, and writes the
  source UV it lands on.
* :class:`Sampler` pulls an image through a lookup (bilinear, repeat) and
  box-filters the supersampled texels, in one step.

A lookup is one RG32F texture holding a source UV per destination texel.
Margin texels hold the UV of the covered texel they copy, so sampling needs
no separate margin step.  Two marker values in the first channel: ``MISS``
(inside an island, ray hit nothing) and ``OUTSIDE`` (not covered); both
sample as zero.

Everything is dispatched in tiles with a readback in between: that keeps
single GPU jobs short (Windows resets the driver on long ones) and the
working textures small.

The BVH is implicit: triangles sorted along a Morton curve, four per leaf,
four children per node, so it is built with a sort and a few reductions and
stays quick on multi-million-triangle scans.  Triangles and nodes are stored
in textures read as images (storage buffers only exist since Blender 5.3).
"""

import math

import numpy as np

import bpy
import gpu

from ..model import meshdata

MISS = 1.0e30
OUTSIDE = 2.0e30

TRACE_TILE = 1024       # destination texels per side, one raster + trace
SAMPLE_TILE = 2048      # output texels per side, one sampling dispatch
GROUP = 8
LEAF = 4                # triangles per BVH leaf
FAN = 4                 # children per BVH node
STACK = 48
DATA_W = 8192           # width of the textures holding triangles and nodes
# Blender's bake rasteriser samples texel x at uv * w = x + 0.501 (0.502 in
# y), which breaks ties for UV edges lying exactly on texel centres.
PIXEL_BIAS = np.array((0.001, 0.002))


class GpuError(Exception):
    """The GPU can't do this here; the caller uses the CPU path."""


# ----------------------------------------------------------- availability --

# None: not tried, "": usable, else the reason it isn't
_ready = None
_trace_ready = None


def unavailable():
    """'' when images can be sampled on the GPU, otherwise why not."""
    global _ready
    if _ready is None:
        _ready = _probe()
    return _ready


def trace_unavailable():
    """'' when lookups can be ray cast on the GPU, otherwise why not (call
    after :func:`unavailable`)."""
    global _trace_ready
    if _trace_ready is None:
        try:
            _trace_self_test()
            _trace_ready = ""
        except Exception as exc:  # noqa: BLE001
            _trace_ready = f"{type(exc).__name__}: {exc}"
    return _trace_ready


def _probe():
    try:
        if bpy.app.background:
            if not hasattr(gpu, "init"):
                return "no GPU access in background mode"
            gpu.init()
        if not gpu.capabilities.compute_shader_support_get():
            return "no compute shader support"
        _self_test()
    except Exception as exc:  # noqa: BLE001 - any failure means "use the CPU"
        return f"{type(exc).__name__}: {exc}"
    return ""


# ---------------------------------------------------------------- shaders --

_shaders = {}


def _shader(name):
    sh = _shaders.get(name)
    if sh is None:
        sh = _shaders[name] = _BUILD[name]()
    return sh


_SAMPLE = """
vec4 wrapped(ivec2 p, ivec2 s)
{
  /* not %: undefined in GLSL for negative values */
  p -= s * ivec2(floor(vec2(p) / vec2(s)));
  return imageLoad(src, p);
}

void main()
{
  ivec2 t = ivec2(gl_GlobalInvocationID.xy);
  if (any(greaterThanEqual(t, size))) {
    return;
  }
  ivec2 o = (t + origin) * ss;
  ivec2 s = src_size;
  vec4 acc = vec4(0.0);
  for (int j = 0; j < ss; j++) {
    for (int i = 0; i < ss; i++) {
      vec2 uv = texelFetch(lookup, o + ivec2(i, j), 0).xy;
      if (uv.x > 0.5e30) {
        continue;
      }
      vec2 p = uv * vec2(s) - 0.5;
      vec2 f = floor(p);
      vec2 w = p - f;
      ivec2 b = ivec2(f);
      acc += mix(mix(wrapped(b, s), wrapped(b + ivec2(1, 0), s), w.x),
                 mix(wrapped(b + ivec2(0, 1), s),
                     wrapped(b + ivec2(1, 1), s), w.x), w.y);
    }
  }
  imageStore(dst, t, acc / float(ss * ss));
}
"""


def _build_sample():
    info = gpu.types.GPUShaderCreateInfo()
    info.local_group_size(GROUP, GROUP)
    # One sampler per compute shader, other inputs as images: with two,
    # Blender's OpenGL backend binds the wrong one depending on their names.
    info.sampler(0, 'FLOAT_2D', "lookup")
    info.image(0, 'RGBA32F', 'FLOAT_2D', "dst", qualifiers={'WRITE'})
    info.image(1, 'RGBA32F', 'FLOAT_2D', "src", qualifiers={'READ'})
    info.push_constant('IVEC2', "src_size")
    info.push_constant('IVEC2', "origin")
    info.push_constant('IVEC2', "size")
    info.push_constant('INT', "ss")
    info.compute_source(_SAMPLE)
    return gpu.shader.create_from_info(info)


# One ring of margin: an uncovered texel takes the UV of its first covered
# neighbour, straight ones before diagonal ones.
_MARGIN = """
void main()
{
  ivec2 t = ivec2(gl_GlobalInvocationID.xy) + origin;
  ivec2 s = textureSize(prev, 0);
  if (any(greaterThanEqual(t, s))) {
    return;
  }
  vec2 v = texelFetch(prev, t, 0).xy;
  if (v.x > 1.5e30) {
    ivec2 nb[8] = ivec2[8](ivec2(1, 0), ivec2(-1, 0), ivec2(0, 1),
                           ivec2(0, -1), ivec2(1, 1), ivec2(-1, 1),
                           ivec2(1, -1), ivec2(-1, -1));
    for (int k = 0; k < 8; k++) {
      ivec2 q = t + nb[k];
      if (any(lessThan(q, ivec2(0))) || any(greaterThanEqual(q, s))) {
        continue;
      }
      vec2 w = texelFetch(prev, q, 0).xy;
      if (w.x < 1.5e30) {
        v = w;
        break;
      }
    }
  }
  imageStore(next, t, vec4(v, 0.0, 0.0));
}
"""


def _build_margin():
    info = gpu.types.GPUShaderCreateInfo()
    info.local_group_size(GROUP, GROUP)
    info.sampler(0, 'FLOAT_2D', "prev")
    info.image(0, 'RG32F', 'FLOAT_2D', "next", qualifiers={'WRITE'})
    info.push_constant('IVEC2', "origin")
    info.compute_source(_MARGIN)
    return gpu.shader.create_from_info(info)


# The rasteriser snaps vertices to a fraction of a texel; the screen-space
# gradient moves position and normal back onto the exact texel centre.
_RASTER_VERT = """
void main()
{
  v_px = px;
  v_pos = pos;
  v_nrm = nrm;
  gl_Position = vec4(px / tile * 2.0 - 1.0, 0.0, 1.0);
}
"""
_RASTER_FRAG = """
void main()
{
  vec2 d = gl_FragCoord.xy - v_px;
  if (any(greaterThan(abs(d), vec2(0.5)))) {
    d = vec2(0.0);
  }
  out_pos = vec4(v_pos + dFdx(v_pos) * d.x + dFdy(v_pos) * d.y, 1.0);
  out_nrm = vec4(v_nrm + dFdx(v_nrm) * d.x + dFdy(v_nrm) * d.y, 1.0);
}
"""


def _build_raster():
    iface = gpu.types.GPUStageInterfaceInfo("wd_resample_raster")
    iface.smooth('VEC2', "v_px")
    iface.smooth('VEC3', "v_pos")
    iface.smooth('VEC3', "v_nrm")
    info = gpu.types.GPUShaderCreateInfo()
    info.push_constant('VEC2', "tile")
    info.vertex_in(0, 'VEC2', "px")
    info.vertex_in(1, 'VEC3', "pos")
    info.vertex_in(2, 'VEC3', "nrm")
    info.vertex_out(iface)
    info.fragment_out(0, 'VEC4', "out_pos")
    info.fragment_out(1, 'VEC4', "out_nrm")
    info.vertex_source(_RASTER_VERT)
    info.fragment_source(_RASTER_FRAG)
    return gpu.shader.create_from_info(info)


# Triangles: 4 texels each, (corner xyz, u) x 3 and (v0, v1, v2, -).
# Nodes: 2 texels each, (box min, first) and (box max, count); a node with a
# positive count has that many children starting at node ``first``, a leaf
# has -count triangles starting at ``first * LEAF``.
_TRACE = """
vec4 tri_texel(int t, int k)
{
  int i = t * 4 + k;
  return imageLoad(tris, ivec2(i % DATA_W, i / DATA_W));
}

vec4 node_texel(int n, int k)
{
  int i = n * 2 + k;
  return imageLoad(nodes, ivec2(i % DATA_W, i / DATA_W));
}

bool box_hit(vec3 bmin, vec3 bmax, vec3 ro, vec3 inv, float tmax)
{
  vec3 lo = (bmin - ro) * inv;
  vec3 hi = (bmax - ro) * inv;
  vec3 a = min(lo, hi), b = max(lo, hi);
  float t0 = max(max(a.x, a.y), max(a.z, 0.0));
  float t1 = min(min(b.x, b.y), min(b.z, tmax));
  return t0 <= t1 * 1.000001 + 1e-9;
}

void main()
{
  ivec2 t = ivec2(gl_GlobalInvocationID.xy);
  if (any(greaterThanEqual(t, size))) {
    return;
  }
  vec2 res = vec2(2.0e30, 0.0);
  vec4 P = texelFetch(g_pos, t, 0);
  if (P.w > 0.5) {
    res.x = 1.0e30;
    vec3 N = normalize(imageLoad(g_nrm, t).xyz);
    vec3 ro = P.xyz + N * ext;
    vec3 rd = -N;
    vec3 safe = mix(rd, vec3(1e-12), lessThan(abs(rd), vec3(1e-12)));
    vec3 inv = 1.0 / safe;
    float best = reach;
    int best_tri = -1;
    vec2 best_bc = vec2(0.0);

    int stack[STACK];
    int top = 0;
    stack[top++] = root;
    while (top > 0) {
      int n = stack[--top];
      vec4 bmin = node_texel(n, 0);
      vec4 bmax = node_texel(n, 1);
      if (!box_hit(bmin.xyz, bmax.xyz, ro, inv, best)) {
        continue;
      }
      int first = int(bmin.w);
      int cnt = int(bmax.w);
      if (cnt > 0) {
        for (int k = 0; k < cnt && top < STACK; k++) {
          stack[top++] = first + k;
        }
        continue;
      }
      for (int k = 0; k < -cnt; k++) {
        int tri = first * LEAF + k;
        vec3 v0 = tri_texel(tri, 0).xyz;
        vec3 e1 = tri_texel(tri, 1).xyz - v0;
        vec3 e2 = tri_texel(tri, 2).xyz - v0;
        vec3 pv = cross(rd, e2);
        float det = dot(e1, pv);
        if (abs(det) < 1e-30) {
          continue;
        }
        float f = 1.0 / det;
        vec3 s = ro - v0;
        float u = dot(s, pv) * f;
        if (u < -EPS || u > 1.0 + EPS) {
          continue;
        }
        vec3 q = cross(s, e1);
        float v = dot(rd, q) * f;
        if (v < -EPS || u + v > 1.0 + EPS) {
          continue;
        }
        float d = dot(e2, q) * f;
        if (d < 0.0 || d > best) {
          continue;
        }
        vec3 ng = cross(e1, e2);
        if (dot(ng, N) < cos_lim * length(ng)) {
          continue;            /* facing too far away: pass through */
        }
        best = d;
        best_tri = tri;
        best_bc = vec2(u, v);
      }
    }
    if (best_tri >= 0) {
      vec3 bc = vec3(1.0 - best_bc.x - best_bc.y, best_bc);
      vec3 us = vec3(tri_texel(best_tri, 0).w, tri_texel(best_tri, 1).w,
                     tri_texel(best_tri, 2).w);
      res = vec2(dot(us, bc), dot(tri_texel(best_tri, 3).xyz, bc));
    }
  }
  imageStore(lookup, t + origin, vec4(res, 0.0, 0.0));
}
"""


def _build_trace():
    info = gpu.types.GPUShaderCreateInfo()
    info.define("STACK", str(STACK))
    info.define("LEAF", str(LEAF))
    info.define("DATA_W", str(DATA_W))
    info.define("EPS", "1e-5")
    info.local_group_size(GROUP, GROUP)
    info.sampler(0, 'FLOAT_2D', "g_pos")
    info.image(1, 'RGBA32F', 'FLOAT_2D', "g_nrm", qualifiers={'READ'})
    info.image(0, 'RG32F', 'FLOAT_2D', "lookup", qualifiers={'WRITE'})
    info.image(2, 'RGBA32F', 'FLOAT_2D', "tris", qualifiers={'READ'})
    info.image(3, 'RGBA32F', 'FLOAT_2D', "nodes", qualifiers={'READ'})
    info.push_constant('IVEC2', "origin")
    info.push_constant('IVEC2', "size")
    info.push_constant('FLOAT', "ext")
    info.push_constant('FLOAT', "reach")
    info.push_constant('FLOAT', "cos_lim")
    info.push_constant('INT', "root")
    info.compute_source(_TRACE)
    return gpu.shader.create_from_info(info)


_BUILD = {"sample": _build_sample, "margin": _build_margin,
          "raster": _build_raster, "trace": _build_trace}


# ---------------------------------------------------------------- helpers --

def _buffer(a):
    a = np.ascontiguousarray(a, np.float32).ravel()
    return gpu.types.Buffer('FLOAT', a.size, a)


def _texture(w, h, fmt, data=None):
    limit = gpu.capabilities.max_texture_size_get()
    if max(w, h) > limit:
        raise GpuError(f"{w} x {h} is over this GPU's texture limit")
    if data is None:
        return gpu.types.GPUTexture((w, h), format=fmt)
    return gpu.types.GPUTexture((w, h), format=fmt, data=_buffer(data))


def _data_texture(texels):
    """(n, 4) float32 values as a DATA_W wide RGBA32F texture."""
    n = len(texels)
    rows = (n + DATA_W - 1) // DATA_W
    full = np.zeros((rows * DATA_W, 4), np.float32)
    full[:n] = texels
    return _texture(DATA_W, rows, 'RGBA32F', full)


def _read(tex):
    """Texture contents as float32 (h, w, channels)."""
    buf = tex.read()
    return np.array(buf, dtype=np.float32).reshape(tex.height, tex.width, -1)


def _groups(n):
    return (n + GROUP - 1) // GROUP


def _tiles(w, h, size):
    for y in range(0, h, size):
        for x in range(0, w, size):
            yield x, y, min(size, w - x), min(size, h - y)


class _Fence:
    """Wait for the GPU: reading any texture back flushes what's queued."""

    def __init__(self):
        self.tex = _texture(1, 1, 'R32F')

    def wait(self):
        self.tex.read()


# ----------------------------------------------------------------- lookup --

class Lookup:
    """Source UV per destination texel (see the module docstring)."""

    def __init__(self, tex):
        self.tex = tex

    @property
    def size(self):
        return self.tex.width, self.tex.height

    @classmethod
    def from_texels(cls, bw, bh, idx, uv, mdst, msrc):
        """From a CPU lookup: hit texels and their UVs, margin copies."""
        m = np.zeros((bh * bw, 2), np.float32)
        m[:, 0] = OUTSIDE
        m[idx] = uv
        m[mdst] = m[msrc]
        return cls(_texture(bw, bh, 'RG32F', m))

    def grow(self, radius):
        """Extend the islands ``radius`` texels outward."""
        if radius <= 0:
            return
        w, h = self.size
        sh = _shader("margin")
        a, b = self.tex, _texture(w, h, 'RG32F')
        fence = _Fence()
        for _ in range(radius):
            sh.bind()
            sh.uniform_sampler("prev", a)
            sh.image("next", b)
            for x, y, tw, th in _tiles(w, h, 4096):
                sh.uniform_int("origin", (x, y))
                gpu.compute.dispatch(sh, _groups(tw), _groups(th), 1)
            fence.wait()
            a, b = b, a
        self.tex = a


class Sampler:
    @staticmethod
    def sample(px, lookup, w, h, ss):
        """``px`` (float32 (sh, sw, 4), bottom-up) through ``lookup``, box
        filtered by ``ss``: float32 (h, w, 4)."""
        sh = _shader("sample")
        src = _texture(px.shape[1], px.shape[0], 'RGBA32F', px)
        out = np.empty((h, w, 4), np.float32)
        tiles = {}
        for x, y, tw, th in _tiles(w, h, SAMPLE_TILE):
            dst = tiles.get((tw, th))
            if dst is None:
                dst = tiles[(tw, th)] = _texture(tw, th, 'RGBA32F')
            sh.bind()
            sh.image("src", src)
            sh.uniform_int("src_size", (src.width, src.height))
            sh.uniform_sampler("lookup", lookup.tex)
            sh.image("dst", dst)
            sh.uniform_int("origin", (x, y))
            sh.uniform_int("size", (tw, th))
            sh.uniform_int("ss", ss)
            gpu.compute.dispatch(sh, _groups(tw), _groups(th), 1)
            out[y:y + th, x:x + tw] = _read(dst)
        return out


# -------------------------------------------------------------- geometry --

def _spread(v):
    """Space the low 21 bits of each value out to every third bit."""
    v = v & np.uint64(0x1FFFFF)
    for s, m in ((32, 0x1F00000000FFFF), (16, 0x1F0000FF0000FF),
                 (8, 0x100F00F00F00F00F), (4, 0x10C30C30C30C30C3),
                 (2, 0x1249249249249249)):
        v = (v | (v << np.uint64(s))) & np.uint64(m)
    return v


def morton_order(centres):
    lo, hi = centres.min(0), centres.max(0)
    span = max(float((hi - lo).max()), 1e-30)
    q = ((centres - lo) / span * 2097151.0).astype(np.uint64)
    code = _spread(q[:, 0])
    code |= _spread(q[:, 1]) << np.uint64(1)
    code |= _spread(q[:, 2]) << np.uint64(2)
    return np.argsort(code, kind='stable')


def _corner_bounds(tri):
    """Per-triangle (min, max) of (t, 3, n) corners; much faster than
    reducing over the 3-long axis."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    return (np.minimum(np.minimum(a, b), c), np.maximum(np.maximum(a, b), c))


def build_nodes(tri):
    """Implicit tree over ``tri`` (t, 3, 3), already in curve order:
    ((n, 2, 4) float32 nodes as the trace shader reads them, root index)."""
    lo, hi = _corner_bounds(tri)
    out = []
    offset, below, step = 0, 0, LEAF
    while True:
        n_in = len(lo)
        at = np.arange(0, n_in, step)
        count = np.minimum(n_in - at, step)
        lo = np.minimum.reduceat(lo, at, axis=0)
        hi = np.maximum.reduceat(hi, at, axis=0)
        node = np.empty((len(at), 2, 4), np.float32)
        node[:, 0, :3], node[:, 1, :3] = lo, hi
        if step == LEAF and not out:
            node[:, 0, 3], node[:, 1, 3] = np.arange(len(at)), -count
        else:
            node[:, 0, 3], node[:, 1, 3] = below + at, count
        out.append(node)
        below, offset, step = offset, offset + len(at), FAN
        if len(at) == 1:
            break
    if offset >= 1 << 24:       # node ids are stored in floats
        raise GpuError("too many source triangles")
    return np.concatenate(out), offset - 1


def mesh_triangles(obj, uv_name):
    """World-space triangles of a mesh object: dict of ``co`` (verts, 3),
    ``tri`` (t, 3) vertex ids, ``loop`` (t, 3) corner ids, ``uv`` (corners,
    2), ``mesh`` and ``matrix``."""
    me = obj.data
    me.calc_loop_triangles()
    n = len(me.loop_triangles)
    loops = meshdata.triangle_corners(me)
    verts = meshdata.corner_vertices(me)
    M = np.array(obj.matrix_world, np.float64)
    co = meshdata.positions(me).astype(np.float64) @ M[:3, :3].T + M[:3, 3]
    uv = np.empty(len(me.loops) * 2, np.float32)
    me.uv_layers[uv_name].uv.foreach_get("vector", uv)
    return {"mesh": me, "matrix": M, "co": co, "loop": loops,
            "tri": verts[loops].reshape(n, 3), "uv": uv.reshape(-1, 2),
            "flip": np.linalg.det(M[:3, :3]) < 0.0}


def corner_normals(geo):
    """(t, 3, 3) world normals the bake ray leaves along: vertex normals on
    smooth faces, the face normal on flat ones."""
    me = geo["mesh"]
    vn = np.empty(len(me.vertices) * 3, np.float32)
    me.vertex_normals.foreach_get("vector", vn)
    nrm = vn.reshape(-1, 3)[geo["tri"]]
    sharp = me.attributes.get("sharp_face")
    if sharp is not None and len(sharp.data) == len(me.polygons):
        flat = np.empty(len(me.polygons), bool)
        sharp.data.foreach_get("value", flat)
        if flat.any():
            starts = meshdata.face_starts(me)
            face = np.searchsorted(starts, geo["loop"][:, 0],
                                   side='right') - 1
            on = flat[face]
            fn = np.empty(len(me.polygons) * 3, np.float32)
            me.polygon_normals.foreach_get("vector", fn)
            nrm[on] = fn.reshape(-1, 3)[face[on]][:, None, :]
    M3 = geo["matrix"][:3, :3]
    nrm = nrm @ np.linalg.inv(M3).astype(np.float32)    # inverse transpose
    nrm /= np.maximum(np.linalg.norm(nrm, axis=2, keepdims=True), 1e-20)
    return nrm


class Tracer:
    """Builds lookups between two triangle soups (``geo`` dicts)."""

    @classmethod
    def between(cls, src_obj, src_uv, dst_obj, dst_uv, extrusion, max_angle):
        """From mesh objects (evaluated data) and their UV map names."""
        dst = mesh_triangles(dst_obj, dst_uv)
        dst["nrm"] = corner_normals(dst)
        return cls(mesh_triangles(src_obj, src_uv), dst, extrusion,
                   max_angle)

    def __init__(self, src, dst, extrusion, max_angle):
        if not len(src["tri"]) or not len(dst["tri"]):
            raise GpuError("no triangles")
        # One centred, unit-sized frame for both: float32 stays precise.
        lo = np.minimum(src["co"].min(0), dst["co"].min(0))
        hi = np.maximum(src["co"].max(0), dst["co"].max(0))
        self.centre = (lo + hi) / 2.0
        self.scale = 1.0 / max(float((hi - lo).max()), 1e-30)
        self.ext = extrusion * self.scale
        self.reach = self.ext * 2.0
        # 180 degrees (off): accept every face.
        self.cos_lim = (math.cos(max_angle)
                        if max_angle < math.pi - 1e-4 else -2.0)
        self._upload_source(src)
        self._keep_destination(dst)

    def _norm(self, co):
        return ((co - self.centre) * self.scale).astype(np.float32)

    def _upload_source(self, geo):
        tri = self._norm(geo["co"])[geo["tri"]]
        uv = geo["uv"][geo["loop"]]
        if geo["flip"]:             # mirrored object: winding flips
            tri, uv = tri[:, ::-1], uv[:, ::-1]
        order = morton_order((tri[:, 0] + tri[:, 1] + tri[:, 2]) / 3.0)
        tri, uv = tri[order], uv[order]
        nodes, self.root = build_nodes(tri)
        texels = np.zeros((len(tri), 4, 4), np.float32)
        texels[:, :3, :3] = tri
        texels[:, :3, 3] = uv[..., 0]
        texels[:, 3, :3] = uv[..., 1]
        del tri, uv
        self.tex_tris = _data_texture(texels.reshape(-1, 4))
        del texels
        self.tex_nodes = _data_texture(nodes.reshape(-1, 4))

    def _keep_destination(self, geo):
        self.dst_co = self._norm(geo["co"])
        self.dst_tri = geo["tri"]
        self.dst_nrm = geo["nrm"]
        self.dst_uv = geo["uv"][geo["loop"]].astype(np.float64)

    def lookup(self, bw, bh, margin):
        tex = _texture(bw, bh, 'RG32F')
        px = self.dst_uv * (bw, bh) - PIXEL_BIAS
        lo, hi = _corner_bounds(px)
        on = ((hi[:, 0] > 0) & (hi[:, 1] > 0) & (lo[:, 0] < bw)
              & (lo[:, 1] < bh) & np.isfinite(lo).all(1)
              & np.isfinite(hi).all(1))
        ids = np.flatnonzero(on)
        bins = _Bins(ids, lo[ids], hi[ids], bw, bh)
        raster, trace = _shader("raster"), _shader("trace")
        gbuf = {}
        for x, y, tw, th in _tiles(bw, bh, TRACE_TILE):
            g = gbuf.get((tw, th))
            if g is None:
                p = _texture(tw, th, 'RGBA32F')
                n = _texture(tw, th, 'RGBA32F')
                g = gbuf[(tw, th)] = (p, n, gpu.types.GPUFrameBuffer(
                    color_slots=(p, n)))
            self._raster(raster, g, bins.tile(x, y), px, x, y, tw, th)
            trace.bind()
            trace.uniform_sampler("g_pos", g[0])
            trace.image("g_nrm", g[1])
            trace.image("lookup", tex)
            trace.image("tris", self.tex_tris)
            trace.image("nodes", self.tex_nodes)
            trace.uniform_int("origin", (x, y))
            trace.uniform_int("size", (tw, th))
            trace.uniform_float("ext", self.ext)
            trace.uniform_float("reach", self.reach)
            trace.uniform_float("cos_lim", self.cos_lim)
            trace.uniform_int("root", self.root)
            gpu.compute.dispatch(trace, _groups(tw), _groups(th), 1)
            bins.fence.wait()
        out = Lookup(tex)
        out.grow(margin)
        return out

    def _raster(self, sh, g, ids, px, x, y, tw, th):
        fb = g[2]
        with fb.bind():
            fb.clear(color=(0.0, 0.0, 0.0, 0.0))
            if not len(ids):
                return
            fmt = gpu.types.GPUVertFormat()
            for name, n in (("px", 2), ("pos", 3), ("nrm", 3)):
                fmt.attr_add(id=name, comp_type='F32', len=n,
                             fetch_mode='FLOAT')
            vbo = gpu.types.GPUVertBuf(format=fmt, len=len(ids) * 3)
            co = self.dst_co[self.dst_tri[ids]].reshape(-1, 3)
            vbo.attr_fill("px", (px[ids] - (x, y)).reshape(-1, 2).astype(
                np.float32))
            vbo.attr_fill("pos", np.ascontiguousarray(co))
            vbo.attr_fill("nrm", np.ascontiguousarray(
                self.dst_nrm[ids].reshape(-1, 3)))
            batch = gpu.types.GPUBatch(type='TRIS', buf=vbo)
            gpu.state.blend_set('NONE')
            gpu.state.depth_test_set('NONE')
            gpu.state.depth_mask_set(False)
            gpu.state.face_culling_set('NONE')
            sh.bind()
            sh.uniform_float("tile", (tw, th))
            batch.draw(sh)


class _Bins:
    """Triangles per raster tile, in mesh order (later ones draw on top)."""

    def __init__(self, ids, lo, hi, bw, bh):
        self.fence = _Fence()
        nx = (bw + TRACE_TILE - 1) // TRACE_TILE
        ny = (bh + TRACE_TILE - 1) // TRACE_TILE
        # one texel of slack: rasterisation rounds
        tlo = np.floor((lo - 1.0) / TRACE_TILE).astype(np.int64)
        thi = np.floor((hi + 1.0) / TRACE_TILE).astype(np.int64)
        tlo = np.clip(tlo, 0, (nx - 1, ny - 1))
        thi = np.clip(thi, 0, (nx - 1, ny - 1))
        one = (tlo == thi).all(axis=1)
        key = tlo[one, 1] * nx + tlo[one, 0]
        order = np.argsort(key, kind='stable')
        self.single = ids[one][order]
        self.start = np.searchsorted(key[order], np.arange(nx * ny + 1))
        self.many, self.mlo, self.mhi = ids[~one], tlo[~one], thi[~one]
        self.nx = nx

    def tile(self, x, y):
        tx, ty = x // TRACE_TILE, y // TRACE_TILE
        k = ty * self.nx + tx
        a = self.single[self.start[k]:self.start[k + 1]]
        m = ((self.mlo[:, 0] <= tx) & (tx <= self.mhi[:, 0])
             & (self.mlo[:, 1] <= ty) & (ty <= self.mhi[:, 1]))
        return np.sort(np.concatenate((a, self.many[m])))


# -------------------------------------------------------------- self-test --

def _self_test():
    """Sample a tiny lookup and compare with the known answer, so a driver
    that runs the shaders but gets them wrong is not used."""
    rng = np.random.default_rng(7)
    px = rng.random((7, 5, 4), np.float32)
    uv = np.array([[0.5, 0.5], [0.13, 0.9], [-0.2, 1.3], [0.99, 0.01]])
    lk = Lookup.from_texels(6, 4, np.array([0, 1, 7, 23]), uv,
                            np.array([2]), np.array([1]))
    out = Sampler.sample(px, lk, 6, 4, 1).reshape(-1, 4)
    ref = np.zeros((24, 4))
    for k, (u, v) in zip((0, 1, 7, 23), uv):
        x, y = u * 5 - 0.5, v * 7 - 0.5
        x0, y0 = math.floor(x), math.floor(y)
        tx, ty = x - x0, y - y0

        def at(dx, dy):
            return px[(y0 + dy) % 7, (x0 + dx) % 5].astype(np.float64)
        ref[k] = ((at(0, 0) * (1 - tx) + at(1, 0) * tx) * (1 - ty)
                  + (at(0, 1) * (1 - tx) + at(1, 1) * tx) * ty)
    ref[2] = ref[1]
    if np.abs(out - ref).max() > 1e-4:
        raise GpuError("sampling self-test failed")


def _trace_self_test():
    """Cast a small plane onto another one, with a nearer plane facing away
    in between, and check every texel's UV, the margin and the misses."""
    quad = np.array([[0, 1, 2], [0, 2, 3]])
    corners = np.array([[0, 0], [2, 0], [2, 1], [0, 1]], np.float64)

    def plane(z, uv, flip=False):
        tri = quad[:, ::-1] if flip else quad
        co = np.column_stack((corners, np.full(4, z)))
        return co, tri, np.asarray(uv, np.float32)[tri]

    good = plane(0.0, corners * (0.4, 0.8) + 0.1)
    away = plane(0.02, corners, flip=True)
    tri = np.concatenate((good[1], away[1] + 4))
    src = {"co": np.concatenate((good[0], away[0])), "tri": tri,
           "uv": np.concatenate((good[2], away[2])).reshape(-1, 2),
           "loop": np.arange(12).reshape(4, 3), "flip": False}
    # Destination: the same plane, a bit larger, in the UV box
    # [0.25, 0.75] x [0.25, 1.0]; its right edge hangs over the source.
    dco = np.array([[0, 0, .01], [2.5, 0, .01], [2.5, 1, .01], [0, 1, .01]])
    duv = np.array([[.25, .25], [.75, .25], [.75, 1.], [.25, 1.]],
                   np.float32)
    dst = {"co": dco, "tri": quad, "uv": duv[quad].reshape(-1, 2),
           "loop": np.arange(6).reshape(2, 3), "flip": False,
           "nrm": np.tile(np.float32((0, 0, 1)), (2, 3, 1))}
    w, h = 40, 24
    got = _read(Tracer(src, dst, 0.05, math.radians(60.0)).lookup(
        w, h, 2).tex)[..., :2]
    ys, xs = np.mgrid[0:h, 0:w]
    u = (xs + 0.5 + PIXEL_BIAS[0]) / w
    v = (ys + 0.5 + PIXEL_BIAS[1]) / h
    px, py = (u - 0.25) / 0.5 * 2.5, (v - 0.25) / 0.75
    inside = (px > 0) & (px < 2.5) & (py > 0) & (py < 1)
    hit = inside & (px < 2)
    ref = np.stack((px * 0.4 + 0.1, py * 0.8 + 0.1), axis=-1)
    if np.abs(got[hit] - ref[hit]).max() > 1e-3:
        raise GpuError("ray casting self-test failed")
    if not (np.abs(got[inside & ~hit][:, 0] / MISS - 1) < 1e-3).all():
        raise GpuError("ray casting self-test failed (misses)")
    # two texels of margin left of the island copy its first column
    col = int(np.flatnonzero(inside.any(0))[0])
    rows = np.flatnonzero(inside.any(1))[1:-1]
    if (np.abs(got[rows, col - 2] - got[rows, col]).max() > 1e-6
            or not (got[rows, col - 3, 0] > 1.5e30).all()):
        raise GpuError("ray casting self-test failed (margin)")
