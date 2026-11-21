"""Small immediate-mode drawing kit on top of ``gpu`` and ``blf``.

2D helpers take region pixel coordinates; 3D helpers draw in whatever space
the current GPU matrix stack is in (callers push ``matrix_world``).
"""

import math

import blf
import bpy
import gpu
from gpu_extras.batch import batch_for_shader

FONT = 0
_SHADERS = {}


def shader(name):
    sh = _SHADERS.get(name)
    if sh is None:
        try:
            sh = gpu.shader.from_builtin(name)
        except (ValueError, SystemError):
            sh = gpu.shader.from_builtin('UNIFORM_COLOR')
        _SHADERS[name] = sh
    return sh


def ui_scale():
    try:
        return bpy.context.preferences.system.ui_scale
    except AttributeError:
        return 1.0


def px(v):
    """Logical pixels -> device pixels."""
    return v * ui_scale()


def _viewport():
    vp = gpu.state.viewport_get()
    return (float(vp[2]), float(vp[3]))


# ------------------------------------------------------------------- 2D ---

def _fill(sh_coords, color, kind='TRIS', indices=None):
    sh = shader('UNIFORM_COLOR')
    batch = batch_for_shader(sh, kind, {"pos": sh_coords}, indices=indices)
    sh.bind()
    sh.uniform_float("color", color)
    batch.draw(sh)


def _rounded_outline(x0, y0, x1, y1, r, seg=5):
    r = max(0.0, min(r, (x1 - x0) / 2.0, (y1 - y0) / 2.0))
    pts = []
    for cx, cy, a0 in ((x1 - r, y1 - r, 0.0), (x0 + r, y1 - r, 90.0),
                       (x0 + r, y0 + r, 180.0), (x1 - r, y0 + r, 270.0)):
        for i in range(seg + 1):
            a = math.radians(a0 + 90.0 * i / seg)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def rect(x0, y0, x1, y1, color, radius=0.0):
    gpu.state.blend_set('ALPHA')
    if radius <= 0.0:
        _fill([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], color,
              indices=((0, 1, 2), (0, 2, 3)))
    else:
        ring = _rounded_outline(x0, y0, x1, y1, radius)
        centre = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        _fill([centre] + ring + [ring[0]], color, 'TRI_FAN')
    gpu.state.blend_set('NONE')


def rect_outline(x0, y0, x1, y1, color, radius=0.0, width=1.0):
    ring = _rounded_outline(x0, y0, x1, y1, radius) if radius > 0 else \
        [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    polyline_2d(ring + [ring[0]], color, width)


def circle(x, y, r, color, seg=20):
    gpu.state.blend_set('ALPHA')
    pts = [(x, y)] + [(x + r * math.cos(2 * math.pi * i / seg),
                       y + r * math.sin(2 * math.pi * i / seg))
                      for i in range(seg + 1)]
    _fill(pts, color, 'TRI_FAN')
    gpu.state.blend_set('NONE')


def ring(x, y, r, color, width=1.5, seg=32):
    pts = [(x + r * math.cos(2 * math.pi * i / seg),
            y + r * math.sin(2 * math.pi * i / seg)) for i in range(seg + 1)]
    polyline_2d(pts, color, width)


def polyline_2d(pts, color, width=1.0):
    if len(pts) < 2:
        return
    sh = shader('POLYLINE_UNIFORM_COLOR')
    batch = batch_for_shader(sh, 'LINE_STRIP',
                             {"pos": [(p[0], p[1], 0.0) for p in pts]})
    gpu.state.blend_set('ALPHA')
    sh.bind()
    sh.uniform_float("color", color)
    sh.uniform_float("lineWidth", width)
    sh.uniform_float("viewportSize", _viewport())
    batch.draw(sh)
    gpu.state.blend_set('NONE')


# ----------------------------------------------------------------- text ---

def text_size(text, size):
    blf.size(FONT, px(size))
    return blf.dimensions(FONT, text)


def text(x, y, s, size, color, align='LEFT', shadow=False):
    """Draw ``s`` with its baseline at y; returns the drawn width."""
    blf.size(FONT, px(size))
    w, _h = blf.dimensions(FONT, s)
    if align == 'CENTER':
        x -= w / 2.0
    elif align == 'RIGHT':
        x -= w
    if shadow:
        blf.enable(FONT, blf.SHADOW)
        blf.shadow(FONT, 3, 0.0, 0.0, 0.0, 0.7)
        blf.shadow_offset(FONT, 1, -1)
    blf.position(FONT, x, y, 0)
    blf.color(FONT, *color)
    blf.draw(FONT, s)
    if shadow:
        blf.disable(FONT, blf.SHADOW)
    return w


# ------------------------------------------------------------------- 3D ---

def lines_3d(pts, color, width=2.0, kind='LINE_STRIP'):
    if len(pts) < 2:
        return
    sh = shader('POLYLINE_UNIFORM_COLOR')
    batch = batch_for_shader(sh, kind, {"pos": [tuple(p) for p in pts]})
    sh.bind()
    sh.uniform_float("color", color)
    sh.uniform_float("lineWidth", width)
    sh.uniform_float("viewportSize", _viewport())
    batch.draw(sh)


def points_3d(pts, color, size=8.0):
    if len(pts) == 0:
        return
    sh = shader('POINT_UNIFORM_COLOR')
    batch = batch_for_shader(sh, 'POINTS', {"pos": [tuple(p) for p in pts]})
    gpu.state.point_size_set(size)
    sh.bind()
    sh.uniform_float("color", color)
    batch.draw(sh)
    gpu.state.point_size_set(1.0)


def with_alpha(color, a):
    return (color[0], color[1], color[2], a)


def mix(a, b, t):
    return tuple(a[i] * (1.0 - t) + b[i] * t for i in range(len(a)))
