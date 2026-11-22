"""The editor's 2D interface, drawn over each view.

    top-left      side badge (CAGE / TARGET + object name)
    top-right     pair counter
    top-centre    prompt banner when this side still needs drawing
    bottom        hint line, then the dock (tools | toggles | help, done)
    centre        transient toast messages and the shortcut sheet

Layout is computed by pure functions of the region size so that drawing and
mouse hit-testing always agree.
"""

import time

import bpy

from ..model import pairs as P
from . import paint
from . import stage as stage_mod
from . import theme as T

TOOLS = (("tool:DRAW", "Draw"), ("tool:EDIT", "Edit"))
TOGGLES = (("toggle:xray", "X-Ray"), ("toggle:symmetry", "Mirror X"),
           ("toggle:snap_vertices", "Snap"), ("toggle:link_views", "Link"),
           ("toggle:view_world_sync", "World"), ("view:wire", "Wire"),
           ("view:texture", "Texture"))
ACTIONS = (("help", "?"), ("finish", "Done"))

SHORTCUTS = (
    ("Draw", (
        ("Drag", "freehand curve"),
        ("Click", "place points (drag one to move it)"),
        ("Enter / Right-click", "finish points"),
        ("Backspace", "remove last point"),
        ("Ctrl + Z / Ctrl + Shift + Z", "undo / redo points and preview"),
        ("Wheel / arrows / digits", "point count (preview)"),
        ("Click (preview)", "confirm the curve"),
        ("Click border edge", "curve from a mesh border"),
        ("X / Delete", "delete hovered curve"),
        ("Esc / Right-click", "cancel the stroke"),
    )),
    ("Edit", (
        ("Drag point", "slide it on the surface"),
        ("Ctrl + Click curve", "insert a point"),
        ("X / Delete", "delete point under cursor"),
        ("Click curve", "make pair active"),
    )),
    ("Pairs", (
        ("Ctrl + Wheel  /  [ ]", "fewer / more samples"),
        ("R", "reverse this side"),
        ("M", "mirror pair across X"),
        ("H  /  Alt + H", "hide pair / show all"),
        ("Shift + X", "delete active pair"),
        ("C", "toggle closed loop"),
    )),
    ("View", (
        ("Tab", "switch Draw / Edit"),
        ("F", "focus on surface under cursor"),
        ("Alt + Z", "x-ray curves"),
        ("Wire / Texture", "per-view wireframe / textures"),
        ("L", "link views"),
        ("Shift + L", "sync in world space / per mesh"),
        ("Ctrl + Z / Ctrl + Shift + Z", "undo / redo"),
        ("Esc / Enter", "close the editor"),
    )),
)


# ------------------------------------------------------------------ layout --

GUIDE = (("guide:back", "Back"),)


def dock_layout(region, extra=()):
    """[(id, label, (x0, y0, x1, y1))] for the dock in this region.
    ``extra`` is a leading group shown only in some states (template
    stepping: Back)."""
    h = paint.px(T.DOCK_H)
    pad = paint.px(T.DOCK_PAD)
    gap = paint.px(T.CHIP_GAP)
    group_gap = paint.px(T.GROUP_GAP)
    chips = []
    x = 0.0
    groups = ((tuple(extra),) if extra else ()) + (TOOLS, TOGGLES, ACTIONS)
    for gi, group in enumerate(groups):
        if gi:
            x += group_gap
        for k, (ident, label) in enumerate(group):
            if k:
                x += gap
            w = paint.text_size(label, T.TEXT_MD)[0] + paint.px(22.0)
            w = max(w, paint.px(34.0))
            chips.append((ident, label, x, w))
            x += w
    total = x + 2 * pad
    x0 = (region.width - total) / 2.0 + pad
    y0 = paint.px(T.MARGIN) + pad
    return [(ident, label, (x0 + cx, y0, x0 + cx + w, y0 + h - 2 * pad))
            for ident, label, cx, w in chips]


def dock_hit(region, xy, extra=()):
    for ident, _label, (x0, y0, x1, y1) in dock_layout(region, extra):
        if x0 <= xy[0] <= x1 and y0 <= xy[1] <= y1:
            return ident
    return None


def over_ui(region, xy, extra=()):
    """True when ``xy`` lies on the dock (clicks there aren't for the mesh)."""
    items = dock_layout(region, extra)
    if not items:
        return False
    pad = paint.px(T.DOCK_PAD)
    x0 = items[0][2][0] - pad
    x1 = items[-1][2][2] + pad
    y0 = items[0][2][1] - pad
    y1 = items[0][2][3] + pad
    return x0 <= xy[0] <= x1 and y0 <= xy[1] <= y1


# -------------------------------------------------------------------- draw --

def _is_on(mw, ident):
    """Lit state of a dock toggle (scene setting, or this view's shading)."""
    if ident.startswith("toggle:"):
        return bool(getattr(mw, ident[7:]))
    if ident == "view:texture":
        return stage_mod.shows_texture(bpy.context.space_data)
    if ident == "view:wire":
        return stage_mod.shows_wireframe(bpy.context.space_data)
    return False


def draw(ed, side, region):
    """Draw the HUD for one view.  ``ed`` is the running editor."""
    mw = ed.mw
    _badge(region, side, mw)
    _counter(region, mw)
    _prompt(ed, region, side)
    _dock(ed, region)
    _hints(ed, region)
    _toast(ed, region)
    if ed.show_help:
        _help(region)


def _badge(region, side, mw):
    m = paint.px(T.MARGIN)
    y1 = region.height - m
    h = paint.px(26.0)
    sides = ('CAGE', 'TARGET') if side == 'BOTH' else (side,)
    x = m
    for s in sides:
        obj = P.obj_for(mw, s)
        name = obj.name if obj else "—"
        label = T.SIDE_LABEL[s]
        w_label = paint.text_size(label, T.TEXT_SM)[0]
        w_name = paint.text_size(name, T.TEXT_MD)[0]
        w = paint.px(28.0) + w_label + paint.px(10.0) + w_name + paint.px(12.0)
        paint.rect(x, y1 - h, x + w, y1, T.PANEL, paint.px(T.RADIUS))
        paint.circle(x + paint.px(14.0), y1 - h / 2, paint.px(4.5),
                     T.SIDE_TINT[s])
        base = y1 - h / 2 - paint.px(4.0)
        paint.text(x + paint.px(26.0), base, label, T.TEXT_SM, T.SIDE_TINT[s])
        paint.text(x + paint.px(26.0) + w_label + paint.px(10.0), base, name,
                   T.TEXT_MD, T.TEXT)
        x += w + paint.px(8.0)


def _counter(region, mw):
    done, open_ = P.counts(mw)
    s = f"{done} matched"
    if open_:
        s += f"  ·  {open_} open"
    m = paint.px(T.MARGIN)
    w = paint.text_size(s, T.TEXT_SM)[0] + paint.px(20.0)
    h = paint.px(26.0)
    x1 = region.width - m
    y1 = region.height - m
    paint.rect(x1 - w, y1 - h, x1, y1, T.PANEL, paint.px(T.RADIUS))
    paint.text(x1 - w / 2, y1 - h / 2 - paint.px(3.5), s, T.TEXT_SM,
               T.TEXT_DIM, align='CENTER')


def _prompt(ed, region, side):
    pair = P.active_pair(ed.mw)
    if pair is None or ed.state not in ('IDLE', 'POINTS', 'STROKE'):
        return
    need = P.missing_side(pair)
    if need is None or not P.has_side(pair, P.other(need)):
        return
    color = tuple(pair.color) + (1.0,)
    m = paint.px(T.MARGIN)
    y1 = region.height - m - paint.px(36.0)
    if side == need or side == 'BOTH':
        title = f"Draw the match for {pair.name}"
        sub = ("Click once to place the point" if P.is_point_pair(pair)
               else "Drag or click along the same feature")
        w = max(paint.text_size(title, T.TEXT_LG)[0],
                paint.text_size(sub, T.TEXT_SM)[0]) + paint.px(48.0)
        h = paint.px(46.0)
        x0 = (region.width - w) / 2
        paint.rect(x0, y1 - h, x0 + w, y1, T.PANEL, paint.px(T.RADIUS))
        paint.rect(x0, y1 - h, x0 + paint.px(5.0), y1, color, paint.px(2.0))
        paint.text(x0 + paint.px(22.0), y1 - paint.px(19.0), title,
                   T.TEXT_LG, T.TEXT)
        paint.text(x0 + paint.px(22.0), y1 - paint.px(36.0), sub,
                   T.TEXT_SM, T.TEXT_DIM)
    else:
        arrow = "→" if need == 'TARGET' else "←"
        s = f"Waiting for the match on {T.SIDE_LABEL[need].lower()}  {arrow}"
        w = paint.text_size(s, T.TEXT_SM)[0] + paint.px(24.0)
        h = paint.px(24.0)
        x0 = (region.width - w) / 2
        paint.rect(x0, y1 - h, x0 + w, y1, T.PANEL, paint.px(T.RADIUS))
        paint.text(region.width / 2, y1 - h / 2 - paint.px(3.5), s,
                   T.TEXT_SM, T.TEXT_DIM, align='CENTER')


def _dock(ed, region):
    items = dock_layout(region, ed.dock_extra())
    if not items:
        return
    pad = paint.px(T.DOCK_PAD)
    x0 = items[0][2][0] - pad
    x1 = items[-1][2][2] + pad
    y0 = items[0][2][1] - pad
    y1 = items[0][2][3] + pad
    r = paint.px(T.RADIUS)
    paint.rect(x0, y0, x1, y1, T.PANEL, r + pad)
    paint.rect_outline(x0, y0, x1, y1, T.PANEL_EDGE, r + pad, 1.0)
    mw = ed.mw
    for ident, label, (a, b, c, d) in items:
        hovered = ed.hover_button == ident
        fg = T.TEXT_DIM
        bg = T.CHIP_HOVER if hovered else None
        if ident.startswith("tool:"):
            if ed.tool == ident[5:]:
                bg, fg = T.ACCENT, T.TEXT_DARK
            elif hovered:
                fg = T.TEXT
        elif ident.startswith("toggle:") or ident.startswith("view:"):
            on = _is_on(mw, ident)
            if on:
                fg = T.TEXT
            if hovered:
                fg = T.TEXT
        elif ident == "finish":
            bg = T.ON if hovered else T.CHIP
            fg = T.TEXT_DARK if hovered else T.TEXT
        elif ident == "guide:back":
            bg = T.CHIP_HOVER if hovered else T.CHIP
            fg = T.TEXT
        elif ident == "help":
            if ed.show_help:
                bg, fg = T.CHIP_HOVER, T.TEXT
        if bg is not None:
            paint.rect(a, b, c, d, bg, r)
        if _is_on(mw, ident):
            paint.rect((a + c) / 2 - paint.px(8.0), b + paint.px(3.0),
                       (a + c) / 2 + paint.px(8.0), b + paint.px(5.0),
                       T.ACCENT, paint.px(1.0))
        paint.text((a + c) / 2, (b + d) / 2 - paint.px(4.0), label,
                   T.TEXT_MD, fg, align='CENTER')


def _hints(ed, region):
    hint = ed.hint_text()
    if not hint:
        return
    y = paint.px(T.MARGIN + T.DOCK_H + 12.0)
    paint.text(region.width / 2, y, hint, T.TEXT_SM, T.TEXT_DIM,
               align='CENTER', shadow=True)


def _toast(ed, region):
    msg, t0 = ed.toast
    if not msg:
        return
    age = time.monotonic() - t0
    if age > 1.8:
        return
    alpha = 1.0 if age < 1.3 else max(0.0, 1.0 - (age - 1.3) / 0.5)
    w = paint.text_size(msg, T.TEXT_MD)[0] + paint.px(28.0)
    h = paint.px(30.0)
    x0 = (region.width - w) / 2
    y0 = paint.px(T.MARGIN + T.DOCK_H + 30.0)
    paint.rect(x0, y0, x0 + w, y0 + h,
               paint.with_alpha(T.PANEL, T.PANEL[3] * alpha),
               paint.px(T.RADIUS))
    paint.text(region.width / 2, y0 + h / 2 - paint.px(4.0), msg, T.TEXT_MD,
               paint.with_alpha(T.TEXT, alpha), align='CENTER')


def _help(region):
    paint.rect(0, 0, region.width, region.height, T.SCRIM)
    col_w = paint.px(330.0)
    row_h = paint.px(19.0)
    cols = 2 if region.width > col_w * 2.3 else 1
    groups = list(SHORTCUTS)
    per_col = [groups[i::cols] for i in range(cols)]
    heights = [sum(paint.px(30.0) + row_h * len(g[1]) for g in c)
               for c in per_col]
    h = max(heights) + paint.px(56.0)
    w = col_w * cols + paint.px(40.0)
    x0 = (region.width - w) / 2
    y1 = (region.height + h) / 2
    paint.rect(x0, y1 - h, x0 + w, y1, T.PANEL, paint.px(10.0))
    paint.text(x0 + paint.px(20.0), y1 - paint.px(28.0), "Shortcuts",
               T.TEXT_LG, T.ACCENT)
    for ci, col in enumerate(per_col):
        x = x0 + paint.px(20.0) + ci * col_w
        y = y1 - paint.px(56.0)
        for title, rows in col:
            paint.text(x, y, title.upper(), T.TEXT_SM, T.TEXT_FAINT)
            y -= paint.px(20.0)
            for key, what in rows:
                paint.text(x, y, key, T.TEXT_SM, T.TEXT)
                paint.text(x + paint.px(165.0), y, what, T.TEXT_SM,
                           T.TEXT_DIM)
                y -= row_h
            y -= paint.px(10.0)
