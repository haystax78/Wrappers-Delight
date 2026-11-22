"""The curve editor: a modal session over the two-view stage.

Tools
    DRAW  drag for a freehand curve, click to place points, Shift+click a
          mesh border.  A finished stroke either completes the active pair
          (when that pair is still missing this side) or starts a new one.
    EDIT  slide control points along the surface, insert and delete points,
          pick the active pair.

The session keeps its own undo history (scene snapshots) and pushes a single
Blender undo step when it closes.
"""

import time
import traceback

import bpy
import numpy as np
from mathutils import Vector

from .. import preferences
from ..model import display, sampling
from ..model import pairs as P
from ..model.properties import RESULT_KEY, redraw_views
from . import guide, hud, paint, picking, scene_draw, symmetry
from . import theme as T
from . import stage as stage_mod
from .stage import Stage

UNDO_LIMIT = 64

# drawing
CURVE_MIN = 3
CURVE_MAX = 64
STROKE_SAMPLE_PX = 4.0       # screen spacing of freehand samples
STROKE_MAX_SAMPLES = 4000
DRAG_THRESHOLD_PX = 6.0      # press-and-move beyond this = freehand stroke
PICK_PX = 14.0
GUIDE_START_DELAY = 0.6      # s: let the views settle before framing
_DIAL_UP = {'WHEELUPMOUSE', 'WHEELINMOUSE', 'UP_ARROW', 'RIGHT_ARROW',
            'PAGE_UP', 'NUMPAD_PLUS', 'EQUAL', 'PLUS'}
_DIAL_DOWN = {'WHEELDOWNMOUSE', 'WHEELOUTMOUSE', 'DOWN_ARROW', 'LEFT_ARROW',
              'PAGE_DOWN', 'NUMPAD_MINUS', 'MINUS'}
_DIGITS = {'ZERO': 0, 'ONE': 1, 'TWO': 2, 'THREE': 3, 'FOUR': 4, 'FIVE': 5,
           'SIX': 6, 'SEVEN': 7, 'EIGHT': 8, 'NINE': 9}

_NAV = {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'WHEELINMOUSE',
        'WHEELOUTMOUSE', 'TRACKPADPAN', 'TRACKPADZOOM', 'MOUSEROTATE',
        'MOUSESMARTZOOM'}
_WHEEL_UP = {'WHEELUPMOUSE', 'WHEELINMOUSE'}
_WHEEL_DOWN = {'WHEELDOWNMOUSE', 'WHEELOUTMOUSE'}

_RUNNING = None          # the live operator instance
_STOP = False


def is_running():
    return _RUNNING is not None


def request_stop():
    global _STOP
    if _RUNNING is not None:
        _STOP = True


def sync_space_changed():
    """``view_world_sync`` changed: re-anchor the live editor's camera link."""
    op = _RUNNING
    if op is None or op.stage is None:
        return
    try:
        op.stage.link.set_space()
        op.stage.redraw()
    except (ReferenceError, RuntimeError, AttributeError):
        pass


def _is_navigation(event):
    t = event.type
    if t == 'NUMPAD_ENTER':
        return False
    if t in _NAV or t.startswith('NDOF_') or t.startswith('NUMPAD_'):
        return not (event.ctrl and (t in _WHEEL_UP or t in _WHEEL_DOWN))
    if event.alt and t in {'LEFTMOUSE', 'RIGHTMOUSE'}:
        return True
    if t in {'Z', 'ACCENT_GRAVE', 'HOME'} and not event.ctrl and not event.alt:
        return True             # shading pie, view pie, view all
    return False


def _digit(event):
    d = _DIGITS.get(event.type)
    if d is not None:
        return d
    if event.type.startswith('NUMPAD_') and event.type[7:].isdigit():
        return int(event.type[7:])
    return None


# ------------------------------------------------------ X symmetry (world) --

def _sym_tol(mw, obj):
    """Centre-plane tolerance for ``obj`` in world units."""
    return 0.0 if obj is None else max(obj.dimensions) * mw.symmetry_tolerance


def _plane_side(x, tol):
    """-1 / 0 / +1 side of the X=0 plane (0 within the tolerance)."""
    return 1 if x > tol else (-1 if x < -tol else 0)


def _stroke_plane_side(hits, tol):
    """Side of the first off-plane point of a stroke, 0 if none yet."""
    for h in hits:
        s = _plane_side(h.world.x, tol)
        if s:
            return s
    return 0


def _snap_hit_to_plane(obj, depsgraph, hit):
    """Move a Hit onto world X=0, back onto the surface, X forced to 0."""
    w = Vector(hit.world)
    w.x = 0.0
    inv = obj.matrix_world.inverted_safe()
    local = inv @ w
    try:
        ok, loc, _n, _f = obj.closest_point_on_mesh(local, depsgraph=depsgraph)
        if ok:
            local = loc
    except RuntimeError:
        pass
    w = obj.matrix_world @ local
    w.x = 0.0
    hit.world = w
    hit.local = inv @ w
    return hit


def _pin_centre_points(obj, hits, tol):
    """X=0 exactly on points within ``tol`` of the plane (no re-projection)."""
    inv = obj.matrix_world.inverted_safe()
    for h in hits:
        if abs(h.world.x) <= tol:
            w = Vector(h.world)
            w.x = 0.0
            h.world = w
            h.local = inv @ w


def _size_local(obj):
    scale = max(abs(s) for s in obj.matrix_world.to_scale()) or 1.0
    return max(max(obj.dimensions) / scale, 1e-6)


class MESHWRAP_OT_editor(bpy.types.Operator):
    """Draw and edit matching curves on the cage and the target"""
    bl_idname = "meshwrap.editor"
    bl_label = "Curve Editor"
    bl_description = ("Open the cage and target side by side to draw and edit "
                      "matching curves")
    bl_options = {'REGISTER'}

    tool: bpy.props.EnumProperty(
        items=[('DRAW', "Draw", ""), ('EDIT', "Edit", "")], default='DRAW')

    @classmethod
    def poll(cls, context):
        mw = context.scene.mesh_wrap
        return (mw.cage is not None and mw.target is not None
                and not is_running())

    # ============================================================ startup ==

    def invoke(self, context, event):
        global _RUNNING, _STOP
        if context.area is None or context.area.type != 'VIEW_3D':
            self.report({'ERROR'}, "Start the curve editor from a 3D Viewport")
            return {'CANCELLED'}
        if context.mode != 'OBJECT':
            try:
                bpy.ops.object.mode_set(mode='OBJECT')
            except RuntimeError:
                self.report({'ERROR'}, "Switch to Object Mode first")
                return {'CANCELLED'}
        self.scene = context.scene
        mw = self.mw

        self.state = 'IDLE'      # IDLE PRESS STROKE POINTS PREVIEW DRAG BUTTON
        self.show_help = False
        self.hover_button = None
        self.hover = None        # (side, uid, index) control point
        self.hover_curve = None  # (side, uid, segment index)
        self.border = None       # (side, closed, local points)
        self.cursor = None       # (side, picking.Hit)
        self.toast = ("", 0.0)
        self.stroke = []         # picking.Hit samples of the stroke in progress
        self.stroke_side = None
        self.kind = 'FREE'       # FREE (dragged), POINTS (clicked), BORDER
        self.stroke_closed = False
        self.pt_drag = None      # placed point being dragged (POINTS)
        self.pt_hover = None
        self.preview = []        # resampled surface points (PREVIEW)
        self.point_count = CURVE_MIN
        self.typed = ""
        self.pending_raw = None  # stroke behind the half waiting for a match
        self.guide_uids = set()  # template pairs this session steps through
        self.guide_history = []  # finished template pairs, for Back
        self._guide_start_at = None
        self.press_xy = None
        self.drag = None
        self.dirty = False
        self.undo_stack = []
        self.redo_stack = []
        self.borders = picking.BorderCache()
        self._draw_error = False

        # edit on the rest shape
        self._result_value = None
        key = _result_key(mw.cage)
        if key is not None:
            self._result_value = key.value
            key.value = 0.0
            mw.cage.data.update()

        self.stage = Stage()
        if not self.stage.open(context, mw.cage, mw.target,
                               preferences.get().editor_layout):
            self._restore_result()
            self.report({'ERROR'}, "Could not set up the editor views")
            return {'CANCELLED'}

        self._h3d = bpy.types.SpaceView3D.draw_handler_add(
            _draw_view, (self,), 'WINDOW', 'POST_VIEW')
        self._h2d = bpy.types.SpaceView3D.draw_handler_add(
            _draw_hud, (self,), 'WINDOW', 'POST_PIXEL')
        wm = context.window_manager
        self.timer = wm.event_timer_add(0.05, window=self.stage.window)
        with context.temp_override(window=self.stage.window):
            wm.modal_handler_add(self)
        _RUNNING = self
        _STOP = False
        display.invalidate_all()
        # template curves waiting for the target: step through them
        # (framed once the views have settled: local view animates them and
        # the first redraw may swap them, see Stage.tidy)
        self.guide_uids = {p.uid for p in guide.waiting(mw)}
        self._guide_start_at = (time.monotonic() + GUIDE_START_DELAY
                                if self.guide_uids else None)
        self.stage.redraw()
        redraw_views()
        return {'RUNNING_MODAL'}

    # ------------------------------------------------------------- guide --

    def guide_progress(self):
        """'k of N' for the template pairs this session is walking through."""
        mw = self.mw
        total = sum(1 for p in mw.pairs if p.uid in self.guide_uids)
        done = sum(1 for p in mw.pairs
                   if p.uid in self.guide_uids and P.is_complete(p))
        return f"{min(done + 1, total)} of {total}"

    def shown_uids(self):
        """While template curves are still waiting for their match, only
        the one being placed is shown (and can be hovered); None = all."""
        if not self.guide_uids:
            return None
        pending = self.pending_pair()
        if pending is None or pending.uid not in self.guide_uids:
            return None
        return {pending.uid}

    def guide_back(self):
        """Reopen the last template curve placed (and the mirror filled in
        with it) so it can be drawn again, and frame it."""
        mw = self.mw
        if self.state != 'IDLE':
            self.stroke = []
            self.preview = []
            self.pt_drag = self.pt_hover = None
            self.state = 'IDLE'
        while self.guide_history:
            entry = self.guide_history.pop()
            pair = P.by_uid(mw, entry['uid'])
            if pair is not None and P.has_side(pair, 'TARGET'):
                break
        else:
            self.say("No earlier template curve to go back to")
            return
        self.push_undo()
        P.set_coords(pair, 'TARGET', [])
        twin = P.by_uid(mw, entry['twin']) if entry['twin'] else None
        if twin is not None:
            P.set_coords(twin, 'TARGET', [])
        made = P.by_uid(mw, entry['made']) if entry['made'] else None
        if made is not None:
            P.remove(mw, P.index_of_uid(mw, made.uid))
        self.pending_raw = None
        mw.pair_index = P.index_of_uid(mw, pair.uid)
        try:
            guide.frame_pair(self, pair)
        except (RuntimeError, ValueError, ReferenceError):
            traceback.print_exc()
        self.changed()
        self.say(f"Back to '{pair.name}': draw it again on the target  "
                 f"({self.guide_progress()})")

    def _guide_next(self):
        """Make the next waiting template pair active and frame it."""
        if not self.guide_uids:
            return None
        pending = self.pending_pair()
        if pending is not None and P.missing_side(pending) != 'TARGET':
            return None              # the artist is mid-way through a pair
        left = [p for p in guide.waiting(self.mw)
                if p.uid in self.guide_uids]
        if not left:
            return None
        pair = pending if pending in left else left[0]
        self.mw.pair_index = P.index_of_uid(self.mw, pair.uid)
        try:
            guide.frame_pair(self, pair)
        except (RuntimeError, ValueError, ReferenceError):
            traceback.print_exc()
        return pair

    @property
    def mw(self):
        return self.scene.mesh_wrap

    # =========================================================== shutdown ==

    def _finish(self, context):
        global _RUNNING, _STOP
        if getattr(self, "_finished", False):
            return
        self._finished = True
        _RUNNING = None
        _STOP = False
        for h in ("_h3d", "_h2d"):
            handle = getattr(self, h, None)
            if handle is not None:
                bpy.types.SpaceView3D.draw_handler_remove(handle, 'WINDOW')
                setattr(self, h, None)
        if getattr(self, "timer", None) is not None:
            try:
                context.window_manager.event_timer_remove(self.timer)
            except (ReferenceError, RuntimeError):
                pass
            self.timer = None
        self.stage.close(context)
        self._restore_result()
        display.invalidate_all()
        if self.dirty:
            try:
                bpy.ops.ed.undo_push(message="Edit Correspondences")
            except RuntimeError:
                pass
        redraw_views()

    def _restore_result(self):
        if self._result_value is None:
            return
        try:
            key = _result_key(self.mw.cage)
            if key is not None:
                key.value = self._result_value
                self.mw.cage.data.update()
        except ReferenceError:
            pass

    def cancel(self, context):
        self._finish(context)

    # ============================================================== modal ==

    def modal(self, context, event):
        try:
            return self._modal(context, event)
        except Exception:  # noqa: BLE001 - never leave the stage up on a bug
            traceback.print_exc()
            self._finish(context)
            self.report({'ERROR'}, "Curve editor stopped after an error "
                                   "(see the system console)")
            return {'CANCELLED'}

    def _modal(self, context, event):
        mw = self.mw
        if (_STOP or mw.cage is None or mw.target is None
                or not self.stage.areas()):
            self._finish(context)
            return {'FINISHED'}

        if event.type == 'TIMER':
            self.stage.tidy(context, mw.cage, mw.target)
            if self._guide_start_at is not None and \
                    time.monotonic() >= self._guide_start_at:
                self._guide_start_at = None
                nxt = self._guide_next()
                if nxt is not None:
                    self.say(f"Template: draw '{nxt.name}' on the target  "
                             f"({self.guide_progress()})")
            if self.toast[0] and time.monotonic() - self.toast[1] < 2.0:
                self.stage.redraw()
            return {'PASS_THROUGH'}

        side, area, region, xy = self.stage.under_mouse(event)
        rv3d = area.spaces.active.region_3d if area else None
        view = (side, region, rv3d, Vector(xy) if xy else None)

        if area is None and self.pt_drag is not None and \
                event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            self.pt_drag = None
            self.stage.redraw()
            return {'RUNNING_MODAL'}
        if area is None and self.state not in ('STROKE', 'DRAG', 'PRESS'):
            if self.hover_button or self.cursor:
                self.hover_button = None
                self.cursor = None
                self.stage.redraw()
            return {'PASS_THROUGH'}

        if self.show_help and event.value == 'PRESS' and \
                event.type not in _NAV and event.type != 'MOUSEMOVE':
            self.show_help = False
            self.stage.redraw()
            if event.type in {'LEFTMOUSE', 'ESC', 'SLASH', 'F1'}:
                return {'RUNNING_MODAL'}

        # --- dock ---------------------------------------------------------
        if region is not None and self.state in ('IDLE', 'BUTTON'):
            extra = self.dock_extra()
            over = hud.dock_hit(region, view[3], extra)
            if over != self.hover_button:
                self.hover_button = over
                self.stage.redraw()
            if event.type == 'LEFTMOUSE' and hud.over_ui(region, view[3],
                                                          extra):
                if event.value == 'PRESS':
                    self.state = 'BUTTON'
                elif event.value == 'RELEASE' and self.state == 'BUTTON':
                    self.state = 'IDLE'
                    if over:
                        return self._button(context, over, area)
                return {'RUNNING_MODAL'}
        if self.state == 'BUTTON' and event.type == 'LEFTMOUSE' and \
                event.value == 'RELEASE':
            self.state = 'IDLE'
            return {'RUNNING_MODAL'}

        # --- navigation ---------------------------------------------------
        # In the preview the wheel and numpad +/-/digits set the point count.
        if self.tool == 'DRAW' and self.state == 'PREVIEW' and (
                event.type in _WHEEL_UP or event.type in _WHEEL_DOWN
                or event.type in {'NUMPAD_PLUS', 'NUMPAD_MINUS'}
                or _digit(event) is not None):
            self._dial_event(event)
            return {'RUNNING_MODAL'}
        if _is_navigation(event):
            # no orbiting in the middle of a stroke or a point drag
            if self.state in ('PRESS', 'STROKE') or self.pt_drag is not None:
                return {'RUNNING_MODAL'}
            self.cursor = None
            return {'PASS_THROUGH'}

        if event.type == 'MOUSEMOVE' or event.type == 'INBETWEEN_MOUSEMOVE':
            self._mouse_move(context, view, event)
            return {'RUNNING_MODAL'}

        handler = (self._draw_event if self.tool == 'DRAW'
                   else self._edit_event)
        result = handler(context, view, event)
        if result is not None:
            return result
        result = self._common_keys(context, view, event)
        if result is not None:
            return result
        # swallow everything else so stray hotkeys can't edit the scene
        if event.type in {'LEFTMOUSE', 'RIGHTMOUSE'} or event.value in {
                'PRESS', 'RELEASE', 'CLICK', 'DOUBLE_CLICK'}:
            return {'RUNNING_MODAL'}
        return {'PASS_THROUGH'}

    # ============================================================ helpers ==

    def depsgraph(self):
        return bpy.context.evaluated_depsgraph_get()

    def view_sides(self, side):
        return ('CAGE', 'TARGET') if side == 'BOTH' else (side,)

    def surface(self, view):
        """(side, Hit) of the nearest surface under the mouse."""
        side, region, rv3d, xy = view
        if side is None or xy is None:
            return None, None
        dg = self.depsgraph()
        best = (None, None, float("inf"))
        origin, _d = picking.mouse_ray(region, rv3d, xy)
        for s in self.view_sides(side):
            hit = picking.surface_hit(P.obj_for(self.mw, s), dg, region,
                                      rv3d, xy, self.mw.snap_vertices)
            if hit is not None:
                dist = (hit.world - origin).length
                if dist < best[2]:
                    best = (s, hit, dist)
        return best[0], best[1]

    def say(self, msg):
        self.toast = (msg, time.monotonic())
        self.stage.redraw()

    def changed(self, uid=None):
        self.dirty = True
        if uid is None:
            display.invalidate_all()
        else:
            display.invalidate(uid)
        self.stage.redraw()
        redraw_views()

    def push_undo(self):
        self.undo_stack.append(P.snapshot(self.mw))
        del self.undo_stack[:-UNDO_LIMIT]
        self.redo_stack.clear()

    def undo(self):
        if not self.undo_stack:
            self.say("Nothing to undo")
            return
        self.redo_stack.append(P.snapshot(self.mw))
        P.restore(self.mw, self.undo_stack.pop())
        self.pending_raw = None
        self.changed()
        self.say("Undo")

    def redo(self):
        if not self.redo_stack:
            self.say("Nothing to redo")
            return
        self.undo_stack.append(P.snapshot(self.mw))
        P.restore(self.mw, self.redo_stack.pop())
        self.pending_raw = None
        self.changed()
        self.say("Redo")

    def hint_text(self):
        if self.show_help:
            return ""
        st = self.state
        if self.tool == 'DRAW':
            if st == 'PRESS':
                return "Click to place a point, or drag to draw freehand"
            if st == 'STROKE':
                return "Drawing... release to preview"
            if st == 'POINTS':
                return (f"{len(self.stroke)} points placed  ·  click to add "
                        "more  ·  drag a point to move it  ·  Enter / RMB to "
                        "finish  ·  Backspace to undo")
            if st == 'PREVIEW':
                return (f"Points: {self.point_count}  ·  scroll wheel / "
                        "arrows / type a number  ·  LMB confirm  ·  RMB "
                        "cancel")
            pending = self.pending_pair()
            if pending is not None and pending.uid in self.guide_uids:
                back = ("  ·  Back reopens the previous one"
                        if self.guide_history else "")
                return (f"Template {self.guide_progress()}: trace "
                        f"'{pending.name}' on the target, the same feature "
                        f"as on the cage{back}")
            if pending is not None:
                return (f"Draw the matching curve for '{pending.name}' on "
                        f"the {P.missing_side(pending).lower()} mesh "
                        "(blue border)")
            if self.border is not None:
                return "Click to make a curve from this mesh border"
            return ("Click to place points, or drag to draw freehand  ·  "
                    "X deletes the hovered curve  ·  ? for all shortcuts")
        if st == 'DRAG':
            return "Release to drop the point"
        return ("Drag points to move  ·  Ctrl+click a curve to insert  ·  "
                "X deletes  ·  click a curve to select")

    # ============================================================== dock ==

    def dock_extra(self):
        """Extra dock chips for the current state."""
        return hud.GUIDE if self.guide_history else ()

    def _button(self, context, ident, area=None):
        mw = self.mw
        if ident == "guide:back":
            self.guide_back()
        elif ident.startswith("tool:"):
            self.set_tool(ident[5:])
        elif ident.startswith("toggle:"):
            attr = ident[7:]
            setattr(mw, attr, not getattr(mw, attr))
            if attr == "link_views" and mw.link_views:
                self.stage.link.start()
        elif ident == "view:wire":
            state = stage_mod.toggle_wireframe(area)
            if state is not None:
                label = T.SIDE_LABEL.get(self.stage.side_of(area), "View")
                self.say(f"{label.title()} wireframe {'on' if state else 'off'}")
        elif ident == "view:texture":
            state = stage_mod.toggle_texture(area)
            if state is not None:
                label = T.SIDE_LABEL.get(self.stage.side_of(area), "View")
                self.say(f"{label.title()} texture {'on' if state else 'off'}")
        elif ident == "help":
            self.show_help = not self.show_help
        elif ident == "finish":
            self._finish(context)
            return {'FINISHED'}
        self.stage.redraw()
        return {'RUNNING_MODAL'}

    def set_tool(self, tool):
        self.tool = tool
        self.hover = self.hover_curve = self.border = self.cursor = None
        self.stroke = []
        self.preview = []
        self.pt_drag = self.pt_hover = None
        self.drag = None
        self.state = 'IDLE'
        self.stage.redraw()

    # ======================================================== mouse move ==

    def _mouse_move(self, context, view, event):
        side, region, rv3d, xy = view
        st = self.state
        if self.tool == 'DRAW':
            if st == 'PRESS':
                if (xy - self.press_xy).length > paint.px(DRAG_THRESHOLD_PX):
                    self.state = st = 'STROKE'
            if st == 'STROKE':
                self._extend_stroke(view)
                self.cursor = self._stroke_cursor(view)
            elif st == 'POINTS':
                on_side = side in self.view_sides_of(self.stroke_side)
                if self.pt_drag is not None:
                    if on_side:
                        self._move_stroke_pt(self.pt_drag, view)
                else:
                    self.pt_hover = (self._stroke_pt_hover(view)
                                     if on_side else None)
                    self.cursor = (None if self.pt_hover is not None
                                   else self._stroke_cursor(view))
            elif st == 'IDLE':
                s, hit = self.surface(view)
                self.cursor = (s, hit) if hit is not None else None
                self._update_hover(view)
                self.hover_curve = None
                pending = self.pending_pair()
                allowed = (pending is None
                           or P.missing_side(pending) in self.view_sides(side))
                if self.hover is None and allowed:
                    self._update_border(view)
                else:
                    self.border = None
        else:
            if st == 'DRAG':
                self._drag_to(view)
            else:
                self._update_hover(view)
            self.cursor = None
        self.stage.redraw()

    # ============================================================== draw ==
    # Press and drag for a freehand stroke; press and release without moving
    # to start placing points one click at a time.  Either way the stroke
    # then goes to a preview where the number of points is chosen before it
    # is confirmed.  Hovering a mesh border and clicking turns the border
    # into a curve.  After the first half of a pair is confirmed, the other
    # mesh (blue border) takes the matching curve.

    def view_sides_of(self, stroke_side):
        return ('BOTH', stroke_side)

    def _stroke_obj(self):
        return P.obj_for(self.mw, self.stroke_side)

    def _stroke_cursor(self, view):
        """Hit on the mesh being drawn on, for the cursor ring."""
        side, region, rv3d, xy = view
        if side not in self.view_sides_of(self.stroke_side) or xy is None:
            return None
        hit = picking.surface_hit(self._stroke_obj(), self.depsgraph(),
                                  region, rv3d, xy, self.mw.snap_vertices)
        return (self.stroke_side, hit) if hit is not None else None

    def _draw_event(self, context, view, event):
        t, v = event.type, event.value
        side, region, rv3d, xy = view
        st = self.state
        mouse_btn = t in {'LEFTMOUSE', 'RIGHTMOUSE'}

        if v == 'PRESS' and t == 'ESC' and st in ('PRESS', 'STROKE', 'POINTS',
                                                  'PREVIEW'):
            self._cancel_stroke()
            return {'RUNNING_MODAL'}
        if v == 'PRESS' and st == 'POINTS' and t in {'RET', 'NUMPAD_ENTER',
                                                     'RIGHTMOUSE'}:
            self._finish_points()
            return {'RUNNING_MODAL'}
        if v == 'PRESS' and t == 'RIGHTMOUSE' and st in ('PRESS', 'STROKE',
                                                         'PREVIEW'):
            self._cancel_stroke()
            return {'RUNNING_MODAL'}
        if v == 'PRESS' and t in {'RET', 'NUMPAD_ENTER'} and st == 'PREVIEW':
            self._finish(context)
            return {'FINISHED'}

        if st == 'IDLE':
            if t == 'LEFTMOUSE' and v == 'PRESS':
                self._start_stroke(view)
                return {'RUNNING_MODAL'}
            if v == 'PRESS' and t in {'X', 'DEL'} and not (
                    event.shift or event.ctrl or event.alt):
                if self.hover is not None:
                    self._delete_hovered_pair()
                elif t == 'X':
                    self.mw.xray = not self.mw.xray
                    self.say(f"X-ray {'on' if self.mw.xray else 'off'}")
                return {'RUNNING_MODAL'}
            return None

        if st in ('PRESS', 'STROKE'):
            if t == 'LEFTMOUSE' and v == 'RELEASE':
                self._release_stroke()
            return {'RUNNING_MODAL'} if mouse_btn else None

        if st == 'POINTS':
            if self.pt_drag is not None:
                if t == 'LEFTMOUSE' and v == 'RELEASE':
                    self.pt_drag = None
                    self.pt_hover = (
                        self._stroke_pt_hover(view)
                        if side in self.view_sides_of(self.stroke_side)
                        else None)
                    self.stage.redraw()
                return {'RUNNING_MODAL'} if mouse_btn else None
            if t == 'LEFTMOUSE' and v == 'PRESS':
                self._points_click(view)
                return {'RUNNING_MODAL'}
            if t == 'BACK_SPACE' and v == 'PRESS':
                self.pt_hover = None
                if self.stroke:
                    self.stroke.pop()
                if not self.stroke:
                    self.state = 'IDLE'
                self.stage.redraw()
                return {'RUNNING_MODAL'}
            return {'RUNNING_MODAL'} if mouse_btn else None

        if st == 'PREVIEW':
            if t == 'LEFTMOUSE' and v == 'PRESS':
                self._confirm_curve()
                return {'RUNNING_MODAL'}
            if self._dial_event(event):
                return {'RUNNING_MODAL'}
            return {'RUNNING_MODAL'} if mouse_btn else None
        return None

    def _dial_event(self, event):
        """Point-count dial in PREVIEW: wheel, arrows, +/- or typed digits."""
        t, v = event.type, event.value
        if v != 'PRESS':
            return False
        if t in _DIAL_UP:
            self._change_point_count(self.point_count + 1)
            return True
        if t in _DIAL_DOWN:
            self._change_point_count(self.point_count - 1)
            return True
        d = _digit(event)
        if d is not None:
            self.typed += str(d)
            n = int(self.typed)
            if CURVE_MIN <= n <= CURVE_MAX:
                self._change_point_count(n)
            return True
        return False

    def _cancel_stroke(self):
        self.stroke = []
        self.preview = []
        self.pt_drag = self.pt_hover = None
        self.state = 'IDLE'
        self.say("Cancelled — draw a new curve")

    def _start_stroke(self, view):
        side, region, rv3d, xy = view
        mw = self.mw
        pending = self.pending_pair()
        if pending is not None and \
                P.missing_side(pending) not in self.view_sides(side):
            self.say(f"Draw on the {P.missing_side(pending).lower()} mesh "
                     "(blue border)")
            return
        if self.border is not None:
            self._start_border_curve()
            return
        s, hit = self.surface(view)
        if pending is not None:
            s = P.missing_side(pending)
            hit = picking.surface_hit(P.obj_for(mw, s), self.depsgraph(),
                                      region, rv3d, xy, mw.snap_vertices)
        elif s is None:
            s = side if side in P.SIDES else None
        if s is None:
            return
        self.push_undo()
        self.stroke = []
        self.preview = []
        self.typed = ""
        self.kind = 'FREE'
        self.stroke_closed = False
        self.stroke_side = s
        self.press_xy = xy.copy()
        if hit is not None:
            self._append_stroke_pt(hit)
        self.state = 'PRESS'
        self.stage.redraw()

    def _extend_stroke(self, view):
        side, region, rv3d, xy = view
        if side not in self.view_sides_of(self.stroke_side) or \
                len(self.stroke) >= STROKE_MAX_SAMPLES:
            return
        hit = picking.surface_hit(self._stroke_obj(), self.depsgraph(),
                                  region, rv3d, xy, self.mw.snap_vertices)
        if hit is None:
            return
        if self.stroke:
            p, front = picking.project(region, rv3d,
                                       [self.stroke[-1].world, hit.world])
            if front.all() and np.linalg.norm(p[1] - p[0]) < \
                    paint.px(STROKE_SAMPLE_PX):
                return
        self._append_stroke_pt(hit)

    def _append_stroke_pt(self, hit):
        """Add a stroke sample, skipping repeats of the last one.  With
        symmetry on, samples near X=0 snap onto the plane and samples on the
        far side of the plane from the stroke so far are rejected."""
        if self.mw.symmetry:
            obj = self._stroke_obj()
            tol = _sym_tol(self.mw, obj)
            s = _plane_side(hit.world.x, tol)
            if s == 0:
                _snap_hit_to_plane(obj, self.depsgraph(), hit)
            else:
                stroke_s = _stroke_plane_side(self.stroke, tol)
                if stroke_s and s != stroke_s:
                    self.say("Strokes can't cross the symmetry plane (X=0)")
                    return
        if self.stroke and \
                (hit.world - self.stroke[-1].world).length_squared < 1e-16:
            return
        self.stroke.append(hit)

    def _release_stroke(self):
        if self.state == 'STROKE':
            if len(self.stroke) < 2:
                self.state = 'IDLE'
                self.stroke = []
                self.say("Stroke too short — try again")
                return
            self.kind = 'FREE'
            self._enter_preview(self.mw.default_samples)
        elif self.stroke:
            self.kind = 'POINTS'
            self.state = 'POINTS'
            self.pt_hover = self.pt_drag = None
            self.stage.redraw()
        else:
            self.state = 'IDLE'
            self.say("No surface under cursor — try again")

    # --- click-placed points ---

    def _stroke_pt_hover(self, view):
        """Index of the placed point under the mouse, or None."""
        _side, region, rv3d, xy = view
        if not self.stroke or xy is None:
            return None
        p, front = picking.project(region, rv3d,
                                   [h.world for h in self.stroke])
        d = np.linalg.norm(p - np.array(xy), axis=1)
        d[~front] = np.inf
        i = int(np.argmin(d))
        return i if d[i] < paint.px(PICK_PX) else None

    def _points_click(self, view):
        side, region, rv3d, xy = view
        if side not in self.view_sides_of(self.stroke_side):
            self.say(f"Continue clicking on the {self.stroke_side.lower()} "
                     "mesh")
            return
        grab = self._stroke_pt_hover(view)
        if grab is not None:
            self.pt_drag = self.pt_hover = grab
            self.cursor = None
            self.say(f"Moving point {grab + 1} — release to drop")
            return
        if len(self.stroke) < STROKE_MAX_SAMPLES:
            hit = picking.surface_hit(self._stroke_obj(), self.depsgraph(),
                                      region, rv3d, xy, self.mw.snap_vertices)
            if hit is not None:
                self._append_stroke_pt(hit)
        self.cursor = self._stroke_cursor(view)
        self.stage.redraw()

    def _move_stroke_pt(self, idx, view):
        """Drag placed point ``idx`` to the surface under the mouse (same
        symmetry rules as placing a point)."""
        side, region, rv3d, xy = view
        if not 0 <= idx < len(self.stroke):
            return
        obj = self._stroke_obj()
        hit = picking.surface_hit(obj, self.depsgraph(), region, rv3d, xy,
                                  self.mw.snap_vertices)
        if hit is None:
            return
        if self.mw.symmetry:
            tol = _sym_tol(self.mw, obj)
            s = _plane_side(hit.world.x, tol)
            if s == 0:
                _snap_hit_to_plane(obj, self.depsgraph(), hit)
            else:
                others = self.stroke[:idx] + self.stroke[idx + 1:]
                stroke_s = _stroke_plane_side(others, tol)
                if stroke_s and s != stroke_s:
                    self.say("Points can't cross the symmetry plane (X=0)")
                    return
        self.stroke[idx] = hit

    def _finish_points(self):
        if len(self.stroke) < 2:
            self.say("Place at least 2 points before finishing "
                     "(Backspace to undo)")
            return
        self.pt_drag = self.pt_hover = None
        self._enter_preview(len(self.stroke))

    # --- mesh borders ---

    def _update_border(self, view):
        side, region, rv3d, xy = view
        self.border = None
        if side is None or xy is None:
            return
        best = None
        q = np.array(xy)
        dg = self.depsgraph()
        for s in self.view_sides(side):
            obj = P.obj_for(self.mw, s)
            for closed, loop in self.borders.get(self.mw, s, dg):
                world = picking.to_world(obj, loop)
                p, front = picking.project(region, rv3d, world)
                if not front.all():
                    continue
                k, _t, d = picking.segment_distance(p, q, closed)
                if d >= paint.px(PICK_PX) or (best is not None
                                              and d >= best[0]):
                    continue
                if not self.mw.xray and not picking.is_visible(
                        obj, dg, region, rv3d, Vector(world[k]),
                        0.01 * _size_local(obj)):
                    continue
                best = (d, s, closed, loop)
        if best is not None:
            self.border = best[1:]

    def _start_border_curve(self):
        side, closed, loop = self.border
        self.border = None
        obj = P.obj_for(self.mw, side)
        self.push_undo()
        self.typed = ""
        self.stroke_side = side
        self.kind = 'BORDER'
        self.stroke_closed = bool(closed)
        world = picking.to_world(obj, loop)
        self.stroke = [picking.Hit(Vector(l), Vector((0.0, 0.0, 1.0)), -1,
                                   Vector(w)) for l, w in zip(loop, world)]
        self._enter_preview(len(self.stroke) // 2 or self.mw.default_samples)

    # --- preview and point count ---

    def _enter_preview(self, count):
        pending = self.pending_pair()
        if pending is not None:
            count = len(P.coords(pending, P.other(self.stroke_side)))
        self.point_count = max(CURVE_MIN, min(CURVE_MAX, int(count)))
        self.typed = ""
        self._build_preview()
        self.state = 'PREVIEW'
        self.cursor = None
        self.stage.redraw()

    def _build_preview(self):
        """Resample the stroke into ``point_count`` surface points."""
        obj = self._stroke_obj()
        world = np.array([tuple(h.world) for h in self.stroke])
        pts = sampling.resample_stroke(world, self.kind, self.stroke_closed,
                                       self.point_count)
        self.preview = self._project_world(obj, pts)
        if self.mw.symmetry:
            tol = _sym_tol(self.mw, obj)
            if self.kind == 'BORDER':
                _pin_centre_points(obj, self.preview, tol)
            else:
                side = _stroke_plane_side(self.stroke, tol)
                dg = self.depsgraph()
                for h in self.preview:
                    s = _plane_side(h.world.x, tol)
                    if s == 0 or (side and s != side):
                        _snap_hit_to_plane(obj, dg, h)

    def _project_world(self, obj, world_pts):
        """World points -> Hits on the nearest surface point of ``obj``."""
        M = obj.matrix_world
        inv = M.inverted_safe()
        local = [tuple(inv @ Vector(w)) for w in world_pts]
        snapped, nrm = display.project_local(obj, self.depsgraph(), local)
        return [picking.Hit(Vector(l), Vector(n), -1, M @ Vector(l))
                for l, n in zip(snapped, nrm)]

    def _change_point_count(self, n):
        """Change the count, resampling the preview and - when this stroke
        completes a pair - the pair's first half, so both stay in step."""
        n = max(CURVE_MIN, min(CURVE_MAX, n))
        if n == self.point_count:
            return
        self.point_count = n
        self._build_preview()
        pending = self.pending_pair()
        if pending is not None:
            self._resample_existing_side(pending)
        self.stage.redraw()

    def _resample_existing_side(self, pair):
        """Resample the already-drawn half of ``pair`` to ``point_count``,
        from its original stroke when it is still known."""
        side = P.other(self.stroke_side)
        obj = P.obj_for(self.mw, side)
        raw = self.pending_raw
        if raw is not None and raw['uid'] == pair.uid and raw['side'] == side:
            world, kind, closed = raw['world'], raw['kind'], raw['closed']
        else:
            world = picking.to_world(obj, P.coords(pair, side))
            kind, closed = 'FREE', False
        if len(world) < 2:
            return
        pts = sampling.resample_stroke(np.asarray(world, float), kind, closed,
                                       self.point_count)
        hits = self._project_world(obj, pts)
        P.set_coords(pair, side, [tuple(h.local) for h in hits])
        pair.samples = self.point_count
        self.changed(pair.uid)

    def _confirm_curve(self):
        """Store the preview: complete the waiting pair, or start a new one
        whose other half must be drawn next."""
        mw = self.mw
        hits = self.preview
        self.state = 'IDLE'
        self.preview = []
        if not hits:
            self.stroke = []
            return
        side = self.stroke_side
        pending = self.pending_pair()
        guided = (pending is not None and pending.uid in self.guide_uids
                  and side == 'TARGET' and P.missing_side(pending) == side)
        if guided:
            # template curves: run the same way as the cage curve
            hits = guide.orient(self, pending, hits)
        local = [tuple(h.local) for h in hits]
        if pending is not None and P.missing_side(pending) == side:
            pair = pending
            P.set_coords(pair, side, local)
            pair.samples = len(local)
            mw.pair_index = P.index_of_uid(mw, pair.uid)
            self.pending_raw = None
            msg = (f"Matched '{pair.name}' ({len(local)} points) - draw next "
                   "curve on either mesh")
            twin = made = None
            if mw.symmetry and guided:
                twin = guide.fill_mirror_twin(self, pair)
                if twin is not None:
                    msg = (f"Matched '{pair.name}' + its mirror "
                           f"'{twin.name}'")
            if mw.symmetry and twin is None:
                made = self._mirror_completed(pair)
                if made is not None:
                    msg = (f"Matched '{pair.name}' ({len(local)} points) + "
                           f"mirrored as '{made.name}' - draw next curve")
            if guided:
                self.guide_history.append({
                    'uid': pair.uid,
                    'twin': twin.uid if twin is not None else None,
                    'made': made.uid if made is not None else None})
                nxt = self._guide_next()
                if nxt is not None:
                    msg = (f"{msg.split(' - ')[0]} - now draw '{nxt.name}' "
                           f"on the target ({self.guide_progress()})")
                else:
                    msg = (f"{msg.split(' - ')[0]} - all template curves "
                           "placed")
        else:
            pair = P.new_pair(mw)
            pair.samples = len(local)
            pair.closed = self.stroke_closed
            P.set_coords(pair, side, local)
            self.pending_raw = {
                'uid': pair.uid, 'side': side, 'kind': self.kind,
                'closed': self.stroke_closed,
                'world': [tuple(h.world) for h in self.stroke]}
            msg = (f"Draw matching curve for '{pair.name}' on the "
                   f"{P.other(side).lower()} mesh (blue border)")
        self.stroke = []
        self.changed()
        self.say(msg)

    def _mirror_completed(self, pair):
        """Mirrored twin of a just-completed pair, unless it runs along the
        centre line or mirrors onto itself."""
        from ..operators.pairs import mirror_pair
        mw = self.mw
        max_x, tol = 0.0, 0.0
        for side in P.SIDES:
            obj = P.obj_for(mw, side)
            world = picking.to_world(obj, P.coords(pair, side))
            if len(world):
                max_x = max(max_x, float(np.abs(world[:, 0]).max()))
            tol = max(tol, _sym_tol(mw, obj))
        if max_x <= tol:
            return None
        return mirror_pair(bpy.context, mw, pair)

    def _delete_hovered_pair(self):
        _side, uid, _idx = self.hover
        mw = self.mw
        index = P.index_of_uid(mw, uid)
        if index < 0:
            return
        self.push_undo()
        P.remove(mw, index)
        self.hover = None
        self.pending_raw = None
        self.changed()
        self.say("Removed curve")

    def pending_pair(self):
        """The pair still waiting for its second half, if any."""
        mw = self.mw

        def half_done(pair):
            return P.has_side(pair, 'CAGE') != P.has_side(pair, 'TARGET')

        active = P.active_pair(mw)
        if active is not None and half_done(active):
            return active
        return next((p for p in mw.pairs if half_done(p)), None)


    # ============================================================== edit ==

    def _edit_event(self, context, view, event):
        t, v = event.type, event.value
        mw = self.mw
        if t == 'LEFTMOUSE' and v == 'PRESS':
            self._update_hover(view)
            if self.hover is not None:
                side, uid, idx = self.hover
                mw.pair_index = P.index_of_uid(mw, uid)
                self.push_undo()
                self.drag = (side, uid, idx, self._mirror_partner(
                    side, uid, idx))
                self.state = 'DRAG'
            elif self.hover_curve is not None:
                side, uid, seg = self.hover_curve
                mw.pair_index = P.index_of_uid(mw, uid)
                if event.ctrl:
                    self._insert_point(view, side, uid, seg)
            else:
                mw.pair_index = -1
            self.stage.redraw()
            return {'RUNNING_MODAL'}
        if t == 'LEFTMOUSE' and v == 'RELEASE' and self.state == 'DRAG':
            self.state = 'IDLE'
            uid = self.drag[1]
            self.drag = None
            self.changed(uid)
            return {'RUNNING_MODAL'}
        if t in {'X', 'DEL'} and v == 'PRESS' and not event.shift and \
                self.state == 'IDLE':
            self._update_hover(view)
            if self.hover is not None:
                self._delete_point(*self.hover)
            return {'RUNNING_MODAL'}
        if t == 'ESC' and v == 'PRESS' and self.state == 'DRAG':
            self.state = 'IDLE'
            self.drag = None
            self.undo()
            return {'RUNNING_MODAL'}
        return None

    def _visible_halves(self, side):
        dg = self.depsgraph()
        only = self.shown_uids()
        for s in self.view_sides(side):
            obj = P.obj_for(self.mw, s)
            for pair in self.mw.pairs:
                if pair.hidden or not P.has_side(pair, s):
                    continue
                if only is not None and pair.uid not in only:
                    continue
                geo = display.half(obj, dg, pair, s)
                if geo is not None:
                    yield s, obj, pair, geo

    def _update_hover(self, view):
        side, region, rv3d, xy = view
        self.hover = self.hover_curve = None
        if side is None or xy is None:
            return
        q = np.array(xy)
        reach = paint.px(T.PICK_PX)
        best_pt = best_curve = None
        dg = self.depsgraph()
        for s, obj, pair, geo in self._visible_halves(side):
            p, front = picking.project(region, rv3d,
                                       picking.to_world(obj, geo.controls))
            d = np.linalg.norm(p - q, axis=1)
            d[~front] = np.inf
            for i in np.argsort(d)[:3]:
                if d[i] > reach:
                    break
                if best_pt is not None and d[i] >= best_pt[0]:
                    break
                if self._seen(obj, dg, region, rv3d, geo.controls[i]):
                    best_pt = (d[i], s, pair.uid, int(i))
                    break
            if len(geo.controls) > 1:
                pc = p
                k, _t, dc = picking.segment_distance(pc, q, geo.closed)
                if dc <= reach and (best_curve is None or dc < best_curve[0]):
                    best_curve = (dc, s, pair.uid, k)
        if best_pt is not None:
            self.hover = best_pt[1:]
        elif best_curve is not None:
            self.hover_curve = best_curve[1:]

    def _seen(self, obj, dg, region, rv3d, local):
        if self.mw.xray:
            return True
        world = obj.matrix_world @ Vector(local)
        return picking.is_visible(obj, dg, region, rv3d, world,
                                  0.01 * _size_local(obj))

    def _drag_to(self, view):
        side, uid, idx, partner = self.drag
        _s, region, rv3d, xy = view
        if region is None or xy is None:
            return
        mw = self.mw
        obj = P.obj_for(mw, side)
        hit = picking.surface_hit(obj, self.depsgraph(), region, rv3d, xy,
                                  mw.snap_vertices)
        if hit is None:
            return
        pair = P.by_uid(mw, uid)
        if pair is None:
            return
        pts = P.points(pair, side)
        if not 0 <= idx < len(pts):
            return
        co = hit.local.copy()
        if partner == 'CENTRE':
            co.x = 0.0
        pts[idx].co = co
        if partner not in (None, 'CENTRE'):
            p_uid, p_idx = partner
            p_pair = P.by_uid(mw, p_uid)
            if p_pair is not None:
                mirrored = [(-co.x, co.y, co.z)]
                snapped, _n = display.project_local(obj, self.depsgraph(),
                                                    mirrored)
                P.points(p_pair, side)[p_idx].co = snapped[0]
                display.invalidate(p_uid)
        display.invalidate(uid)
        self.dirty = True

    def _mirror_partner(self, side, uid, idx):
        """The point that mirrors the dragged one across local X, if any."""
        mw = self.mw
        if not mw.symmetry:
            return None
        obj = P.obj_for(mw, side)
        pair = P.by_uid(mw, uid)
        co = np.array(P.points(pair, side)[idx].co)
        tol = 0.015 * _size_local(obj)
        if abs(co[0]) <= tol * 0.5:
            return 'CENTRE'
        want = co * np.array([-1.0, 1.0, 1.0])
        best = None
        for other in mw.pairs:
            for j, p in enumerate(P.points(other, side)):
                if other.uid == uid and j == idx:
                    continue
                d = float(np.linalg.norm(np.array(p.co) - want))
                if d <= tol and (best is None or d < best[0]):
                    best = (d, other.uid, j)
        return None if best is None else (best[1], best[2])

    def _insert_point(self, view, side, uid, seg):
        obj = P.obj_for(self.mw, side)
        _s, region, rv3d, xy = view
        hit = picking.surface_hit(obj, self.depsgraph(), region, rv3d, xy)
        pair = P.by_uid(self.mw, uid)
        if hit is None or pair is None:
            return
        self.push_undo()
        cos = P.coords(pair, side)
        cos.insert(seg + 1, tuple(hit.local))
        P.set_coords(pair, side, cos)
        self.changed(uid)
        self.say("Point inserted")

    def _delete_point(self, side, uid, idx):
        mw = self.mw
        pair = P.by_uid(mw, uid)
        if pair is None:
            return
        self.push_undo()
        cos = P.coords(pair, side)
        del cos[idx]
        if len(cos) < 2 and not P.is_point_pair(pair):
            cos = []
        P.set_coords(pair, side, cos)
        if not pair.cage_points and not pair.target_points:
            P.remove(mw, P.index_of_uid(mw, uid))
            self.say("Pair deleted")
        elif not cos:
            self.say(f"{T.SIDE_LABEL[side].title()} half cleared")
        self.hover = None
        self.changed(uid)

    # ======================================================= shared keys ==

    def _common_keys(self, context, view, event):
        t, v = event.type, event.value
        mw = self.mw
        pair = P.active_pair(mw)

        if event.ctrl and (t in _WHEEL_UP or t in _WHEEL_DOWN):
            self._change_samples(1 if t in _WHEEL_UP else -1)
            return {'RUNNING_MODAL'}
        if v != 'PRESS':
            return None
        idle = self.state == 'IDLE'

        if t in {'ESC', 'RET', 'NUMPAD_ENTER'} and idle:
            self._finish(context)
            return {'FINISHED'}
        if t == 'TAB' and not event.ctrl:
            self.set_tool('EDIT' if self.tool == 'DRAW' else 'DRAW')
            return {'RUNNING_MODAL'}
        if t == 'Z' and event.ctrl:
            if idle:
                self.redo() if event.shift else self.undo()
            return {'RUNNING_MODAL'}
        if t == 'Y' and event.ctrl:
            if idle:
                self.redo()
            return {'RUNNING_MODAL'}
        if t == 'Z' and event.alt:
            mw.xray = not mw.xray
            self.say(f"X-ray {'on' if mw.xray else 'off'}")
            return {'RUNNING_MODAL'}
        if t == 'F1' or (t == 'SLASH' and event.shift):
            self.show_help = not self.show_help
            self.stage.redraw()
            return {'RUNNING_MODAL'}
        if t in {'LEFT_BRACKET', 'RIGHT_BRACKET'}:
            self._change_samples(1 if t == 'RIGHT_BRACKET' else -1)
            return {'RUNNING_MODAL'}
        if t == 'L' and event.shift and not event.ctrl:
            mw.view_world_sync = not mw.view_world_sync
            self.say("View sync: " + ("world space" if mw.view_world_sync
                                      else "per mesh"))
            return {'RUNNING_MODAL'}
        if t == 'L' and not event.ctrl:
            mw.link_views = not mw.link_views
            if mw.link_views:
                self.stage.link.start()
            self.say(f"Views {'linked' if mw.link_views else 'unlinked'}")
            return {'RUNNING_MODAL'}
        if t == 'F' and not (event.ctrl or event.alt or event.shift
                             or event.oskey):
            if self.state not in ('PRESS', 'STROKE', 'DRAG'):
                self._focus_under_mouse(view)
            return {'RUNNING_MODAL'}
        if not idle:
            return None

        if t == 'H':
            if event.alt:
                self.push_undo()
                for p in mw.pairs:
                    p.hidden = False
                self.changed()
                self.say("All pairs shown")
            elif pair is not None:
                self.push_undo()
                pair.hidden = True
                self.changed(pair.uid)
                self.say(f"{pair.name} hidden")
            return {'RUNNING_MODAL'}
        if pair is None:
            return None
        if t == 'X' and event.shift:
            self.push_undo()
            name = pair.name
            P.remove(mw, mw.pair_index)
            self.changed()
            self.say(f"{name} deleted")
            return {'RUNNING_MODAL'}
        if t == 'R' and not event.ctrl:
            side = view[0] if view[0] in P.SIDES else 'TARGET'
            cos = P.coords(pair, side)
            if len(cos) > 1:
                self.push_undo()
                cos = ([cos[0]] + cos[:0:-1]) if pair.closed else cos[::-1]
                P.set_coords(pair, side, cos)
                self.changed(pair.uid)
                self.say(f"{pair.name}: {side.lower()} side reversed")
            return {'RUNNING_MODAL'}
        if t == 'C' and not event.ctrl:
            if not P.is_point_pair(pair):
                self.push_undo()
                pair.closed = not pair.closed
                self.changed(pair.uid)
                self.say(f"{pair.name} {'closed' if pair.closed else 'opened'}")
            return {'RUNNING_MODAL'}
        if t == 'M' and not event.ctrl:
            from ..operators.pairs import mirror_pair
            self.push_undo()
            made = mirror_pair(context, mw, pair)
            if made is None:
                self.undo_stack.pop()
                self.say("Already symmetric across X")
            else:
                self.changed()
                self.say(f"Mirrored to {made.name}")
            return {'RUNNING_MODAL'}
        return None

    def _change_samples(self, delta):
        pair = P.active_pair(self.mw)
        if pair is None or P.is_point_pair(pair):
            return
        pair.samples = max(2, min(256, pair.samples + delta))
        self.dirty = True
        display.invalidate(pair.uid)
        self.say(f"{pair.name}: {pair.samples} samples")

    def _focus_under_mouse(self, view):
        """Orbit the hovered view around the surface point under the mouse,
        keeping the zoom.  When the views are linked, the other view moves
        to the matching point on its own mesh."""
        side, region, rv3d, _xy = view
        if rv3d is None:
            return
        _s, hit = self.surface(view)
        if hit is None:
            self.say("Focus: no surface under the cursor")
            return
        rv3d.view_location = hit.world
        self.cursor = None
        if side in P.SIDES and self.mw.link_views:
            self.stage.link.follow(side)
        self.stage.redraw()
        self.say("Focused on surface")


def _result_key(cage):
    keys = getattr(getattr(cage, "data", None), "shape_keys", None)
    return keys.key_blocks.get(RESULT_KEY) if keys else None


# =================================================================== draw ==

def _draw_view(op):
    """POST_VIEW: the X symmetry line and the surface-following cursor."""
    try:
        area = bpy.context.area
        side = op.stage.side_of(area)
        if side is None:
            return
        if op.mw.symmetry:
            dg = bpy.context.evaluated_depsgraph_get()
            for s in op.view_sides(side):
                obj = P.obj_for(op.mw, s)
                if obj is not None:
                    symmetry.draw(obj, obj.evaluated_get(dg).data)
        if op.tool != 'DRAW' or op.cursor is None or \
                op.cursor[0] not in op.view_sides(side):
            return
        rv3d = bpy.context.region_data
        if rv3d is None:
            return
        s, hit = op.cursor
        obj = P.obj_for(op.mw, s)
        normal = (obj.matrix_world.to_3x3() @ Vector(hit.normal)).normalized()
        _ring_3d(hit.world, normal, rv3d.view_distance * 0.016)
    except Exception:  # noqa: BLE001 - report once, keep drawing
        if not op._draw_error:
            op._draw_error = True
            traceback.print_exc()


def _ring_3d(centre, normal, radius):
    import gpu
    n = Vector(normal)
    if n.length < 1e-9:
        n = Vector((0.0, 0.0, 1.0))
    a = n.orthogonal().normalized()
    b = n.cross(a)
    c = Vector(centre)
    pts = [tuple(c + (a * np.cos(t) + b * np.sin(t)) * radius)
           for t in np.linspace(0.0, 2.0 * np.pi, 33)]
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('NONE')
    paint.lines_3d(pts, (0.0, 0.0, 0.0, 0.5), 3.6)
    paint.lines_3d(pts, (1.0, 1.0, 1.0, 0.9), 1.6)
    gpu.state.blend_set('NONE')


def _draw_curves(op, side, region, rv3d):
    """Stored pairs, then whatever is being drawn right now."""
    mw = op.mw
    active = P.active_pair(mw)
    hover = None
    if op.hover is not None:
        hover = (op.hover[1], op.hover[2])
    opts = scene_draw.DrawOptions(
        controls=op.tool == 'EDIT', xray=mw.xray, hover=hover,
        dim_others=op.tool == 'EDIT',
        active_uid=active.uid if active else None,
        focus=op.shown_uids())
    for s in op.view_sides(side):
        scene_draw.draw_side(mw, s, P.obj_for(mw, s), region, rv3d, opts)
    if opts.focus:
        # placing a template curve: show which way it runs
        pair = P.by_uid(mw, next(iter(opts.focus)))
        if pair is not None and 'CAGE' in op.view_sides(side):
            scene_draw.draw_arrows(mw, 'CAGE', mw.cage, region, rv3d,
                                   pair.uid, tuple(pair.color))
    if op.tool == 'DRAW':
        _draw_live(op, side, region, rv3d)


def _xy(region, rv3d, world_pts):
    p, front = picking.project(region, rv3d, world_pts)
    return [tuple(q) for q, f in zip(p, front) if f]


def _draw_live(op, side, region, rv3d):
    st = op.state
    here = op.stroke_side is not None and op.stroke_side in op.view_sides(side)
    line = paint.with_alpha(T.STROKE, 1.0)
    out = scene_draw.OUTLINE

    if st in ('PRESS', 'STROKE') and here and op.stroke:
        pts = _xy(region, rv3d, [h.world for h in op.stroke])
        if len(pts) >= 2:
            pts = scene_draw.smooth_polyline(pts)
            paint.polyline_2d(pts, out, paint.px(5.0))
            paint.polyline_2d(pts, line, paint.px(2.5))

    if st == 'POINTS' and here and op.stroke:
        world = [tuple(h.world) for h in op.stroke]
        dense = scene_draw.dense_spline(world) if len(world) >= 3 else world
        pts = _xy(region, rv3d, dense)
        if len(pts) >= 2:
            paint.polyline_2d(pts, out, paint.px(4.5))
            paint.polyline_2d(pts, line, paint.px(2.0))
        if op.cursor is not None and op.pt_drag is None:
            tent = world + [tuple(op.cursor[1].world)]
            dense = scene_draw.dense_spline(tent) if len(tent) >= 3 else tent
            pts = _xy(region, rv3d, dense)
            if len(pts) >= 2:
                paint.polyline_2d(pts, paint.with_alpha(line, 0.5),
                                  paint.px(1.5))
        hot = op.pt_drag if op.pt_drag is not None else op.pt_hover
        p, front = picking.project(region, rv3d, world)
        for i, (q, f) in enumerate(zip(p, front)):
            if not f:
                continue
            if i == hot:
                scene_draw.ring_marker(q[0], q[1], paint.px(7.5), T.ACCENT,
                                       paint.px(2.0),
                                       fill=op.pt_drag is not None)
            else:
                scene_draw.ring_marker(q[0], q[1], paint.px(5.0),
                                       (1.0, 1.0, 1.0, 1.0), paint.px(1.4),
                                       fill=True)

    if st == 'PREVIEW' and here and op.preview:
        pts = _xy(region, rv3d, [h.world for h in op.preview])
        if len(pts) >= 2:
            pts = scene_draw.smooth_polyline(pts, closed=op.stroke_closed)
            paint.polyline_2d(pts, out, paint.px(5.0))
            paint.polyline_2d(pts, line, paint.px(2.5))
        for q in _xy(region, rv3d, [h.world for h in op.preview]):
            scene_draw.ring_marker(q[0], q[1], paint.px(6.0),
                                   (1.0, 1.0, 1.0, 1.0), paint.px(1.6))

    if st == 'IDLE' and op.border is not None and \
            op.border[0] in op.view_sides(side):
        s, closed, loop = op.border
        pts = _xy(region, rv3d, picking.to_world(P.obj_for(op.mw, s), loop))
        if closed and pts:
            pts.append(pts[0])
        if len(pts) >= 2:
            paint.polyline_2d(pts, out, paint.px(5.5))
            paint.polyline_2d(pts, T.BORDER_HINT, paint.px(3.0))

    # blue border: the view a curve is being drawn in, from the first press
    # until the curve is accepted (or cancelled); when idle, the view that
    # takes the matching curve
    pending = op.pending_pair()
    drawing = st in ('PRESS', 'STROKE', 'POINTS', 'PREVIEW') and here
    waiting = (st == 'IDLE' and pending is not None
               and P.missing_side(pending) in op.view_sides(side))
    if drawing or waiting:
        w = paint.px(5.0)
        paint.rect_outline(w / 2, w / 2, region.width - w / 2,
                           region.height - w / 2, T.ACTIVE_SIDE, width=w)


def _draw_hud(op):
    try:
        area = bpy.context.area
        side = op.stage.side_of(area)
        if side is None:
            return
        region = bpy.context.region
        _draw_curves(op, side, region, bpy.context.region_data)
        hud.draw(op, side, region)
    except Exception:  # noqa: BLE001
        if not op._draw_error:
            op._draw_error = True
            traceback.print_exc()


CLASSES = (MESHWRAP_OT_editor,)
