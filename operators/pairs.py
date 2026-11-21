"""Operators that manage the correspondence list from the sidebar."""

import bpy
import numpy as np
from mathutils import Vector

from ..model import display, legacy, sampling, snapping
from ..model import pairs as P
from ..model.properties import redraw_views


def _changed():
    display.invalidate_all()
    redraw_views()


class _PairOp:
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return P.active_pair(context.scene.mesh_wrap) is not None


class MESHWRAP_OT_pair_remove(_PairOp, bpy.types.Operator):
    """Delete the active pair"""
    bl_idname = "meshwrap.pair_remove"
    bl_label = "Delete Pair"

    def execute(self, context):
        mw = context.scene.mesh_wrap
        P.remove(mw, mw.pair_index)
        _changed()
        return {'FINISHED'}


class MESHWRAP_OT_pair_clear_side(_PairOp, bpy.types.Operator):
    """Erase one half of the active pair so it can be redrawn"""
    bl_idname = "meshwrap.pair_clear_side"
    bl_label = "Clear Side"

    side: bpy.props.EnumProperty(items=[('CAGE', "Cage", ""),
                                        ('TARGET', "Target", "")])

    def execute(self, context):
        P.points(P.active_pair(context.scene.mesh_wrap), self.side).clear()
        _changed()
        return {'FINISHED'}


class MESHWRAP_OT_pairs_clear(bpy.types.Operator):
    """Delete every pair"""
    bl_idname = "meshwrap.pairs_clear"
    bl_label = "Delete All Pairs"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return len(context.scene.mesh_wrap.pairs) > 0

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        mw = context.scene.mesh_wrap
        mw.pairs.clear()
        mw.pair_index = -1
        _changed()
        return {'FINISHED'}


class MESHWRAP_OT_pair_move(_PairOp, bpy.types.Operator):
    """Move the active pair up or down the list"""
    bl_idname = "meshwrap.pair_move"
    bl_label = "Move Pair"

    direction: bpy.props.EnumProperty(items=[('UP', "Up", ""),
                                             ('DOWN', "Down", "")])

    def execute(self, context):
        mw = context.scene.mesh_wrap
        i = mw.pair_index
        j = i - 1 if self.direction == 'UP' else i + 1
        if 0 <= j < len(mw.pairs):
            mw.pairs.move(i, j)
            mw.pair_index = j
        return {'FINISHED'}


class MESHWRAP_OT_pair_reverse(_PairOp, bpy.types.Operator):
    """Reverse the direction of one half of the active pair"""
    bl_idname = "meshwrap.pair_reverse"
    bl_label = "Reverse Direction"
    bl_description = ("Run one half of the pair the other way, for when the "
                      "matched points come out crossed")

    side: bpy.props.EnumProperty(items=[('CAGE', "Cage", ""),
                                        ('TARGET', "Target", "")],
                                 default='TARGET')

    def execute(self, context):
        pair = P.active_pair(context.scene.mesh_wrap)
        cos = P.coords(pair, self.side)
        if pair.closed and len(cos) > 2:
            cos = [cos[0]] + cos[:0:-1]     # same start, other way round
        else:
            cos = cos[::-1]
        P.set_coords(pair, self.side, cos)
        _changed()
        return {'FINISHED'}


class MESHWRAP_OT_pair_mirror(_PairOp, bpy.types.Operator):
    """Add a copy of the active pair mirrored across local X on both meshes"""
    bl_idname = "meshwrap.pair_mirror"
    bl_label = "Mirror Pair"
    bl_description = ("Duplicate the active pair, mirrored across each "
                      "object's local X axis and snapped to the surface")

    @classmethod
    def poll(cls, context):
        mw = context.scene.mesh_wrap
        return (P.active_pair(mw) is not None and mw.cage is not None
                and mw.target is not None)

    def execute(self, context):
        mw = context.scene.mesh_wrap
        made = mirror_pair(context, mw, P.active_pair(mw))
        if made is None:
            self.report({'WARNING'},
                        "That pair is already symmetric across X")
            return {'CANCELLED'}
        _changed()
        return {'FINISHED'}


def mirror_pair(context, mw, pair):
    """Append the X mirror of ``pair``; None if it mirrors onto itself."""
    dg = context.evaluated_depsgraph_get()
    mirrored = {}
    self_symmetric = True
    for side in P.SIDES:
        cos = np.array(P.coords(pair, side))
        if not len(cos):
            mirrored[side] = []
            continue
        flipped = snapping.mirror_x(cos)
        size = float(np.ptp(cos, axis=0).max()) if len(cos) > 1 else 0.0
        obj = P.obj_for(mw, side)
        size = max(size, 0.01 * max(obj.dimensions))
        if len(cos) > 1:
            gap = max(sampling.nearest_segment(cos, q, pair.closed)[2]
                      for q in flipped)
        else:
            gap = float(np.linalg.norm(flipped[0] - cos[0]))
        if gap > 0.03 * size:
            self_symmetric = False
        mirrored[side] = snapping.to_side(mw, side, dg, flipped).tolist()
    if self_symmetric:
        return None
    src = dict(name=pair.name, closed=pair.closed, samples=pair.samples,
               weight=pair.weight, lock=pair.lock)
    new = P.new_pair(mw, name=_mirror_name(src['name']))
    new.closed = src['closed']
    new.samples = src['samples']
    new.weight = src['weight']
    new.lock = src['lock']
    for side in P.SIDES:
        P.set_coords(new, side, mirrored[side])
    return new


def _mirror_name(name):
    for a, b in ((".L", ".R"), (".R", ".L"), ("_L", "_R"), ("_R", "_L"),
                 ("Left", "Right"), ("Right", "Left")):
        if name.endswith(a):
            return name[:-len(a)] + b
    return name + " (mirror)"


class MESHWRAP_OT_legacy_import(bpy.types.Operator):
    """Convert correspondences saved by an earlier version of the add-on"""
    bl_idname = "meshwrap.legacy_import"
    bl_label = "Convert Old Landmarks"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return legacy.has_legacy(context.scene.mesh_wrap)

    def execute(self, context):
        n = legacy.convert(context.scene.mesh_wrap)
        _changed()
        self.report({'INFO'}, f"Converted {n} pairs")
        return {'FINISHED'}


class MESHWRAP_OT_pair_recolor(bpy.types.Operator):
    """Give every pair a fresh, distinct colour"""
    bl_idname = "meshwrap.pair_recolor"
    bl_label = "Recolor Pairs"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        for k, pair in enumerate(context.scene.mesh_wrap.pairs):
            pair.color = P.palette_color(k)
        return {'FINISHED'}


def _view_under(screen, x, y):
    """(area, region) of the 3D view's main region under window ``x, y``."""
    for area in screen.areas:
        if area.type != 'VIEW_3D':
            continue
        for region in area.regions:
            if (region.type == 'WINDOW'
                    and region.x <= x < region.x + region.width
                    and region.y <= y < region.y + region.height):
                # the sidebar / header overlap the main region: skip them
                for other in area.regions:
                    if (other.type in {'UI', 'HEADER', 'TOOL_HEADER', 'TOOLS'}
                            and other.width > 1 and other.height > 1
                            and other.x <= x < other.x + other.width
                            and other.y <= y < other.y + other.height):
                        return None, None
                return area, region
    return None, None


def pair_under_mouse(context, area, region, xy, reach_px=12.0):
    """Index of the visible pair drawn nearest ``xy`` (region pixels)."""
    from ..editor import picking
    from .fit import result_key
    mw = context.scene.mesh_wrap
    space = area.spaces.active
    rv3d = space.region_3d
    dg = context.evaluated_depsgraph_get()
    xray = bool(getattr(space.shading, "show_xray", False))
    q = np.asarray(xy, float)
    reach = reach_px * context.preferences.system.ui_scale
    sides = []
    cage = mw.cage
    if cage is not None and cage.visible_get():
        key = result_key(cage)
        if key is None or key.value == 0.0:   # points live on the rest shape
            sides.append(('CAGE', cage))
    if mw.target is not None and mw.target.visible_get():
        sides.append(('TARGET', mw.target))
    best = None
    for side, obj in sides:
        scale = max(abs(s) for s in obj.scale) or 1.0
        size = (max(obj.dimensions) / scale) or 1.0     # local units
        for index, pair in enumerate(mw.pairs):
            if pair.hidden or not P.has_side(pair, side):
                continue
            geo = display.half(obj, dg, pair, side)
            if geo is None or not len(geo.controls):
                continue
            world = picking.to_world(obj, geo.controls)
            p, front = picking.project(region, rv3d, world)
            if not front.all():
                continue
            k, t, d = picking.segment_distance(p, q, geo.closed)
            if d > reach or (best is not None and d >= best[0]):
                continue
            if not xray:
                n = len(world)
                a = world[k]
                b = world[(k + 1) % n] if n > 1 else a
                hit = Vector(a + t * (b - a))
                if not picking.is_visible(obj, dg, region, rv3d, hit,
                                          0.01 * size):
                    continue
            best = (d, index)
    return None if best is None else best[1]


class MESHWRAP_OT_pair_pick(bpy.types.Operator):
    """Pick a pair by clicking one of its curves in the viewport"""
    bl_idname = "meshwrap.pair_pick"
    bl_label = "Pick Pair in Viewport"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        from ..editor import session
        mw = context.scene.mesh_wrap
        return (len(mw.pairs) > 0 and mw.show_in_viewport
                and not session.is_running())

    def invoke(self, context, event):
        if not any(a.type == 'VIEW_3D' for a in context.screen.areas):
            self.report({'WARNING'}, "No 3D viewport to pick from")
            return {'CANCELLED'}
        self.start = context.scene.mesh_wrap.pair_index
        context.window.cursor_modal_set('EYEDROPPER')
        context.workspace.status_text_set(
            "Pick pair: click a curve  ·  Esc / right-click cancels")
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _finish(self, context):
        context.window.cursor_modal_restore()
        context.workspace.status_text_set(None)
        redraw_views()

    def _under(self, context, event):
        area, region = _view_under(context.screen, event.mouse_x,
                                   event.mouse_y)
        if area is None:
            return None
        xy = (event.mouse_x - region.x, event.mouse_y - region.y)
        return pair_under_mouse(context, area, region, xy)

    def modal(self, context, event):
        mw = context.scene.mesh_wrap
        if event.type in {'ESC', 'RIGHTMOUSE'} and event.value == 'PRESS':
            mw.pair_index = self.start
            self._finish(context)
            return {'CANCELLED'}
        if event.type == 'MOUSEMOVE':
            # preview: the hovered pair shows as active while picking
            index = self._under(context, event)
            target = self.start if index is None else index
            if mw.pair_index != target:
                mw.pair_index = target
            return {'RUNNING_MODAL'}
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            index = self._under(context, event)
            if index is None:
                return {'RUNNING_MODAL'}     # missed: keep picking
            mw.pair_index = index
            self._finish(context)
            self.report({'INFO'}, f"Picked {mw.pairs[index].name}")
            return {'FINISHED'}
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE',
                          'TRACKPADPAN', 'TRACKPADZOOM'}:
            return {'PASS_THROUGH'}          # navigate while picking
        return {'RUNNING_MODAL'}


CLASSES = (
    MESHWRAP_OT_pair_pick,
    MESHWRAP_OT_pair_remove,
    MESHWRAP_OT_pair_clear_side,
    MESHWRAP_OT_pairs_clear,
    MESHWRAP_OT_pair_move,
    MESHWRAP_OT_pair_reverse,
    MESHWRAP_OT_pair_mirror,
    MESHWRAP_OT_legacy_import,
    MESHWRAP_OT_pair_recolor,
)
