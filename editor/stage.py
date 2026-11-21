"""The editor's two-view stage: cage on the left, target on the right.

The stage is built either in a new window (the artist's layout is never
touched) or by splitting the 3D view the editor was started from.  Each half
isolates its object in local view, hides Blender's own viewport chrome, and
the two cameras can be linked so orbiting or zooming one moves the other.
"""

import bpy
from mathutils import Quaternion, Vector

_CHROME = ("show_region_header", "show_region_toolbar", "show_region_ui",
           "show_region_tool_header", "show_gizmo")
_FRONT = Quaternion((0.70710678, 0.70710678, 0.0, 0.0))


def _window_region(area):
    for region in area.regions:
        if region.type == 'WINDOW':
            return region
    return None


def _toggle_local_view(window, area, frame=False):
    region = _window_region(area)
    if region is None:
        return
    try:
        with bpy.context.temp_override(window=window, area=area,
                                       region=region):
            bpy.ops.view3d.localview(frame_selected=frame)
    except RuntimeError:
        pass


_SCREEN_TAG = "meshwrap_stage"


def release_local_views(context):
    """Leave local views left behind in a stage window by a crashed session
    (Blender only has a handful of local-view slots)."""
    for window in context.window_manager.windows:
        if window.screen.get(_SCREEN_TAG) is None:
            continue
        for area in window.screen.areas:
            if area.type == 'VIEW_3D' and \
                    area.spaces.active.local_view is not None:
                _toggle_local_view(window, area)


def bounds(*objs):
    """World-space (lo, hi) bounding box around ``objs``."""
    corners = [o.matrix_world @ Vector(c) for o in objs if o is not None
               for c in o.bound_box]
    if not corners:
        return Vector((-1, -1, -1)), Vector((1, 1, 1))
    lo = Vector([min(c[i] for c in corners) for i in range(3)])
    hi = Vector([max(c[i] for c in corners) for i in range(3)])
    return lo, hi


def _frame(rv3d, *objs):
    """Centre the view on the objects' bounding box at a comfortable distance."""
    lo, hi = bounds(*objs)
    rv3d.view_location = (lo + hi) / 2.0
    rv3d.view_distance = max((hi - lo).length * 1.25, 1e-3)


def world_sync():
    mw = getattr(bpy.context.scene, "mesh_wrap", None)
    return bool(mw is None or mw.view_world_sync)


def _hide_chrome(space):
    saved = {a: getattr(space, a) for a in _CHROME}
    saved["show_text"] = space.overlay.show_text
    for a in _CHROME:
        setattr(space, a, False)
    space.overlay.show_text = False
    return saved


def _restore_chrome(space, saved):
    if not saved:
        return
    for a in _CHROME:
        setattr(space, a, saved[a])
    space.overlay.show_text = saved["show_text"]


class Stage:
    def __init__(self):
        self.window = None
        self.sides = {}             # area pointer -> 'CAGE' | 'TARGET'
        self.mode = None            # 'WINDOW' | 'SPLIT' | 'SINGLE'
        self._source_ptr = 0        # area the editor was started from
        self._added_ptr = 0         # area created by a split
        self._saved_chrome = None
        self._saved_view = None
        self._saved_selection = ([], None)
        self._ordered = False
        self.link = CameraLink(self)

    # --------------------------------------------------------- queries --

    def areas(self):
        if self.window is None:
            return []
        try:
            return [a for a in self.window.screen.areas
                    if a.as_pointer() in self.sides]
        except ReferenceError:
            return []

    def view(self, side):
        """(area, region, rv3d) showing ``side``, or (None,)*3."""
        for area in self.areas():
            if self.sides[area.as_pointer()] == side:
                region = _window_region(area)
                return area, region, area.spaces.active.region_3d
        return None, None, None

    def side_of(self, area):
        return self.sides.get(area.as_pointer()) if area else None

    def under_mouse(self, event):
        """(side, area, region, (x, y) in region) at the event position."""
        mx, my = event.mouse_x, event.mouse_y
        for area in self.areas():
            region = _window_region(area)
            if region is None:
                continue
            if (region.x <= mx < region.x + region.width
                    and region.y <= my < region.y + region.height):
                return (self.sides[area.as_pointer()], area, region,
                        (mx - region.x, my - region.y))
        return None, None, None, None

    def redraw(self):
        for area in self.areas():
            area.tag_redraw()

    # ------------------------------------------------------------ open --

    def open(self, context, cage, target, mode):
        vl = context.view_layer
        self._saved_selection = ([o for o in vl.objects if o.select_get()],
                                 vl.objects.active)
        release_local_views(context)
        ok = False
        if mode == 'WINDOW':
            ok = self._open_window(context)
        if not ok:
            ok = self._open_split(context)
        if not ok:
            return False
        try:
            self._isolate(context, cage, target)
        except (RuntimeError, ReferenceError):
            self.close(context)
            return False
        self.link.start()
        return True

    def _open_window(self, context):
        wm = context.window_manager
        before = {w.as_pointer() for w in wm.windows}
        try:
            bpy.ops.wm.window_new()
        except RuntimeError:
            return False
        win = next((w for w in wm.windows
                    if w.as_pointer() not in before), None)
        if win is None:
            return False
        area = max((a for a in win.screen.areas if a.type == 'VIEW_3D'),
                   key=lambda a: a.width * a.height, default=None)
        if area is None:
            area = max(win.screen.areas, key=lambda a: a.width * a.height)
            area.type = 'VIEW_3D'
        # one 3D view filling the window
        for other in list(win.screen.areas):
            if other != area:
                try:
                    with context.temp_override(window=win, area=other):
                        bpy.ops.screen.area_close()
                except RuntimeError:
                    pass
        self.window = win
        self.mode = 'WINDOW'
        win.screen[_SCREEN_TAG] = True
        return self._split(context, win, area)

    def _open_split(self, context):
        win, area = context.window, context.area
        if area is None or area.type != 'VIEW_3D':
            return False
        space = area.spaces.active
        rv3d = space.region_3d
        self._saved_view = (rv3d.view_location.copy(),
                            rv3d.view_rotation.copy(), rv3d.view_distance,
                            rv3d.view_perspective)
        self._saved_chrome = _hide_chrome(space)
        if space.local_view is not None:
            _toggle_local_view(win, area)
        self.window = win
        self.mode = 'SPLIT'
        self._source_ptr = area.as_pointer()
        if area.width < 400 or not self._split(context, win, area):
            # too narrow to split: edit both meshes in one view
            self.mode = 'SINGLE'
            self.sides = {area.as_pointer(): 'BOTH'}
            return True
        return True

    def _split(self, context, win, area):
        region = _window_region(area)
        before = {a.as_pointer() for a in win.screen.areas}
        try:
            with context.temp_override(window=win, area=area, region=region):
                bpy.ops.screen.area_split(direction='VERTICAL', factor=0.5)
        except RuntimeError:
            return False
        added = next((a for a in win.screen.areas
                      if a.as_pointer() not in before), None)
        if added is None:
            return False
        self._added_ptr = added.as_pointer()
        left, right = sorted((area, added), key=lambda a: a.x)
        self.sides = {left.as_pointer(): 'CAGE', right.as_pointer(): 'TARGET'}
        return True

    def _isolate(self, context, cage, target):
        vl = context.view_layer
        for area in self.areas():
            side = self.sides[area.as_pointer()]
            space = area.spaces.active
            if self.mode != 'SPLIT' or area.as_pointer() != self._source_ptr:
                _hide_chrome(space)
            if side == 'BOTH':
                continue
            obj = cage if side == 'CAGE' else target
            for o in vl.objects:
                if o.select_get():
                    o.select_set(False)
            obj.hide_set(False)
            obj.select_set(True)
            vl.objects.active = obj
            if space.local_view is not None:
                _toggle_local_view(self.window, area)
            _toggle_local_view(self.window, area)
            rv3d = space.region_3d
            rv3d.view_rotation = _FRONT
            rv3d.view_perspective = 'PERSP'
            if world_sync():
                _frame(rv3d, cage, target)
            else:
                _frame(rv3d, obj)
        for o in vl.objects:
            if o.select_get():
                o.select_set(False)

    def tidy(self, context, cage, target):
        """Per-tick upkeep once the window has laid itself out.

        Area positions are only final after the first redraw, so the cage /
        target order is checked here and the views are swapped if the split
        put the cage on the right.  Blender also re-enables some viewport
        chrome (tool header, gizmos) after local view; hide it again.
        """
        for area in self.areas():
            space = area.spaces.active
            for a in _CHROME:
                if getattr(space, a):
                    setattr(space, a, False)
            if space.overlay.show_text:
                space.overlay.show_text = False
        if self._ordered or len(self.sides) != 2:
            return
        cage_area, _r, _v = self.view('CAGE')
        target_area, _r, _v = self.view('TARGET')
        if cage_area is None or target_area is None:
            return
        self._ordered = True
        if cage_area.x > target_area.x:
            self.sides = {cage_area.as_pointer(): 'TARGET',
                          target_area.as_pointer(): 'CAGE'}
            self._isolate(context, cage, target)
            self.link.start()
            self.redraw()

    # ----------------------------------------------------------- close --

    def close(self, context):
        self.link.stop()
        win = self.window
        if win is None:
            return
        try:
            areas = self.areas()
            for area in areas:
                if area.spaces.active.local_view is not None:
                    _toggle_local_view(win, area)
            if self.mode == 'WINDOW':
                ptr = win.as_pointer()
                bpy.app.timers.register(lambda: _close_window(ptr),
                                        first_interval=0.0)
            else:
                self._close_split(context, win)
        except ReferenceError:
            pass
        self.window = None
        self.sides = {}
        sel, act = self._saved_selection
        vl = context.view_layer
        try:
            for o in vl.objects:
                o.select_set(o in sel)
            if act is not None and act.name in vl.objects:
                vl.objects.active = act
        except (ReferenceError, RuntimeError):
            pass

    def _close_split(self, context, win):
        source = added = None
        for area in win.screen.areas:
            if area.as_pointer() == self._source_ptr:
                source = area
            elif area.as_pointer() == self._added_ptr:
                added = area
        if added is not None:
            try:
                with context.temp_override(window=win, area=added):
                    bpy.ops.screen.area_close()
            except RuntimeError:
                pass
        if source is not None:
            space = source.spaces.active
            # region visibility updates need a window in context, or
            # Blender dereferences a null workspace
            with context.temp_override(window=win, screen=win.screen,
                                       area=source):
                _restore_chrome(space, self._saved_chrome)
            if self._saved_view is not None:
                rv3d = space.region_3d
                (rv3d.view_location, rv3d.view_rotation, rv3d.view_distance,
                 rv3d.view_perspective) = self._saved_view
            source.tag_redraw()


def _close_window(ptr):
    wm = bpy.context.window_manager
    for win in wm.windows:
        if win.as_pointer() == ptr:
            try:
                with bpy.context.temp_override(window=win):
                    bpy.ops.wm.window_close()
            except RuntimeError:
                pass
    return None


# ------------------------------------------------------------ camera link --

class CameraLink:
    """Keeps the two views' cameras in step while ``active``.

    Each view has an anchor: the centre and size of a world-space bounding
    box.  When one view moves, the other gets the same rotation and the same
    bbox-relative position, ``(location - centre) / size`` per axis, with the
    distance scaled by the ratio of the two boxes.

    * World space (``view_world_sync``): both views share one anchor, the box
      around both meshes, so the cameras are identical in world space.
    * Per mesh: each view is anchored to its own mesh, so the same relative
      spot on each mesh lines up even when the meshes are apart or differ in
      size.
    """

    def __init__(self, stage):
        self.stage = stage
        self.active = False
        self.anchors = {}
        self.last = {}

    @staticmethod
    def _anchor(lo, hi):
        size = hi - lo
        floor = max(size.length * 0.01, 1e-6)
        return ((lo + hi) / 2.0,
                Vector([max(v, floor) for v in size]),
                max(size.length, 1e-6))

    def _objects(self):
        mw = getattr(bpy.context.scene, "mesh_wrap", None)
        if mw is None:
            return None, None
        return mw.cage, mw.target

    def start(self):
        """(Re)compute anchors for the current sync space and start linking."""
        cage, target = self._objects()
        self.anchors = {}
        if cage is not None and target is not None:
            if world_sync():
                shared = self._anchor(*bounds(cage, target))
                self.anchors = {'CAGE': shared, 'TARGET': shared}
            else:
                self.anchors = {'CAGE': self._anchor(*bounds(cage)),
                                'TARGET': self._anchor(*bounds(target))}
        self.last = {}
        if len(self.anchors) == 2:
            self.active = True
            if not bpy.app.timers.is_registered(self._tick):
                bpy.app.timers.register(self._tick, first_interval=0.05)

    def set_space(self):
        """Switch between world-space and per-mesh sync, keeping the cage view."""
        self.start()
        self.follow('CAGE')

    def stop(self):
        self.active = False

    @staticmethod
    def _state(rv3d):
        return (tuple(round(v, 6) for v in rv3d.view_rotation),
                round(rv3d.view_distance, 6),
                tuple(round(v, 6) for v in rv3d.view_location),
                rv3d.view_perspective)

    def map_point(self, point, src_side):
        """World ``point`` on ``src_side`` -> matching point on the other side."""
        dst_side = 'TARGET' if src_side == 'CAGE' else 'CAGE'
        (sc, ss, _sd), (dc, ds, _dd) = self.anchors[src_side], self.anchors[dst_side]
        d = point - sc
        return dc + Vector((d.x / ss.x * ds.x, d.y / ss.y * ds.y,
                            d.z / ss.z * ds.z))

    def follow(self, src_side, enabled=True):
        """Copy ``src_side``'s camera onto the other view."""
        if not enabled or len(self.anchors) < 2:
            return
        dst_side = 'TARGET' if src_side == 'CAGE' else 'CAGE'
        _a, _r, src = self.stage.view(src_side)
        area, _r, dst = self.stage.view(dst_side)
        if src is None or dst is None:
            return
        ratio = self.anchors[dst_side][2] / self.anchors[src_side][2]
        dst.view_rotation = src.view_rotation
        dst.view_perspective = src.view_perspective
        dst.view_distance = src.view_distance * ratio
        dst.view_location = self.map_point(src.view_location, src_side)
        area.tag_redraw()
        self._remember()

    def _remember(self):
        for side in ('CAGE', 'TARGET'):
            _a, _r, rv3d = self.stage.view(side)
            if rv3d is not None:
                self.last[side] = self._state(rv3d)

    def _tick(self):
        if not self.active:
            return None
        try:
            linked = bpy.context.scene.mesh_wrap.link_views
            moved = None
            for side in ('CAGE', 'TARGET'):
                _a, _r, rv3d = self.stage.view(side)
                if rv3d is None:
                    return 0.1
                state = self._state(rv3d)
                if self.last.get(side) is not None and state != self.last[side]:
                    moved = side
                    break
            if moved is not None and linked:
                self.follow(moved)
            else:
                self._remember()
        except (ReferenceError, RuntimeError, AttributeError):
            return 0.1
        return 1.0 / 60.0
