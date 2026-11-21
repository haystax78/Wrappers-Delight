"""Show correspondences in ordinary 3D views while the editor is closed."""

import traceback

import bpy

from ..editor import scene_draw, session
from ..model import pairs as P
from ..operators.fit import result_key

_HANDLE = None
_FAILED = False


def _visible(obj, space):
    try:
        return obj.visible_get() and obj.visible_in_viewport_get(space)
    except (AttributeError, RuntimeError, TypeError):
        return obj.visible_get()


def _draw():
    global _FAILED
    try:
        if session.is_running():
            return
        context = bpy.context
        mw = getattr(context.scene, "mesh_wrap", None)
        if mw is None or not mw.show_in_viewport or not len(mw.pairs):
            return
        space = context.space_data
        region, rv3d = context.region, context.region_data
        active = P.active_pair(mw)
        opts = scene_draw.DrawOptions(
            active_uid=active.uid if active else None)
        cage = mw.cage
        if cage is not None and _visible(cage, space):
            key = result_key(cage)
            # cage points live on the rest shape: skip while it's deformed
            if key is None or key.value == 0.0:
                scene_draw.draw_side(mw, 'CAGE', cage, region, rv3d, opts)
        if mw.target is not None and _visible(mw.target, space):
            scene_draw.draw_side(mw, 'TARGET', mw.target, region, rv3d,
                                 opts)
    except Exception:  # noqa: BLE001
        if not _FAILED:
            _FAILED = True
            traceback.print_exc()


def register():
    global _HANDLE
    _HANDLE = bpy.types.SpaceView3D.draw_handler_add(
        _draw, (), 'WINDOW', 'POST_PIXEL')


def unregister():
    global _HANDLE
    if _HANDLE is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_HANDLE, 'WINDOW')
        _HANDLE = None
