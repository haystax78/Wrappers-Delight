"""Wrapper's Delight: fit a cage mesh onto a target surface from matching curves.

Draw a curve on the cage and the same feature on the target (eye rims, lip
line, jaw, ears ...).  Each pair yields evenly spaced matched points; the fit
aligns the cage to them, warps it smoothly through them and finally conforms
it to the target surface.  The result lands in a 'Wrapped' shape key, so the
original shape is always one slider away.

Package layout
    model/      scene properties, pair helpers, polyline maths, draw cache
    solver/     alignment, landmark warp, surface conform, job pipeline
    operators/  sidebar actions and the (modal) fit
    editor/     the two-view curve editor
    ui/         sidebar panels and the passive viewport overlay
"""

bl_info = {
    "name": "Wrapper's Delight",
    "author": "haystax78",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > Wrapper's Delight",
    "description": "Fit a cage mesh onto a target surface from matching "
                   "curves",
    "category": "Mesh",
}

import bpy
from bpy.app.handlers import persistent

from . import editor, operators, preferences, ui, updater
from .model import display, properties

_MODULES = (properties, preferences, operators, editor, ui, updater)


@persistent
def _reset_cache(*_args):
    display.invalidate_all()


_HANDLERS = (bpy.app.handlers.load_post, bpy.app.handlers.undo_post,
             bpy.app.handlers.redo_post)


def register():
    for module in _MODULES:
        module.register()
    for handlers in _HANDLERS:
        handlers.append(_reset_cache)


def unregister():
    for handlers in _HANDLERS:
        if _reset_cache in handlers:
            handlers.remove(_reset_cache)
    for module in reversed(_MODULES):
        module.unregister()
