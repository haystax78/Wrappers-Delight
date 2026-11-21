"""Interactive curve editor: two-view stage, picking, drawing, HUD."""

import bpy

from . import session

CLASSES = session.CLASSES


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    session.request_stop()
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
