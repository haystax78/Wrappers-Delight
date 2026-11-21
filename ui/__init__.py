"""Sidebar panels and the passive viewport overlay."""

import bpy

from . import overlay, panels


def register():
    for cls in panels.CLASSES:
        bpy.utils.register_class(cls)
    overlay.register()


def unregister():
    overlay.unregister()
    for cls in reversed(panels.CLASSES):
        bpy.utils.unregister_class(cls)
