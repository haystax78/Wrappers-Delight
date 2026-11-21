"""Sidebar operators: pair management and fitting."""

import bpy

from . import fit, pairs

CLASSES = pairs.CLASSES + fit.CLASSES


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
