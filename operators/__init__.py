"""Sidebar operators: pair management, fitting, templates, texture bake."""

import bpy

from . import fit, pairs, templates, texture_bake

CLASSES = (pairs.CLASSES + fit.CLASSES + templates.CLASSES
           + texture_bake.CLASSES)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
