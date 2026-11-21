"""Add-on preferences."""

import bpy

ADDON_ID = __package__


class MeshWrapPreferences(bpy.types.AddonPreferences):
    bl_idname = ADDON_ID

    editor_layout: bpy.props.EnumProperty(
        name="Curve Editor Opens In",
        items=[
            ('WINDOW', "New Window",
             "Open the cage and target views in a separate window; your "
             "layout is left untouched"),
            ('SPLIT', "Current View",
             "Split the 3D Viewport the editor is started from"),
        ],
        default='WINDOW',
    )

    def draw(self, context):
        self.layout.prop(self, "editor_layout")


class _Defaults:
    editor_layout = 'WINDOW'


def get():
    try:
        return bpy.context.preferences.addons[ADDON_ID].preferences
    except (KeyError, AttributeError):
        return _Defaults


CLASSES = (MeshWrapPreferences,)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
