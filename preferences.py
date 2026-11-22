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
    auto_update: bpy.props.BoolProperty(
        name="Check for Updates",
        default=True,
        description="When Blender starts, look for a newer release on "
                    "GitHub and offer to install it from the sidebar "
                    "(needs Allow Online Access in Preferences > System)",
    )

    def draw(self, context):
        from . import updater
        layout = self.layout
        layout.prop(self, "editor_layout")
        box = layout.box()
        row = box.row()
        row.prop(self, "auto_update")
        row.operator("meshwrap.update_check", icon='FILE_REFRESH')
        st = updater.state
        if st["message"]:
            row = box.row()
            if st["status"] == "AVAILABLE":
                row.operator("meshwrap.update_install", text=st["message"],
                             icon='IMPORT')
            else:
                row.label(text=st["message"],
                          icon='ERROR' if st["status"] in {"ERROR", "OFFLINE"}
                          else 'INFO')
        box.label(text=f"Installed: {updater.version_text(updater.installed_version())}"
                       f"  ·  Releases: {updater.RELEASES_PAGE}")


class _Defaults:
    editor_layout = 'WINDOW'
    auto_update = True


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
