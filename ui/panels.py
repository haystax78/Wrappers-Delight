"""Sidebar interface (View3D > Sidebar > Wrapper's Delight).

    Wrapper's Delight    pick cage and target, see readiness at a glance
      Correspondences    the pair list, the active pair, the curve editor
      Fit                preset and options, the Wrap button
      Result             blend slider, bake, remove
      Texture Bake       move image textures between the two UV layouts
"""

import os

import bpy

from ..editor import session
from ..model import legacy
from ..model import pairs as P
from ..operators import fit as fit_ops
from ..operators import texture_bake as bake_ops

ADDON_NAME = "Wrapper's Delight"
CATEGORY = ADDON_NAME


def _version_text():
    """'1.0' from bl_info (patch shown only when it isn't 0)."""
    from .. import bl_info
    v = tuple(bl_info["version"])
    return ".".join(str(n) for n in (v if len(v) > 2 and v[2] else v[:2]))


def _ready(mw):
    return mw.cage is not None and mw.target is not None


def _update_notice(layout):
    """Small notice under the title when a newer release is out (or one is
    downloading / installed).  Blender can't colour panel text, so the
    green comes from the icon."""
    from .. import updater
    st = updater.state
    status = st["status"]
    if status == "AVAILABLE":
        row = layout.row()
        row.scale_y = 0.85
        row.operator("meshwrap.update_install", emboss=False,
                     text=f"{st['message']}  ·  click to install",
                     icon='STRIP_COLOR_04')
    elif status in {"DOWNLOADING", "INSTALLED"}:
        row = layout.row()
        row.scale_y = 0.85
        row.label(text=st["message"], icon='STRIP_COLOR_04')


class MESHWRAP_UL_pairs(bpy.types.UIList):
    bl_idname = "MESHWRAP_UL_pairs"

    def draw_item(self, context, layout, data, item, icon, active_data,
                  active_prop, index=0, flt_flag=0):
        pair = item
        row = layout.row(align=True)
        swatch = row.row(align=True)
        swatch.ui_units_x = 0.9
        swatch.prop(pair, "color", text="")
        row.separator(factor=0.5)
        name = row.row(align=True)
        name.active = pair.enabled
        name.prop(pair, "name", text="", emboss=False)

        need = P.missing_side(pair)
        info = row.row(align=True)
        info.alignment = 'RIGHT'
        if need is not None:
            info.label(text=f"no {need.lower()}", icon='DOT')
        elif P.is_point_pair(pair):
            info.label(text="point")
        else:
            info.label(text=f"{pair.samples}")
        row.prop(pair, "enabled", text="", emboss=False,
                 icon='CHECKBOX_HLT' if pair.enabled else 'CHECKBOX_DEHLT')
        row.prop(pair, "hidden", text="", emboss=False,
                 icon='HIDE_ON' if pair.hidden else 'HIDE_OFF')


class MESHWRAP_UL_bake_images(bpy.types.UIList):
    bl_idname = "MESHWRAP_UL_bake_images"

    def draw_item(self, context, layout, data, item, icon, active_data,
                  active_prop, index=0, flt_flag=0):
        path = bpy.path.abspath(item.filepath)
        row = layout.row(align=True)
        row.alert = not os.path.isfile(path)
        row.label(text=item.name or os.path.basename(path),
                  icon='ERROR' if row.alert else 'IMAGE_DATA')


class _Panel:
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = CATEGORY


class MESHWRAP_PT_main(_Panel, bpy.types.Panel):
    bl_idname = "MESHWRAP_PT_main"
    bl_label = f"{ADDON_NAME} {_version_text()}"

    def draw_header(self, context):
        self.layout.label(icon='MOD_SHRINKWRAP')

    def draw(self, context):
        layout = self.layout
        mw = context.scene.mesh_wrap
        _update_notice(layout)
        layout.use_property_split = True
        layout.use_property_decorate = False

        col = layout.column(align=True)
        col.prop(mw, "cage", icon='MESH_ICOSPHERE')
        col.prop(mw, "target", icon='MESH_MONKEY')
        row = layout.row(align=True)
        row.operator("meshwrap.template_import", text="Import Template",
                     icon='IMPORT')
        row.operator("meshwrap.template_save", text="Save Template",
                     icon='EXPORT')

        if legacy.has_legacy(mw):
            box = layout.box()
            box.label(text="Landmarks from an older version found",
                      icon='INFO')
            box.operator("meshwrap.legacy_import", icon='IMPORT')

        if not _ready(mw):
            layout.label(text="Pick both meshes to begin", icon='INFO')
            return

        done, _open = P.counts(mw)
        samples = P.matched_sample_total(mw)
        row = layout.row()
        row.alignment = 'CENTER'
        if samples >= 3:
            row.label(text=f"{done} pairs  ·  {samples} matched points",
                      icon='CHECKMARK')
        else:
            row.label(text="Draw at least one matching pair",
                      icon='CURVE_BEZCURVE')


class MESHWRAP_PT_pairs(_Panel, bpy.types.Panel):
    bl_idname = "MESHWRAP_PT_pairs"
    bl_label = "Correspondences"
    bl_parent_id = "MESHWRAP_PT_main"

    @classmethod
    def poll(cls, context):
        return _ready(context.scene.mesh_wrap)

    def draw_header_preset(self, context):
        self.layout.prop(context.scene.mesh_wrap, "show_in_viewport",
                         text="", emboss=False,
                         icon='HIDE_OFF' if context.scene.mesh_wrap
                         .show_in_viewport else 'HIDE_ON')

    def draw(self, context):
        layout = self.layout
        mw = context.scene.mesh_wrap

        big = layout.row()
        big.scale_y = 1.5
        if session.is_running():
            big.label(text="Editor open: Esc or Done to close",
                      icon='GREASEPENCIL')
        else:
            big.operator("meshwrap.editor", text="Open Curve Editor",
                         icon='GREASEPENCIL')
        sync = layout.row(align=True)
        sync.prop(mw, "link_views", text="Link Views", toggle=True,
                  icon='LINKED' if mw.link_views else 'UNLINKED')
        sub = sync.row(align=True)
        sub.active = mw.link_views
        sub.prop(mw, "view_world_sync", text="World Space", toggle=True,
                 icon='WORLD')

        row = layout.row()
        row.template_list("MESHWRAP_UL_pairs", "", mw, "pairs", mw,
                          "pair_index", rows=5)
        side = row.column(align=True)
        side.operator("meshwrap.pair_pick", text="", icon='EYEDROPPER')
        side.separator()
        side.operator("meshwrap.pair_remove", text="", icon='REMOVE')
        side.separator()
        side.operator("meshwrap.pair_move", text="",
                      icon='TRIA_UP').direction = 'UP'
        side.operator("meshwrap.pair_move", text="",
                      icon='TRIA_DOWN').direction = 'DOWN'
        side.separator()
        side.menu("MESHWRAP_MT_pairs_extra", text="", icon='DOWNARROW_HLT')

        pair = P.active_pair(mw)
        if pair is None:
            return
        box = layout.box()
        col = box.column()
        col.use_property_split = True
        col.use_property_decorate = False
        if not P.is_point_pair(pair):
            col.prop(pair, "samples")
            col.prop(pair, "closed")
        col.prop(pair, "weight")
        col.prop(pair, "lock", slider=True)

        row = box.row(align=True)
        for s in P.SIDES:
            n = len(P.points(pair, s))
            sub = row.row(align=True)
            sub.label(text=f"{s.title()}: {n} pts" if n else
                      f"{s.title()}: —",
                      icon='CHECKMARK' if n else 'DOT')
        row = box.row(align=True)
        row.operator("meshwrap.pair_reverse", text="Reverse Target",
                     icon='ARROW_LEFTRIGHT').side = 'TARGET'
        row.operator("meshwrap.pair_mirror", text="Mirror X",
                     icon='MOD_MIRROR')


class MESHWRAP_MT_pairs_extra(bpy.types.Menu):
    bl_idname = "MESHWRAP_MT_pairs_extra"
    bl_label = "Pair Options"

    def draw(self, context):
        layout = self.layout
        mw = context.scene.mesh_wrap
        layout.prop(mw, "default_samples")
        layout.separator()
        layout.operator("meshwrap.pair_reverse", text="Reverse Cage Side",
                        icon='ARROW_LEFTRIGHT').side = 'CAGE'
        layout.operator("meshwrap.pair_clear_side", text="Clear Cage Side",
                        icon='X').side = 'CAGE'
        layout.operator("meshwrap.pair_clear_side", text="Clear Target Side",
                        icon='X').side = 'TARGET'
        layout.separator()
        layout.operator("meshwrap.pair_recolor", icon='COLOR')
        layout.operator("meshwrap.pairs_clear", icon='TRASH')


class MESHWRAP_PT_fit(_Panel, bpy.types.Panel):
    bl_idname = "MESHWRAP_PT_fit"
    bl_label = "Fit"
    bl_parent_id = "MESHWRAP_PT_main"

    @classmethod
    def poll(cls, context):
        return _ready(context.scene.mesh_wrap)

    def draw(self, context):
        layout = self.layout
        mw = context.scene.mesh_wrap

        layout.row().prop(mw, "fit_preset", expand=True)

        col = layout.column()
        col.use_property_split = True
        col.use_property_decorate = False
        col.prop(mw, "fit_surface")
        sub = col.column()
        sub.active = mw.fit_surface
        sub.prop(mw, "reach", slider=True)
        sub.prop(mw, "lock_landmarks")
        if mw.cage is not None:
            sub.prop_search(mw, "exclude_group", mw.cage, "vertex_groups",
                            icon='GROUP_VERTEX')
        col.prop(mw, "warp_relax", slider=True)

        if mw.fit_preset == 'CUSTOM':
            box = col.box()
            box.active = mw.fit_surface
            box.prop(mw, "custom_iterations")
            box.prop(mw, "custom_stiffness", slider=True)
            box.prop(mw, "custom_detail", slider=True)
            box.prop(mw, "custom_relax")

        st = fit_ops.status()
        if st["running"]:
            if hasattr(layout, "progress"):
                layout.progress(factor=st["fraction"],
                                text=f"{st['label']}  (Esc to stop)")
            else:
                layout.label(text=f"{st['label']}  "
                                  f"{int(st['fraction'] * 100)}%")
            return
        row = layout.row()
        row.scale_y = 1.6
        row.enabled = P.matched_sample_total(mw) >= 3
        row.operator("meshwrap.fit", text="Wrap", icon='PLAY')


class MESHWRAP_PT_result(_Panel, bpy.types.Panel):
    bl_idname = "MESHWRAP_PT_result"
    bl_label = "Result"
    bl_parent_id = "MESHWRAP_PT_main"

    @classmethod
    def poll(cls, context):
        mw = context.scene.mesh_wrap
        return _ready(mw) and fit_ops.result_key(mw.cage) is not None

    def draw(self, context):
        layout = self.layout
        key = fit_ops.result_key(context.scene.mesh_wrap.cage)
        layout.prop(key, "value", text="Original ↔ Wrapped", slider=True)
        row = layout.row(align=True)
        row.operator("meshwrap.result_bake", icon='DUPLICATE')
        row.operator("meshwrap.result_clear", text="", icon='TRASH')


class MESHWRAP_PT_texture_bake(_Panel, bpy.types.Panel):
    bl_idname = "MESHWRAP_PT_texture_bake"
    bl_label = "Texture Bake"
    bl_parent_id = "MESHWRAP_PT_main"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return _ready(context.scene.mesh_wrap)

    def draw(self, context):
        layout = self.layout
        mw = context.scene.mesh_wrap
        tb = mw.texture_bake

        layout.row().prop(tb, "mode", expand=True)

        row = layout.row()
        row.template_list("MESHWRAP_UL_bake_images", "", tb, "images", tb,
                          "image_index", rows=4)
        side = row.column(align=True)
        side.operator("meshwrap.bake_image_add", text="", icon='ADD')
        side.operator("meshwrap.bake_image_remove", text="", icon='REMOVE')
        side.separator()
        side.operator("meshwrap.bake_image_clear", text="", icon='TRASH')

        layout.row().prop(tb, "direction", expand=True)

        col = layout.column()
        col.use_property_split = True
        col.use_property_decorate = False
        col.prop(tb, "output_dir", text="Output")
        col.prop(tb, "suffix")
        col.row().prop(tb, "supersample", expand=True)
        col.prop(tb, "margin")
        col.prop(tb, "distance", slider=True)
        col.prop(tb, "normal_angle")
        if tb.mode == 'RESAMPLE':
            col.prop(tb, "use_gpu")

        if fit_ops.result_key(mw.cage) is None:
            layout.label(text="No Wrapped result: cage used as it is",
                         icon='INFO')

        st = bake_ops.status()
        if st["running"]:
            frac = st["done"] / max(st["total"], 1)
            text = (f"{st['done']} / {st['total']}  ·  {st['label']}  "
                    f"(Esc to stop)")
            if hasattr(layout, "progress"):
                layout.progress(factor=frac, text=text)
            else:
                layout.label(text=f"{int(frac * 100)}%  {text}")
            return
        row = layout.row()
        row.scale_y = 1.6
        row.operator("meshwrap.texture_bake", text="Process",
                     icon='RENDER_STILL')


CLASSES = (
    MESHWRAP_UL_pairs,
    MESHWRAP_UL_bake_images,
    MESHWRAP_MT_pairs_extra,
    MESHWRAP_PT_main,
    MESHWRAP_PT_pairs,
    MESHWRAP_PT_fit,
    MESHWRAP_PT_result,
    MESHWRAP_PT_texture_bake,
)
