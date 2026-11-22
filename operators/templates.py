"""Cage templates: a cage mesh with its material, texture and curves.

A template lives in ``<add-on>/templates/<name>/``:

    <name>.blend    the cage object (rest shape, no 'Wrapped' key) with its
                    materials; their images are packed into the file
    curves.json     the cage half of every pair, plus pair settings

Importing one brings the cage into the open scene, makes it the
Wrapper's Delight cage and recreates the pairs with only their cage half drawn.  The curve
editor then walks through them one by one (see ``editor.session``): each
pair waits for its target curve and the views frame the next one.
"""

import json
import os

import bpy

from ..model import display
from ..model import pairs as P
from ..model.properties import RESULT_KEY, redraw_views

FORMAT = 1
CURVES_FILE = "curves.json"


def templates_dir():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "templates")


def list_templates():
    """[(name, folder)] of every saved template, sorted by name."""
    base = templates_dir()
    out = []
    if os.path.isdir(base):
        for name in sorted(os.listdir(base), key=str.lower):
            folder = os.path.join(base, name)
            if (os.path.isfile(os.path.join(folder, name + ".blend"))
                    and os.path.isfile(os.path.join(folder, CURVES_FILE))):
                out.append((name, folder))
    return out


def _template_items(_self, _context):
    items = [(name, name, f"Import the '{name}' template")
             for name, _folder in list_templates()]
    # Blender keeps enum item strings alive only while referenced
    _template_items.cache = items or [('NONE', "No templates saved", "")]
    return _template_items.cache


_template_items.cache = []


def _material_images(obj):
    out = []
    for slot in obj.material_slots:
        mat = slot.material
        if mat is None or not mat.use_nodes or mat.node_tree is None:
            continue
        for node in mat.node_tree.nodes:
            if node.type == 'TEX_IMAGE' and node.image is not None:
                if node.image not in out:
                    out.append(node.image)
    return out


def _activate_texture_nodes(obj):
    """Make each material's image node the active one, so Solid > Texture
    shading (and the editor's Texture toggle) shows it."""
    for slot in obj.material_slots:
        mat = slot.material
        if mat is None or not mat.use_nodes or mat.node_tree is None:
            continue
        nodes = mat.node_tree.nodes
        tex = next((n for n in nodes if n.type == 'TEX_IMAGE'
                    and n.image is not None), None)
        if tex is not None:
            nodes.active = tex


def _pair_record(pair):
    return {
        "name": pair.name,
        "color": list(pair.color),
        "closed": bool(pair.closed),
        "samples": int(pair.samples),
        "weight": float(pair.weight),
        "lock": float(pair.lock),
        "enabled": bool(pair.enabled),
        "cage": [list(co) for co in P.coords(pair, 'CAGE')],
    }


class MESHWRAP_OT_template_save(bpy.types.Operator):
    """Save the cage, its material and texture, and the cage side of every
    curve as a template in the add-on's templates folder"""
    bl_idname = "meshwrap.template_save"
    bl_label = "Save Cage Template"
    bl_options = {'REGISTER'}

    name: bpy.props.StringProperty(name="Name", default="")

    @classmethod
    def poll(cls, context):
        from ..editor import session
        mw = context.scene.mesh_wrap
        return (mw.cage is not None and mw.cage.type == 'MESH'
                and not session.is_running())

    def invoke(self, context, event):
        if not self.name:
            self.name = context.scene.mesh_wrap.cage.name
        return context.window_manager.invoke_props_dialog(self, width=320)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "name")
        name = bpy.path.clean_name(self.name.strip())
        n = sum(1 for p in context.scene.mesh_wrap.pairs
                if P.has_side(p, 'CAGE'))
        layout.label(text=f"{n} curves  ·  saved to templates/{name}",
                     icon='INFO')
        if name and os.path.isdir(os.path.join(templates_dir(), name)):
            layout.label(text="A template with this name will be replaced",
                         icon='ERROR')

    def execute(self, context):
        mw = context.scene.mesh_wrap
        cage = mw.cage
        name = bpy.path.clean_name(self.name.strip())
        if not name:
            self.report({'ERROR'}, "Give the template a name")
            return {'CANCELLED'}
        folder = os.path.join(templates_dir(), name)
        os.makedirs(folder, exist_ok=True)
        blend = os.path.join(folder, name + ".blend")

        # a stand-alone copy of the cage on its rest shape
        tmp_mesh = cage.data.copy()
        tmp = cage.copy()
        tmp.data = tmp_mesh
        tmp.name = name
        keys = tmp_mesh.shape_keys
        if keys is not None and keys.key_blocks.get(RESULT_KEY) is not None:
            tmp.shape_key_remove(keys.key_blocks[RESULT_KEY])
        tmp.parent = None
        tmp.matrix_world = cage.matrix_world.copy()
        for mod in list(tmp.modifiers):
            # modifiers pointing at objects that won't be in the file
            if any(isinstance(getattr(mod, a, None), bpy.types.Object)
                   for a in ("object", "target", "offset_object")):
                tmp.modifiers.remove(mod)

        # pack textures into the template so it travels as one file
        packed_here = []
        skipped = []
        for img in _material_images(cage):
            if img.packed_file is not None or img.source != 'FILE':
                continue
            try:
                img.pack()
                packed_here.append(img)
            except RuntimeError:
                skipped.append(img.name)
        try:
            bpy.data.libraries.write(blend, {tmp}, path_remap='RELATIVE_ALL',
                                     compress=True)
        finally:
            for img in packed_here:
                try:
                    img.unpack(method='USE_ORIGINAL')
                except RuntimeError:
                    pass
            bpy.data.objects.remove(tmp)
            bpy.data.meshes.remove(tmp_mesh)

        data = {
            "format": FORMAT,
            "cage": name,
            "symmetry": bool(mw.symmetry),
            "exclude_group": mw.exclude_group,
            "pairs": [_pair_record(p) for p in mw.pairs
                      if P.has_side(p, 'CAGE')],
        }
        with open(os.path.join(folder, CURVES_FILE), "w",
                  encoding="utf-8") as fh:
            json.dump(data, fh, indent=1)
        msg = f"Saved template '{name}' ({len(data['pairs'])} curves)"
        if skipped:
            msg += f"; could not pack: {', '.join(skipped)}"
            self.report({'WARNING'}, msg)
        else:
            self.report({'INFO'}, msg)
        return {'FINISHED'}


class MESHWRAP_OT_template_import(bpy.types.Operator):
    """Bring a saved cage template into this scene: the cage with its
    textured material becomes the Wrapper's Delight cage, and its curves wait for
    their matching curves on the target"""
    bl_idname = "meshwrap.template_import"
    bl_label = "Import Cage Template"
    bl_options = {'REGISTER', 'UNDO'}

    template: bpy.props.EnumProperty(name="Template", items=_template_items)

    @classmethod
    def poll(cls, context):
        from ..editor import session
        return context.mode == 'OBJECT' and not session.is_running()

    def invoke(self, context, event):
        if not list_templates():
            self.report({'WARNING'}, "No templates saved yet (folder: "
                        f"{templates_dir()})")
            return {'CANCELLED'}
        return context.window_manager.invoke_props_dialog(self, width=320)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "template")
        mw = context.scene.mesh_wrap
        if len(mw.pairs):
            layout.label(text=f"Replaces the {len(mw.pairs)} pairs in this "
                              "scene", icon='ERROR')
        if mw.target is None:
            layout.label(text="Pick the target mesh, then open the curve "
                              "editor", icon='INFO')

    def execute(self, context):
        folder = dict(list_templates()).get(self.template)
        if folder is None:
            self.report({'ERROR'}, "Template not found")
            return {'CANCELLED'}
        name = self.template
        with open(os.path.join(folder, CURVES_FILE), encoding="utf-8") as fh:
            data = json.load(fh)
        blend = os.path.join(folder, name + ".blend")
        with bpy.data.libraries.load(blend, link=False) as (src, dst):
            dst.objects = list(src.objects)
        objs = [o for o in dst.objects if o is not None]
        cage = next((o for o in objs if o.type == 'MESH'
                     and o.name.startswith(data.get("cage", name))), None)
        if cage is None:
            cage = next((o for o in objs if o.type == 'MESH'), None)
        if cage is None:
            self.report({'ERROR'}, "The template has no mesh")
            return {'CANCELLED'}
        coll = context.collection or context.scene.collection
        for o in objs:
            coll.objects.link(o)
        _activate_texture_nodes(cage)

        mw = context.scene.mesh_wrap
        if mw.target == cage:
            mw.target = None
        mw.cage = cage
        mw.pairs.clear()
        for rec in data.get("pairs", []):
            pair = P.new_pair(mw, name=rec.get("name", "Pair"))
            pair.color = rec.get("color", pair.color)
            pair.closed = rec.get("closed", False)
            pair.samples = max(2, min(256, int(rec.get("samples", 8))))
            pair.weight = rec.get("weight", 1.0)
            pair.lock = rec.get("lock", 0.0)
            pair.enabled = rec.get("enabled", True)
            P.set_coords(pair, 'CAGE', [tuple(c) for c in rec["cage"]])
        mw.pair_index = 0 if len(mw.pairs) else -1
        mw.symmetry = bool(data.get("symmetry", mw.symmetry))
        group = data.get("exclude_group", "")
        mw.exclude_group = group if group in cage.vertex_groups else ""

        for o in context.view_layer.objects:
            o.select_set(False)
        cage.select_set(True)
        context.view_layer.objects.active = cage
        display.invalidate_all()
        redraw_views()
        n = len(mw.pairs)
        if mw.target is None:
            self.report({'INFO'}, f"Imported '{name}' with {n} curves: pick "
                                  "the target, then open the curve editor")
        else:
            self.report({'INFO'}, f"Imported '{name}' with {n} curves: open "
                                  "the curve editor to place them on the "
                                  "target")
        return {'FINISHED'}


CLASSES = (
    MESHWRAP_OT_template_save,
    MESHWRAP_OT_template_import,
)
