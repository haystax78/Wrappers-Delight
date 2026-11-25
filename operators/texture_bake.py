"""Texture Bake: move image textures from one mesh's UV layout to the other's.

Once the cage is wrapped onto the target the two surfaces coincide, so a
texture painted in one UV layout can be re-projected into the other.  Each
listed image is baked with Cycles (selected-to-active, Emit pass) from the
source mesh, where it is shown through its UVs, onto the destination mesh's
UVs.

Fidelity rules
    * Values are transferred as raw data.  Both the source and the bake
      image are flagged Non-Color with channel-packed alpha, so no colour
      space conversion, (un)premultiplication or view transform is applied
      anywhere: an sRGB PNG comes out sRGB-encoded, a linear EXR linear,
      a normal map stays a normal map.
    * The bake buffer is 32-bit float, optionally at 2x / 4x the output
      size and box-filtered down (anti-aliased edges and seams).
    * Alpha, when the source has it, is baked in a second pass.
    * The result is written back into a fresh copy of the source image and
      saved from there, so Blender keeps the file's own format, bit depth
      (8/16-bit PNG/TIFF, half/float EXR ...), channels, codec and size.
    * PSD / PSB files go through :mod:`..model.psd` instead (Blender can't
      write them): the merged image is read and written raw, keeping mode,
      bit depth, alpha, resolution and the ICC profile.
    * The cage is baked in its Wrapped shape (full strength) when one exists.

Outputs are named ``<source name><suffix><source extension>``.  The UI runs
the bake modally, one image per timer tick, so the panel's progress bar
updates after every image (Esc stops after the current one).  Afterwards
every datablock created during the run (loaded and generated images,
temporary meshes, objects, materials, collection) is removed.

Scene state touched during the bake (engine, sampling, selection, the
Wrapped key value) is restored afterwards.
"""

import os
import zlib

import numpy as np

import bpy

from ..editor import session
from ..model import psd
from . import fit as fit_ops

IMAGE_EXTENSIONS = (
    "*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.exr;*.tga;*.bmp;*.webp;*.hdr;"
    "*.jp2;*.j2c;*.dpx;*.cin;*.rgb;*.sgi;*.psd;*.psb")

# Largest bake buffer (pixels) supersampling may grow to; beyond this the
# factor is reduced (a float RGBA buffer of this size is 1 GiB).
MAX_BAKE_PIXELS = 8192 * 8192

_TMP = "WD_TextureBake"


class BakeError(Exception):
    pass


def _settings(context):
    return context.scene.mesh_wrap.texture_bake


def _abs(path):
    return os.path.normpath(bpy.path.abspath(path)) if path else ""


# ---------------------------------------------------------- list editing ---

class MESHWRAP_OT_bake_image_add(bpy.types.Operator):
    """Add image files to the Texture Bake list"""
    bl_idname = "meshwrap.bake_image_add"
    bl_label = "Add Images"
    bl_options = {'REGISTER', 'UNDO'}

    directory: bpy.props.StringProperty(subtype='DIR_PATH')
    files: bpy.props.CollectionProperty(
        type=bpy.types.OperatorFileListElement,
        options={'HIDDEN', 'SKIP_SAVE'})
    filter_image: bpy.props.BoolProperty(default=True, options={'HIDDEN'})
    filter_folder: bpy.props.BoolProperty(default=True, options={'HIDDEN'})
    filter_glob: bpy.props.StringProperty(default=IMAGE_EXTENSIONS,
                                          options={'HIDDEN'})

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        tb = _settings(context)
        have = {_abs(item.filepath) for item in tb.images}
        added = 0
        for f in self.files:
            if not f.name:
                continue
            path = os.path.join(self.directory, f.name)
            if _abs(path) in have or not os.path.isfile(path):
                continue
            item = tb.images.add()
            item.name = f.name
            item.filepath = path
            have.add(_abs(path))
            added += 1
        if added:
            tb.image_index = len(tb.images) - 1
        return {'FINISHED'}


class MESHWRAP_OT_bake_image_remove(bpy.types.Operator):
    """Remove the selected image from the Texture Bake list"""
    bl_idname = "meshwrap.bake_image_remove"
    bl_label = "Remove Image"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        tb = _settings(context)
        return 0 <= tb.image_index < len(tb.images)

    def execute(self, context):
        tb = _settings(context)
        tb.images.remove(tb.image_index)
        tb.image_index = min(tb.image_index, len(tb.images) - 1)
        return {'FINISHED'}


class MESHWRAP_OT_bake_image_clear(bpy.types.Operator):
    """Remove every image from the Texture Bake list"""
    bl_idname = "meshwrap.bake_image_clear"
    bl_label = "Clear List"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return len(_settings(context).images) > 0

    def execute(self, context):
        tb = _settings(context)
        tb.images.clear()
        tb.image_index = 0
        return {'FINISHED'}


# --------------------------------------------------------------- baking ---

def _active_uv_name(obj):
    uv = obj.data.uv_layers.active
    if uv is None:
        raise BakeError(f"'{obj.name}' has no UV map")
    return uv.name


def _new_material(name):
    mat = bpy.data.materials.new(name)
    if mat.node_tree is None:          # Blender < 5.0
        mat.use_nodes = True
    mat.node_tree.nodes.clear()
    return mat


def _temp_object(context, obj, coll, mat, uv_name):
    """Evaluated copy of ``obj`` (modifiers and shape keys applied) with a
    single material and ``uv_name`` as its active + render UV map."""
    dg = context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(obj.evaluated_get(dg),
                                         preserve_all_data_layers=True,
                                         depsgraph=dg)
    me.name = f"{_TMP}_{obj.name}"
    uv = me.uv_layers.get(uv_name)
    if uv is None:
        bpy.data.meshes.remove(me)
        raise BakeError(f"'{obj.name}' lost its UV map '{uv_name}' in its "
                        "modifiers")
    me.uv_layers.active = uv
    uv.active_render = True
    me.materials.clear()
    idx = me.attributes.get("material_index")
    if idx is not None:
        me.attributes.remove(idx)
    me.materials.append(mat)
    tmp = bpy.data.objects.new(me.name, me)
    tmp.matrix_world = obj.matrix_world.copy()
    coll.objects.link(tmp)
    return tmp


def _world_size(obj):
    me = obj.data
    n = len(me.vertices)
    if n == 0:
        raise BakeError(f"'{obj.name}' has no geometry")
    co = np.empty(n * 3, np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    m = np.array(obj.matrix_world, np.float64)
    co = co @ m[:3, :3].T + m[:3, 3]
    return float(np.linalg.norm(co.max(0) - co.min(0)))


class _Restore:
    """Remember attributes and put them back in reverse order."""

    def __init__(self):
        self._saved = []

    def set(self, owner, attr, value):
        self._saved.append((owner, attr, getattr(owner, attr)))
        setattr(owner, attr, value)

    def restore(self):
        for owner, attr, value in reversed(self._saved):
            try:
                setattr(owner, attr, value)
            except Exception:  # noqa: BLE001 - restore as much as possible
                pass
        self._saved.clear()


class _Rig:
    """Temporary objects, materials and scene settings for one bake run."""

    def __init__(self, context, src_obj, dst_obj, settings):
        self.context = context
        self.scene = context.scene
        self.view_layer = context.view_layer
        self.settings = settings
        self.state = _Restore()
        self.selection = (list(context.selected_objects),
                          context.view_layer.objects.active)
        self.coll = None
        self.mats = []
        self.objs = []
        try:
            self._build(src_obj, dst_obj)
        except BaseException:
            self.close()
            raise

    def _build(self, src_obj, dst_obj):
        scene = self.scene
        if not hasattr(scene, "cycles"):
            raise BakeError("Texture Bake needs the Cycles add-on enabled")
        self.src_uv = _active_uv_name(src_obj)
        self.dst_uv = _active_uv_name(dst_obj)

        self.coll = bpy.data.collections.new(_TMP)
        scene.collection.children.link(self.coll)

        # Source: image (through its UVs) -> emission.
        self.src_mat = _new_material(f"{_TMP}_source")
        self.mats.append(self.src_mat)
        nt = self.src_mat.node_tree
        uv = nt.nodes.new("ShaderNodeUVMap")
        uv.uv_map = self.src_uv
        self.src_tex = nt.nodes.new("ShaderNodeTexImage")
        self.src_tex.interpolation = 'Linear'
        self.src_tex.extension = 'REPEAT'
        self.emit = nt.nodes.new("ShaderNodeEmission")
        self.emit.inputs["Strength"].default_value = 1.0
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(uv.outputs["UV"], self.src_tex.inputs["Vector"])
        nt.links.new(self.emit.outputs["Emission"], out.inputs["Surface"])

        # Destination: the active image node Cycles bakes into.
        self.dst_mat = _new_material(f"{_TMP}_target")
        self.mats.append(self.dst_mat)
        self.dst_tex = self.dst_mat.node_tree.nodes.new("ShaderNodeTexImage")
        self.dst_tex.select = True
        self.dst_mat.node_tree.nodes.active = self.dst_tex

        # Evaluate the cage in its full Wrapped shape.
        cage = scene.mesh_wrap.cage
        key = fit_ops.result_key(cage)
        shape = _Restore()
        if key is not None:
            shape.set(cage, "show_only_shape_key", False)
            shape.set(key, "mute", False)
            shape.set(key, "value", 1.0)
        try:
            self.src = _temp_object(self.context, src_obj, self.coll,
                                    self.src_mat, self.src_uv)
            self.objs.append(self.src)
            self.dst = _temp_object(self.context, dst_obj, self.coll,
                                    self.dst_mat, self.dst_uv)
            self.objs.append(self.dst)
        finally:
            shape.restore()

        size = max(_world_size(self.dst), _world_size(self.src))
        self.extrusion = max(size * self.settings.distance, 1e-6)

        st = self.state
        st.set(scene.render, "engine", 'CYCLES')
        st.set(scene.render, "use_simplify", False)
        cy = scene.cycles
        st.set(cy, "samples", 1)
        st.set(cy, "film_exposure", 1.0)
        for name in ("sample_clamp_direct", "sample_clamp_indirect"):
            if hasattr(cy, name):
                st.set(cy, name, 0.0)
        if hasattr(cy, "use_adaptive_sampling"):
            st.set(cy, "use_adaptive_sampling", False)

        self.select()

    def select(self):
        """Exactly the two temporary objects selected, destination active
        (re-applied before every bake)."""
        vl = self.view_layer
        for o in vl.objects:
            if o.select_get():
                o.select_set(False)
        self.src.select_set(True)
        self.dst.select_set(True)
        vl.objects.active = self.dst

    # -- one image -------------------------------------------------------

    def bake_file(self, path, out_path, supersample):
        """Bake one image file to ``out_path``; returns the supersampling
        factor actually used."""
        if psd.is_psd(path):
            return self._bake_psd(path, out_path, supersample)
        return self._bake_blender_image(path, out_path, supersample)

    def _bake_blender_image(self, path, out_path, supersample):
        img = bpy.data.images.load(path, check_existing=False)
        try:
            if img.source != 'FILE':
                raise BakeError("only single image files are supported")
            # Raw values in and out: no colour or alpha conversion anywhere.
            img.colorspace_settings.name = 'Non-Color'
            img.alpha_mode = 'CHANNEL_PACKED'
            w, h = img.size
            if w == 0 or h == 0:
                raise BakeError("could not be read")
            planes = img.depth // 4 if img.is_float else img.depth
            out, ss = self._bake_texture(img, w, h, planes == 32,
                                         supersample)
            # Write into the loaded source copy: its buffer still carries
            # the file's format, bit depth, channels and codec, which
            # save() reuses.
            img.pixels.foreach_set(out.ravel())
            img.save(filepath=out_path, quality=100)
            return ss
        finally:
            bpy.data.images.remove(img)

    def _bake_psd(self, path, out_path, supersample):
        # Blender can't write PSD and its reader's alpha handling varies, so
        # the merged image is read and written by model.psd (raw values).
        try:
            info, planes = psd.read(path)
        except (psd.PsdError, OSError, zlib.error, ValueError) as exc:
            raise BakeError(str(exc)) from exc
        h, w = info.height, info.width
        px = np.ones((h, w, 4), np.float32)
        if info.base == 1:
            px[..., :3] = planes[0][..., None]
        else:
            px[..., :3] = np.moveaxis(planes[:3], 0, -1)
        if info.has_alpha:
            px[..., 3] = planes[info.base]
        del planes
        img = bpy.data.images.new(f"{_TMP}_psd", w, h, alpha=True,
                                  float_buffer=True, is_data=True)
        try:
            img.alpha_mode = 'CHANNEL_PACKED'
            img.pixels.foreach_set(px[::-1].ravel())    # Blender: bottom up
            del px
            out, ss = self._bake_texture(img, w, h, info.has_alpha,
                                         supersample)
        finally:
            bpy.data.images.remove(img)
        out = out[::-1]                                  # PSD: top down
        k = info.kept
        if info.base == 1:
            res = out[..., [0, 3]] if info.has_alpha else out[..., [0]]
        else:
            res = out[..., :k]
        try:
            psd.write(out_path, info, np.moveaxis(res, -1, 0))
        except OSError as exc:
            raise BakeError(str(exc)) from exc
        return ss

    def _bake_texture(self, img, w, h, has_alpha, supersample):
        """Bake ``img`` (shown through the source UVs) into the destination
        UVs at w x h; returns (float32 (h, w, 4) bottom-up, supersampling)."""
        ss = supersample
        while ss > 1 and w * h * ss * ss > MAX_BAKE_PIXELS:
            ss //= 2
        buf = bpy.data.images.new(f"{_TMP}_buffer", w * ss, h * ss,
                                  alpha=True, float_buffer=True,
                                  is_data=True)
        try:
            self.src_tex.image = img
            self.dst_tex.image = buf
            nt = self.src_mat.node_tree
            nt.links.new(self.src_tex.outputs["Color"],
                         self.emit.inputs["Color"])
            out = self._bake(buf, ss)
            if has_alpha:
                nt.links.new(self.src_tex.outputs["Alpha"],
                             self.emit.inputs["Color"])
                out[..., 3] = self._bake(buf, ss)[..., 0]
            else:
                out[..., 3] = 1.0
            return out, ss
        finally:
            self.src_tex.image = None
            self.dst_tex.image = None
            bpy.data.images.remove(buf)

    def _bake(self, buf, ss):
        b = self.settings
        self.select()
        bpy.ops.object.bake(
            type='EMIT',
            target='IMAGE_TEXTURES',
            save_mode='INTERNAL',
            use_clear=True,
            use_selected_to_active=True,
            use_cage=False,
            cage_extrusion=self.extrusion,
            max_ray_distance=self.extrusion * 2.0,
            margin=b.margin * ss,
            margin_type='ADJACENT_FACES',
            uv_layer=self.dst_uv,
        )
        bw, bh = buf.size
        px = np.empty(bw * bh * 4, np.float32)
        buf.pixels.foreach_get(px)
        px = px.reshape(bh, bw, 4)
        if ss > 1:      # box filter down to the output resolution
            px = px.reshape(bh // ss, ss, bw // ss, ss, 4).mean(axis=(1, 3),
                                                              dtype=np.float64)
            px = px.astype(np.float32)
        return px

    # -- teardown --------------------------------------------------------

    def close(self):
        self.state.restore()
        for obj in self.objs:
            me = obj.data
            bpy.data.objects.remove(obj)
            bpy.data.meshes.remove(me)
        self.objs.clear()
        for mat in self.mats:
            bpy.data.materials.remove(mat)
        self.mats.clear()
        if self.coll is not None:
            bpy.data.collections.remove(self.coll)
            self.coll = None
        selected, active = self.selection
        vl = self.view_layer
        for o in vl.objects:
            if o.select_get():
                o.select_set(False)
        for obj in selected:
            try:
                if obj.name in vl.objects:
                    obj.select_set(True)
            except ReferenceError:
                pass
        try:
            if active is None or active.name in vl.objects:
                vl.objects.active = active
        except ReferenceError:
            pass


class _DataSnapshot:
    """Every datablock the bake can create; ``purge()`` removes whatever
    appeared since, so no loaded or generated image (or temporary mesh,
    material, object, collection) is left in the file."""

    KINDS = ("objects", "meshes", "materials", "images", "collections")

    def __init__(self):
        self.before = {k: {d.as_pointer() for d in getattr(bpy.data, k)}
                       for k in self.KINDS}

    def purge(self):
        removed = 0
        for kind in self.KINDS:
            coll = getattr(bpy.data, kind)
            new = [d for d in coll
                   if d.as_pointer() not in self.before[kind]]
            for d in new:
                try:
                    coll.remove(d)
                    removed += 1
                except (ReferenceError, RuntimeError):
                    pass
        return removed


def _jobs(tb, out_dir):
    """[(source, output)] after checking the list up front."""
    jobs, seen = [], set()
    for item in tb.images:
        src = _abs(item.filepath)
        if not os.path.isfile(src):
            raise BakeError(f"File not found: {src}")
        stem, ext = os.path.splitext(os.path.basename(src))
        name = f"{stem}{tb.suffix}{ext}"
        dst = os.path.normpath(os.path.join(out_dir, name))
        if os.path.normcase(dst) == os.path.normcase(src):
            raise BakeError(f"'{name}' would overwrite its source: set a "
                            "suffix or pick a different output folder")
        key = os.path.normcase(dst)
        if key in seen:
            raise BakeError(f"Two images would both be saved as '{name}'")
        seen.add(key)
        jobs.append((src, dst))
    return jobs


# Progress shown by the panel while the modal bake runs.
_STATUS = {"running": False, "done": 0, "total": 0, "label": ""}


def status():
    return _STATUS


def _redraw():
    from ..model.properties import redraw_views
    redraw_views()


class MESHWRAP_OT_texture_bake(bpy.types.Operator):
    """Bake every listed image from one mesh's UV layout into the other's"""
    bl_idname = "meshwrap.texture_bake"
    bl_label = "Process"
    bl_description = ("Bake every listed image into the other mesh's UV "
                      "layout and save it to the output folder (name + "
                      "suffix) in the same format, bit depth and resolution")
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        mw = context.scene.mesh_wrap
        tb = mw.texture_bake
        if _STATUS["running"]:
            cls.poll_message_set("A texture bake is already running")
            return False
        if mw.cage is None or mw.target is None:
            cls.poll_message_set("Pick the cage and the target first")
            return False
        if context.mode != 'OBJECT':
            cls.poll_message_set("Switch to Object Mode")
            return False
        if session.is_running() or fit_ops.status()["running"]:
            cls.poll_message_set("Close the curve editor / wait for the fit")
            return False
        if not tb.images:
            cls.poll_message_set("Add images to the list")
            return False
        if not tb.output_dir:
            cls.poll_message_set("Choose an output folder")
            return False
        return True

    # -- shared steps ----------------------------------------------------

    def _start(self, context):
        mw = context.scene.mesh_wrap
        tb = mw.texture_bake
        out_dir = tb.output_dir
        if out_dir.startswith("//") and not bpy.data.filepath:
            self.report({'ERROR'}, "Save the .blend file first, or use an "
                                   "absolute output folder")
            return False
        self.out_dir = _abs(out_dir)
        try:
            self.jobs = _jobs(tb, self.out_dir)
            os.makedirs(self.out_dir, exist_ok=True)
        except (BakeError, OSError) as exc:
            self.report({'ERROR'}, str(exc))
            return False

        if tb.direction == 'CAGE_TO_TARGET':
            src_obj, dst_obj = mw.cage, mw.target
        else:
            src_obj, dst_obj = mw.target, mw.cage
        self.supersample = int(tb.supersample)
        self.index = 0
        self.failed, self.notes = [], []
        self.snapshot = _DataSnapshot()
        try:
            self.rig = _Rig(context, src_obj, dst_obj, tb)
        except (BakeError, RuntimeError) as exc:
            self.snapshot.purge()
            self.report({'ERROR'}, f"Texture Bake: {exc}")
            return False
        _STATUS.update(running=True, done=0, total=len(self.jobs),
                       label=os.path.basename(self.jobs[0][0]))
        context.window_manager.progress_begin(0, len(self.jobs))
        return True

    def _step(self, context):
        """Bake the next image; False when the list is done."""
        src, dst = self.jobs[self.index]
        name = os.path.basename(src)
        n = len(self.jobs)
        print(f"Wrapper's Delight: baking {name} ({self.index + 1}/{n})")
        try:
            ss = self.rig.bake_file(src, dst, self.supersample)
            if ss != self.supersample:
                self.notes.append(f"{name}: supersampling lowered to {ss}x "
                                  "(memory)")
        except (BakeError, RuntimeError) as exc:
            self.failed.append(f"{name}: {exc}")
        self.index += 1
        context.window_manager.progress_update(self.index)
        nxt = (os.path.basename(self.jobs[self.index][0])
               if self.index < n else "")
        _STATUS.update(done=self.index, label=nxt)
        _redraw()
        return self.index < n

    def _finish(self, context, cancelled=False):
        context.window_manager.progress_end()
        try:
            self.rig.close()
        finally:
            self.snapshot.purge()
            _STATUS.update(running=False, done=0, total=0, label="")
            _redraw()

        for msg in self.notes:
            self.report({'INFO'}, msg)
        for msg in self.failed:
            self.report({'WARNING'}, msg)
        done = self.index - len(self.failed)
        if cancelled:
            self.report({'WARNING'}, f"Texture Bake stopped: {done} of "
                                     f"{len(self.jobs)} baked")
            return {'CANCELLED'}
        if done == 0:
            self.report({'ERROR'}, f"No images baked ({len(self.failed)} "
                                   "failed)")
            return {'CANCELLED'}
        extra = f", {len(self.failed)} failed" if self.failed else ""
        self.report({'INFO'}, f"Baked {done} image{'s' * (done != 1)} to "
                              f"{self.out_dir}{extra}")
        return {'FINISHED'}

    # -- blocking (scripts, background) ----------------------------------

    def execute(self, context):
        if not self._start(context):
            return {'CANCELLED'}
        try:
            while self._step(context):
                pass
        except BaseException:
            self._finish(context, cancelled=True)
            raise
        return self._finish(context)

    # -- modal (UI): one image per timer tick so the progress bar redraws --

    def invoke(self, context, event):
        if not self._start(context):
            return {'CANCELLED'}
        wm = context.window_manager
        self.timer = wm.event_timer_add(0.05, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'ESC' and event.value == 'PRESS':
            return self._end_modal(context, cancelled=True)
        if event.type != 'TIMER':
            # Swallow input so nothing changes the scene mid-bake.
            return {'RUNNING_MODAL'}
        try:
            more = self._step(context)
        except Exception as exc:  # noqa: BLE001 - always clean up
            self.report({'ERROR'}, f"Texture Bake failed: {exc}")
            return self._end_modal(context, cancelled=True)
        if more:
            return {'RUNNING_MODAL'}
        return self._end_modal(context)

    def _end_modal(self, context, cancelled=False):
        context.window_manager.event_timer_remove(self.timer)
        self.timer = None
        return self._finish(context, cancelled)


CLASSES = (
    MESHWRAP_OT_bake_image_add,
    MESHWRAP_OT_bake_image_remove,
    MESHWRAP_OT_bake_image_clear,
    MESHWRAP_OT_texture_bake,
)
