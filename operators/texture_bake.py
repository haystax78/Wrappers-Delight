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
    * Normal matching: Blender's bake takes the first surface the ray hits,
      even a back face or a nearby part facing another way (the back of the
      ear top vs. the head).  Three extra passes per bake size record the
      destination normal, the normal of the face that was hit and the
      destination position; texels whose hit is more than ``normal_angle``
      off are searched again in Python (same ray, faces outside the angle
      passed through) and filled by sampling the source image at the new
      hit's UV (bilinear, repeat, like the image node).

Modes
    * Bake: every image is baked with Cycles as above.
    * Resample: one Cycles bake per bake size records, for every
      destination texel, the source UV its ray lands on (with normal
      matching applied); each image is then just sampled through that map
      in numpy (bilinear, repeat).  Margin is an outward extension of the
      destination islands (nearest covered texel, computed once per size)
      instead of Blender's adjacent-faces fill.

Outputs are named ``<source name><suffix><source extension>``.  The UI runs
the bake modally, one image per timer tick, so the panel's progress bar
updates after every image (Esc stops after the current one).  Afterwards
every datablock created during the run (loaded and generated images,
temporary meshes, objects, materials, collection) is removed.

Scene state touched during the bake (engine, sampling, selection, the
Wrapped key value) is restored afterwards.
"""

import math
import os
import zlib

import numpy as np

import bpy

from ..editor import session
from ..model import psd
from ..solver.surface import tree_from_arrays, triangles_local
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


def _world_bounds(obj):
    me = obj.data
    co = np.empty(len(me.vertices) * 3, np.float64)
    me.vertices.foreach_get("co", co)
    m = np.array(obj.matrix_world, np.float64)
    co = co.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]
    return co.min(0), co.max(0)


def _decode_unit(px):
    """Normals were emitted as n * 0.5 + 0.5 (no negative emission)."""
    return px[..., :3].astype(np.float64) * 2.0 - 1.0


def _sample(px, uv):
    """Bilinear, repeat-wrapped lookup of ``px`` (h, w, 4, bottom-up) at
    ``uv`` (m, 2), matching Cycles' Linear image interpolation."""
    h, w = px.shape[:2]
    x = uv[:, 0] * w - 0.5
    y = uv[:, 1] * h - 0.5
    x0 = np.floor(x)
    y0 = np.floor(y)
    tx = (x - x0)[:, None]
    ty = (y - y0)[:, None]
    ix0 = x0.astype(np.int64) % w
    iy0 = y0.astype(np.int64) % h
    ix1 = (ix0 + 1) % w
    iy1 = (iy0 + 1) % h
    top = px[iy1, ix0] * (1 - tx) + px[iy1, ix1] * tx
    bot = px[iy0, ix0] * (1 - tx) + px[iy0, ix1] * tx
    return (bot * (1 - ty) + top * ty).astype(np.float32)


def _image_rgba(img, w, h):
    """Pixels of ``img`` as float32 (h, w, 4), bottom-up."""
    n = len(img.pixels)
    ch = n // (w * h)
    raw = np.empty(n, np.float32)
    img.pixels.foreach_get(raw)
    raw = raw.reshape(h, w, ch)
    if ch == 4:
        return raw
    px = np.ones((h, w, 4), np.float32)
    if ch < 3:
        px[..., :3] = raw[..., :1]
        if ch == 2:
            px[..., 3] = raw[..., 1]
    else:
        px[..., :3] = raw[..., :3]
    return px


def _slices(d, n):
    """(destination, source) slices shifting an axis of length n by d."""
    if d >= 0:
        return slice(d, n), slice(0, n - d)
    return slice(0, n + d), slice(-d, n)


_NEIGHBOURS = ((0, 1), (0, -1), (1, 0), (-1, 0),
               (1, 1), (1, -1), (-1, 1), (-1, -1))


def _margin_map(covered, radius):
    """(dst, src) flat texel indices: every uncovered texel within
    ``radius`` steps of the covered ones copies its nearest covered texel
    (grown ring by ring from the island borders, straight neighbours
    before diagonal ones), so the work follows the margin band, not the
    whole image."""
    h, w = covered.shape
    empty = np.empty(0, np.int64)
    if radius <= 0 or covered.all() or not covered.any():
        return empty, empty
    # Border texels: covered with an uncovered neighbour.
    border = np.zeros_like(covered)
    for dy, dx in _NEIGHBOURS:
        ydst, ysrc = _slices(dy, h)
        xdst, xsrc = _slices(dx, w)
        border[ydst, xdst] |= ~covered[ysrc, xsrc]
    border &= covered
    near = np.full(h * w, -1, np.int32 if h * w < 2**31 else np.int64)
    flat = covered.ravel()
    near[flat] = np.flatnonzero(flat)
    front = np.flatnonzero(border.ravel())
    filled = []
    for _ in range(radius):
        if front.size == 0:
            break
        fy, fx = np.divmod(front, w)
        ring = []
        for dy, dx in _NEIGHBOURS:
            ny, nx = fy - dy, fx - dx
            ok = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
            n = ny[ok] * w + nx[ok]
            src = front[ok]
            todo = near[n] < 0
            n, src = n[todo], src[todo]
            if n.size == 0:
                continue
            n, first = np.unique(n, return_index=True)
            near[n] = near[src[first]]
            ring.append(n)
        front = np.concatenate(ring) if ring else empty
        filled.append(front)
    dst = np.concatenate(filled) if filled else empty
    return dst, near[dst].astype(np.int64)


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
        self._fixes = {}        # bake size -> normal-matching corrections
        self._lookups = {}      # bake size -> Resample UV lookup
        self._geo = None        # source BVH + per-triangle data, lazily
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
        self.src_uvmap = uv
        self.src_tex = nt.nodes.new("ShaderNodeTexImage")
        self.src_tex.interpolation = 'Linear'
        self.src_tex.extension = 'REPEAT'
        self.emit = nt.nodes.new("ShaderNodeEmission")
        self.emit.inputs["Strength"].default_value = 1.0
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(uv.outputs["UV"], self.src_tex.inputs["Vector"])
        nt.links.new(self.emit.outputs["Emission"], out.inputs["Surface"])
        # Face normal of the hit, for normal matching.
        geo = nt.nodes.new("ShaderNodeNewGeometry")
        self.src_nrm = self._encode(nt, geo.outputs["True Normal"],
                                    (0.5,) * 3, (0.5,) * 3)

        # Destination: the active image node Cycles bakes into.  Its own
        # emission (normal / position) is only used by the passes that bake
        # the destination by itself, for normal matching.
        self.dst_mat = _new_material(f"{_TMP}_target")
        self.mats.append(self.dst_mat)
        nt = self.dst_mat.node_tree
        self.dst_tex = nt.nodes.new("ShaderNodeTexImage")
        self.dst_emit = nt.nodes.new("ShaderNodeEmission")
        self.dst_emit.inputs["Strength"].default_value = 1.0
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(self.dst_emit.outputs["Emission"],
                     out.inputs["Surface"])
        self.dst_geo = nt.nodes.new("ShaderNodeNewGeometry")
        self.dst_nrm = self._encode(nt, self.dst_geo.outputs["Normal"],
                                    (0.5,) * 3, (0.5,) * 3)
        self.dst_tex.select = True
        nt.nodes.active = self.dst_tex

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

        # Source UV for Resample, emitted as (uv - lo) / span with B = 1
        # marking texels that hit something.
        me = self.src.data
        uvs = np.empty(len(me.loops) * 2, np.float64)
        me.uv_layers[self.src_uv].uv.foreach_get("vector", uvs)
        uvs = uvs.reshape(-1, 2)
        ulo, uhi = ((uvs.min(0), uvs.max(0)) if len(uvs)
                    else (np.zeros(2), np.ones(2)))
        self.uv_span = np.maximum(uhi - ulo, 1e-6)
        self.uv_lo = ulo
        self.src_uvenc = self._encode(
            self.src_mat.node_tree, self.src_uvmap.outputs["UV"],
            (1.0 / self.uv_span[0], 1.0 / self.uv_span[1], 0.0),
            (-ulo[0] / self.uv_span[0], -ulo[1] / self.uv_span[1], 1.0))

        # Destination position, emitted as (p - lo) / span in [0, 1].
        lo, hi = _world_bounds(self.dst)
        span = np.maximum(hi - lo, 1e-6) * 1.02
        self.pos_lo = lo - (span * 0.01)
        self.pos_span = span
        self.dst_pos = self._encode(
            self.dst_mat.node_tree, self.dst_geo.outputs["Position"],
            tuple(1.0 / span), tuple(-self.pos_lo / span))

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

    @staticmethod
    def _encode(nt, socket, scale, offset):
        """socket * scale + offset (keeps emitted values non-negative)."""
        vm = nt.nodes.new("ShaderNodeVectorMath")
        vm.operation = 'MULTIPLY_ADD'
        vm.inputs[1].default_value = scale
        vm.inputs[2].default_value = offset
        nt.links.new(socket, vm.inputs[0])
        return vm.outputs["Vector"]

    def select(self, with_source=True):
        """The temporary objects selected (destination only when baking it
        by itself), destination active (re-applied before every bake)."""
        vl = self.view_layer
        for o in vl.objects:
            if o.select_get():
                o.select_set(False)
        if with_source:
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
        """Move ``img`` (shown through the source UVs) into the destination
        UVs at w x h; returns (float32 (h, w, 4) bottom-up, supersampling)."""
        ss = supersample
        while ss > 1 and w * h * ss * ss > MAX_BAKE_PIXELS:
            ss //= 2
        if self.settings.mode == 'RESAMPLE':
            out = self._resample_image(img, w, h, ss)
        else:
            out = self._bake_image(img, w, h, ss, has_alpha)
        if not has_alpha:
            out[..., 3] = 1.0
        if ss > 1:      # box filter down to the output resolution
            bh, bw = out.shape[:2]
            out = out.reshape(bh // ss, ss, bw // ss, ss, 4).mean(
                axis=(1, 3), dtype=np.float64).astype(np.float32)
        return out, ss

    def _new_buffer(self, bw, bh):
        return bpy.data.images.new(f"{_TMP}_buffer", bw, bh, alpha=True,
                                   float_buffer=True, is_data=True)

    def _bake_image(self, img, w, h, ss, has_alpha):
        """Bake mode: Cycles bakes of the image itself (full size)."""
        buf = self._new_buffer(w * ss, h * ss)
        try:
            self.src_tex.image = img
            self.dst_tex.image = buf
            fix = self._normal_fix(buf, ss, self.settings.margin * ss)
            nt = self.src_mat.node_tree
            nt.links.new(self.src_tex.outputs["Color"],
                         self.emit.inputs["Color"])
            out = self._bake(buf, ss)
            if has_alpha:
                nt.links.new(self.src_tex.outputs["Alpha"],
                             self.emit.inputs["Color"])
                out[..., 3] = self._bake(buf, ss)[..., 0]
            if fix is not None:
                idx, uv, found = fix
                vals = np.zeros((idx.size, 4), np.float32)
                if found.any():
                    vals[found] = _sample(_image_rgba(img, w, h), uv[found])
                out.reshape(-1, 4)[idx] = vals   # no match: like a miss
            return out
        finally:
            self.src_tex.image = None
            self.dst_tex.image = None
            bpy.data.images.remove(buf)

    def _resample_image(self, img, w, h, ss):
        """Resample mode: the image sampled through the cached lookup."""
        bw, bh = w * ss, h * ss
        idx, uv, mdst, msrc = self._lookup(bw, bh, ss)
        out = np.zeros((bh * bw, 4), np.float32)   # misses: like a bake
        if idx.size:
            out[idx] = _sample(_image_rgba(img, w, h), uv)
        out[mdst] = out[msrc]
        return out.reshape(bh, bw, 4)

    def _lookup(self, bw, bh, ss):
        """(hit texels, their source UVs, margin dst, margin src) for a
        bw x bh destination buffer; baked once per size."""
        key = (bw, bh)
        if key in self._lookups:
            return self._lookups[key]
        buf = self._new_buffer(bw, bh)
        try:
            self.dst_tex.image = buf
            # Texels inside the destination islands (for the margin).
            self.dst_mat.node_tree.links.new(self.dst_nrm,
                                             self.dst_emit.inputs["Color"])
            cov = _decode_unit(self._bake(buf, ss, to_active=False,
                                          margin=0))
            covered = np.abs(np.linalg.norm(cov, axis=-1) - 1.0) < 0.25
            del cov
            # Source UV under every texel (no margin: UVs don't blend).
            self.src_mat.node_tree.links.new(self.src_uvenc,
                                             self.emit.inputs["Color"])
            px = self._bake(buf, ss, margin=0).reshape(-1, 4)
            idx = np.flatnonzero(px[:, 2] > 0.5)
            uv = px[idx, :2].astype(np.float64) * self.uv_span + self.uv_lo
            del px
            fix = self._normal_fix(buf, ss, 0)
            if fix is not None:
                fidx, fuv, found = fix
                pos = np.searchsorted(idx, fidx)
                ok = pos < idx.size
                ok[ok] = idx[pos[ok]] == fidx[ok]
                uv[pos[ok & found]] = fuv[ok & found]
                keep = np.ones(idx.size, bool)
                keep[pos[ok & ~found]] = False   # no match: a miss
                idx, uv = idx[keep], uv[keep]
            mdst, msrc = _margin_map(covered,
                                     self.settings.margin * ss)
        finally:
            self.dst_tex.image = None
            bpy.data.images.remove(buf)
        print(f"Wrapper's Delight: resample lookup {bw}x{bh}: "
              f"{idx.size} texels, {mdst.size} margin")
        self._lookups[key] = (idx, uv, mdst, msrc)
        return self._lookups[key]

    def _bake(self, buf, ss, to_active=True, margin=None):
        """One Emit bake into ``buf`` at its full size: from the source
        (selected to active), or the destination by itself."""
        b = self.settings
        if margin is None:
            margin = b.margin * ss
        self.select(with_source=to_active)
        bpy.ops.object.bake(
            type='EMIT',
            target='IMAGE_TEXTURES',
            save_mode='INTERNAL',
            use_clear=True,
            use_selected_to_active=to_active,
            use_cage=False,
            cage_extrusion=self.extrusion,
            max_ray_distance=self.extrusion * 2.0,
            margin=margin,
            margin_type='ADJACENT_FACES',
            uv_layer=self.dst_uv,
        )
        bw, bh = buf.size
        px = np.empty(bw * bh * 4, np.float32)
        buf.pixels.foreach_get(px)
        return px.reshape(bh, bw, 4)

    # -- normal matching -------------------------------------------------

    def _normal_fix(self, buf, ss, margin):
        """(texel indices, source UVs, found) for the texels of a ``buf``
        sized bake whose hit faces too far from the destination normal;
        None when there are none or matching is off.  Cached per size."""
        limit = self.settings.normal_angle
        if limit >= math.pi - 1e-4:
            return None
        key = (tuple(buf.size), margin)
        if key in self._fixes:
            return self._fixes[key]
        cos_lim = math.cos(limit)

        self.dst_mat.node_tree.links.new(self.dst_nrm,
                                         self.dst_emit.inputs["Color"])
        nd = _decode_unit(self._bake(buf, ss, to_active=False,
                                     margin=margin))
        nd = nd.reshape(-1, 3)
        self.src_mat.node_tree.links.new(self.src_nrm,
                                         self.emit.inputs["Color"])
        ns = _decode_unit(self._bake(buf, ss, margin=margin))
        ns = ns.reshape(-1, 3)
        ld = np.linalg.norm(nd, axis=1)
        ls = np.linalg.norm(ns, axis=1)
        # Cleared (missed) texels decode to (-1, -1, -1): length sqrt(3).
        ok = (np.abs(ld - 1.0) < 0.25) & (np.abs(ls - 1.0) < 0.25)
        dot = (nd * ns).sum(axis=1) / np.maximum(ld * ls, 1e-12)
        idx = np.flatnonzero(ok & (dot < cos_lim))

        fix = None
        if idx.size:
            self.dst_mat.node_tree.links.new(self.dst_pos,
                                             self.dst_emit.inputs["Color"])
            pos = self._bake(buf, ss, to_active=False, margin=margin)
            pos = pos.reshape(-1, 4)[idx, :3]
            P = pos.astype(np.float64) * self.pos_span + self.pos_lo
            N = nd[idx] / ld[idx, None]
            uv, found = self._search(P, N, cos_lim)
            print(f"Wrapper's Delight: normal matching re-searched "
                  f"{idx.size} texels, {int(found.sum())} matched")
            fix = (idx, uv, found)
        self._fixes[key] = fix
        return fix

    def _source_geometry(self):
        if self._geo is None:
            me = self.src.data
            v, t = triangles_local(me)
            M = np.array(self.src.matrix_world, np.float64)
            vw = v @ M[:3, :3].T + M[:3, 3]
            loops = np.empty(len(me.loop_triangles) * 3, np.int64)
            me.loop_triangles.foreach_get("loops", loops)
            uvl = me.uv_layers[self.src_uv]
            uv = np.empty(len(me.loops) * 2, np.float64)
            uvl.uv.foreach_get("vector", uv)
            tri_uv = uv.reshape(-1, 2)[loops.reshape(-1, 3)]
            a, b, c = vw[t[:, 0]], vw[t[:, 1]], vw[t[:, 2]]
            n = np.cross(b - a, c - a)
            n /= np.maximum(np.linalg.norm(n, axis=1), 1e-30)[:, None]
            if np.linalg.det(M[:3, :3]) < 0:     # mirrored: winding flips
                n = -n
            self._geo = (tree_from_arrays(vw, t), vw, t, n, tri_uv)
        return self._geo

    def _search(self, P, N, cos_lim):
        """Blender's bake ray (from P + N * extrusion along -N, up to
        2 * extrusion) but passing through faces whose normal is more than
        the limit off N.  Returns (source UVs (m, 2), found (m,))."""
        from mathutils import Vector
        tree, vw, t, tri_n, tri_uv = self._source_geometry()
        ext = self.extrusion
        reach = ext * 2.0
        eps = ext * 1e-4
        m = len(P)
        tri = np.full(m, -1, np.int64)
        loc = np.zeros((m, 3))
        for i in range(m):
            n = N[i]
            d = Vector((-n).tolist())
            o = Vector((P[i] + n * ext).tolist())
            left = reach
            for _ in range(64):
                hit, _nor, ti, dist = tree.ray_cast(o, d, left)
                if ti is None:
                    break
                if tri_n[ti] @ n >= cos_lim:
                    tri[i] = ti
                    loc[i] = hit
                    break
                left -= dist + eps
                if left <= 0.0:
                    break
                o = hit + d * eps
        found = tri >= 0
        uv = np.zeros((m, 2))
        if found.any():
            ti = tri[found]
            a, b, c = (vw[t[ti, k]] for k in range(3))
            v0, v1, v2 = b - a, c - a, loc[found] - a
            d00 = (v0 * v0).sum(1)
            d01 = (v0 * v1).sum(1)
            d11 = (v1 * v1).sum(1)
            d20 = (v2 * v0).sum(1)
            d21 = (v2 * v1).sum(1)
            den = d00 * d11 - d01 * d01
            den = np.where(np.abs(den) > 1e-30, den, 1e-30)
            wb = (d11 * d20 - d01 * d21) / den
            wc = (d00 * d21 - d01 * d20) / den
            wa = 1.0 - wb - wc
            uv[found] = (wa[:, None] * tri_uv[ti, 0] + wb[:, None]
                         * tri_uv[ti, 1] + wc[:, None] * tri_uv[ti, 2])
        return uv, found

    # -- teardown --------------------------------------------------------

    def close(self):
        self.state.restore()
        self._fixes.clear()
        self._lookups.clear()
        self._geo = None
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
