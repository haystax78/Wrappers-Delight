"""Run the wrap and manage its shape-key result.

The fit runs a few stages per timer tick so the viewport shows the cage
converging and Esc can stop it.  Results go to the ``Wrapped`` shape key;
the Basis key (the rest shape) is never modified.
"""

import time

import bpy
import numpy as np

from ..model.properties import RESULT_KEY, redraw_views
from ..solver.pipeline import WrapError, WrapJob

TICK_BUDGET = 0.06   # seconds of solver work per timer tick

_STATUS = {"running": False, "fraction": 0.0, "label": ""}


def status():
    return _STATUS


def result_key(cage):
    keys = getattr(getattr(cage, "data", None), "shape_keys", None)
    return keys.key_blocks.get(RESULT_KEY) if keys else None


def _ensure_result_key(cage):
    if cage.data.shape_keys is None:
        cage.shape_key_add(name="Basis", from_mix=False)
    key = result_key(cage)
    if key is None:
        key = cage.shape_key_add(name=RESULT_KEY, from_mix=False)
    key.slider_min, key.slider_max = 0.0, 1.0
    return key


def _read_key(key):
    co = np.empty(len(key.data) * 3)
    key.data.foreach_get("co", co)
    return co


class MESHWRAP_OT_fit(bpy.types.Operator):
    """Fit the cage onto the target using the matched pairs"""
    bl_idname = "meshwrap.fit"
    bl_label = "Wrap"
    bl_description = ("Align, warp and conform the cage onto the target. "
                      "The result is stored in the 'Wrapped' shape key")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        mw = context.scene.mesh_wrap
        return mw.cage is not None and mw.target is not None

    def _prepare(self, context):
        mw = context.scene.mesh_wrap
        try:
            self.job = WrapJob(context, mw)
        except WrapError as exc:
            self.report({'ERROR'}, str(exc))
            return False
        cage = mw.cage
        existed = result_key(cage) is not None
        self.key = _ensure_result_key(cage)
        self.backup = (existed, _read_key(self.key), self.key.value)
        self.key.value = 1.0
        cage.active_shape_key_index = cage.data.shape_keys.key_blocks.find(
            RESULT_KEY)
        self.steps = self.job.run()
        self.t0 = time.perf_counter()
        return True

    def _show(self, X):
        self.key.data.foreach_set(
            "co", np.ascontiguousarray(self.job.to_cage_local(X)).ravel())
        self.job.cage.data.update()

    def execute(self, context):
        if not self._prepare(context):
            return {'CANCELLED'}
        X = self.job.solve()
        self._show(X)
        return self._done(context)

    def invoke(self, context, event):
        if _STATUS["running"]:
            return {'CANCELLED'}
        if not self._prepare(context):
            return {'CANCELLED'}
        _STATUS.update(running=True, fraction=0.0, label="Starting")
        wm = context.window_manager
        self.timer = wm.event_timer_add(0.01, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'ESC' and event.value == 'PRESS':
            self._stop(context)
            self._restore()
            self.report({'INFO'}, "Wrap cancelled")
            return {'CANCELLED'}
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}
        t_end = time.perf_counter() + TICK_BUDGET
        latest = None
        try:
            while time.perf_counter() < t_end:
                frac, label, X = next(self.steps)
                _STATUS.update(fraction=frac, label=label)
                if X is not None:
                    latest = X
        except StopIteration:
            self._stop(context)
            self._show(self.job.result_world)
            return self._done(context)
        except Exception as exc:  # noqa: BLE001 - report and roll back
            self._stop(context)
            self._restore()
            self.report({'ERROR'}, f"Wrap failed: {exc}")
            return {'CANCELLED'}
        if latest is not None:
            self._show(latest)
        redraw_views()
        return {'RUNNING_MODAL'}

    def _stop(self, context):
        _STATUS.update(running=False, fraction=0.0, label="")
        timer = getattr(self, "timer", None)
        if timer is not None:
            context.window_manager.event_timer_remove(timer)
            self.timer = None
        redraw_views()

    def _restore(self):
        existed, co, value = self.backup
        cage = self.job.cage
        if existed:
            self.key.data.foreach_set("co", co)
            self.key.value = value
        else:
            cage.shape_key_remove(self.key)
        cage.data.update()

    def _done(self, context):
        dt = time.perf_counter() - self.t0
        n = len(self.job.src)
        extra = "".join(f", {note}" for note in self.job.notes)
        self.report({'INFO'}, f"Wrapped in {dt:.1f}s from {n} matched "
                              f"points{extra}")
        redraw_views()
        return {'FINISHED'}


class MESHWRAP_OT_result_bake(bpy.types.Operator):
    """Create a new object with the wrapped shape applied"""
    bl_idname = "meshwrap.result_bake"
    bl_label = "Bake to New Object"
    bl_description = ("Copy the cage as a new object whose mesh is the wrapped "
                      "shape, without shape keys")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return result_key(context.scene.mesh_wrap.cage) is not None

    def execute(self, context):
        cage = context.scene.mesh_wrap.cage
        co = _read_key(result_key(cage))
        obj = cage.copy()
        obj.data = cage.data.copy()
        obj.name = f"{cage.name}_wrapped"
        obj.data.name = obj.name
        obj.shape_key_clear()
        obj.data.vertices.foreach_set("co", co)
        obj.data.update()
        for coll in cage.users_collection:
            coll.objects.link(obj)
        for o in context.view_layer.objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        self.report({'INFO'}, f"Created '{obj.name}'")
        return {'FINISHED'}


class MESHWRAP_OT_result_clear(bpy.types.Operator):
    """Delete the Wrapped shape key"""
    bl_idname = "meshwrap.result_clear"
    bl_label = "Remove Result"
    bl_description = "Delete the 'Wrapped' shape key from the cage"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return result_key(context.scene.mesh_wrap.cage) is not None

    def execute(self, context):
        cage = context.scene.mesh_wrap.cage
        cage.shape_key_remove(result_key(cage))
        keys = cage.data.shape_keys
        if keys is not None and len(keys.key_blocks) == 1:
            cage.shape_key_clear()
        cage.data.update()
        return {'FINISHED'}


CLASSES = (MESHWRAP_OT_fit, MESHWRAP_OT_result_bake, MESHWRAP_OT_result_clear)
