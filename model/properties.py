"""Scene data for Wrapper's Delight.

Everything lives on ``Scene.mesh_wrap`` (a :class:`WrapSettings`):

* the two meshes (``cage`` is deformed, ``target`` is the surface it fits),
* a list of :class:`CurvePair` correspondences, each holding its own control
  points on the cage and on the target,
* fit settings and editor preferences.

Control points are stored in each object's LOCAL space.  Cage points refer to
the cage's rest shape (the Basis shape key), so a wrap never invalidates them.
A pair's two halves may have different numbers of control points: matching
samples are generated along each half by arc length when the fit runs (see
:mod:`.sampling`), which keeps drawing and editing independent of how densely
a pair is sampled.
"""

import bpy

RESULT_KEY = "Wrapped"


def redraw_views(_self=None, _context=None):
    """Property update hook: repaint every 3D viewport."""
    wm = getattr(bpy.context, "window_manager", None)
    if wm is None:
        return
    for win in wm.windows:
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def _mesh_only(_self, obj):
    return obj.type == 'MESH'


def _cage_poll(self, obj):
    return obj.type == 'MESH' and obj != self.target


def _target_poll(self, obj):
    return obj.type == 'MESH' and obj != self.cage


def _meshes_changed(self, context):
    from ..editor import session
    session.request_stop()
    from . import display
    display.invalidate_all()
    redraw_views()


def _pair_changed(self, context):
    from . import display
    display.invalidate_all()
    redraw_views()


def _samples_changed(self, context):
    from . import pairs
    pairs.resample_pair(context, self, self.samples)
    redraw_views()


def _sync_space_changed(self, context):
    from ..editor import session
    session.sync_space_changed()


# --------------------------------------------------------------- points ---

class CurvePoint(bpy.types.PropertyGroup):
    co: bpy.props.FloatVectorProperty(
        name="Location", size=3, subtype='XYZ',
        description="Point location in the owning object's local space",
    )


class CurvePair(bpy.types.PropertyGroup):
    """One correspondence: a curve (or a single point) on each mesh."""

    uid: bpy.props.IntProperty(default=0)
    name: bpy.props.StringProperty(name="Name", default="Pair")
    color: bpy.props.FloatVectorProperty(
        name="Color", size=3, subtype='COLOR', min=0.0, max=1.0,
        default=(0.9, 0.6, 0.2), update=redraw_views,
        description="Display colour of this pair on both meshes",
    )
    closed: bpy.props.BoolProperty(
        name="Closed Loop", default=False, update=_pair_changed,
        description="Treat both halves as closed loops",
    )
    samples: bpy.props.IntProperty(
        name="Samples", default=8, min=2, max=256, update=_samples_changed,
        description="Number of points along this pair. Changing it "
                    "resamples both curves evenly along their shape",
    )
    weight: bpy.props.FloatProperty(
        name="Influence", default=1.0, min=0.0, max=10.0, soft_max=4.0,
        description="Relative influence of this pair on the initial alignment",
    )
    lock: bpy.props.FloatProperty(
        name="Lock", default=0.0, min=0.0, max=1.0, subtype='FACTOR',
        description="Hold this pair's cage curve on its target curve while "
                    "the surface fit relaxes. 0: free to slide, 1: stays "
                    "exactly where the target curve was placed",
    )
    hidden: bpy.props.BoolProperty(
        name="Hidden", default=False, update=_pair_changed,
        description="Hide this pair in the viewport (still used for fitting)",
    )
    enabled: bpy.props.BoolProperty(
        name="Enabled", default=True, update=redraw_views,
        description="Use this pair when fitting",
    )
    cage_points: bpy.props.CollectionProperty(type=CurvePoint)
    target_points: bpy.props.CollectionProperty(type=CurvePoint)


# ------------------------------------------------- legacy data (read-only) ---
# Earlier versions stored per-point records in ``Scene.mesh_wrap.landmarks``.
# Declaring the same property names here lets Blender hand that stored data
# back to us so it can be converted (see model.legacy); nothing writes to it.

class LegacyLandmark(bpy.types.PropertyGroup):
    src_co: bpy.props.FloatVectorProperty(size=3)
    tgt_co: bpy.props.FloatVectorProperty(size=3)
    has_src: bpy.props.BoolProperty()
    has_tgt: bpy.props.BoolProperty()
    curve_id: bpy.props.IntProperty(default=-1)
    curve_closed: bpy.props.BoolProperty()


# ------------------------------------------------------------- settings ---

FIT_PRESETS = [
    ('QUICK', "Quick", "Few iterations: fast preview of the fit"),
    ('STANDARD', "Standard", "Balanced speed and accuracy"),
    ('PRECISE', "Precise", "More iterations and finer detail; slower"),
    ('CUSTOM', "Custom", "Set every fitting parameter yourself"),
]


class WrapSettings(bpy.types.PropertyGroup):
    # --- meshes ---
    cage: bpy.props.PointerProperty(
        name="Cage", type=bpy.types.Object, poll=_cage_poll,
        update=_meshes_changed,
        description="Mesh that gets deformed onto the target",
    )
    target: bpy.props.PointerProperty(
        name="Target", type=bpy.types.Object, poll=_target_poll,
        update=_meshes_changed,
        description="Surface the cage is fitted to",
    )

    # --- correspondences ---
    pairs: bpy.props.CollectionProperty(type=CurvePair)
    pair_index: bpy.props.IntProperty(
        name="Active Pair", default=-1, update=redraw_views)
    uid_counter: bpy.props.IntProperty(default=0)
    default_samples: bpy.props.IntProperty(
        name="Default Points", default=5, min=3, max=64,
        description="Starting point count for a freehand curve (change it "
                    "before confirming with the wheel, arrows or by typing)",
    )
    show_in_viewport: bpy.props.BoolProperty(
        name="Show Pairs", default=True, update=redraw_views,
        description="Draw correspondences in the regular viewport",
    )

    # --- fit ---
    fit_preset: bpy.props.EnumProperty(
        name="Preset", items=FIT_PRESETS, default='STANDARD')
    fit_surface: bpy.props.BoolProperty(
        name="Conform to Surface", default=True,
        description="After the landmark warp, pull the cage onto the target "
                    "surface. Off: only the landmark-driven warp is applied",
    )
    lock_landmarks: bpy.props.BoolProperty(
        name="Lock Landmarks", default=False,
        description="Hold the cage vertices nearest to each matched sample "
                    "exactly on their target positions",
    )
    exclude_group: bpy.props.StringProperty(
        name="Exclude Group", default="",
        description="Cage vertex group left out of the surface fit. Its "
                    "vertices keep the shape the landmark warp gave them "
                    "instead of snapping onto the target (useful for inner "
                    "lips, eye sockets, mouth bags). Vertices between the "
                    "group and the nearest landmarks ease smoothly into the "
                    "full wrap",
    )
    reach: bpy.props.FloatProperty(
        name="Reach", default=0.12, min=0.005, max=1.0, subtype='FACTOR',
        description="Furthest the cage looks for the target surface, as a "
                    "fraction of the cage size",
    )
    warp_relax: bpy.props.FloatProperty(
        name="Warp Relax", default=0.0, min=0.0, max=1.0, subtype='FACTOR',
        description="Let the landmark warp miss landmarks slightly in "
                    "exchange for a smoother deformation (helps with noisy "
                    "or conflicting pairs)",
    )
    custom_iterations: bpy.props.IntProperty(
        name="Iterations", default=10, min=1, max=40,
        description="Number of correspondence refreshes",
    )
    custom_stiffness: bpy.props.FloatProperty(
        name="Stiffness", default=0.5, min=0.0, max=1.0, subtype='FACTOR',
        description="How strongly the cage resists bending while it is "
                    "pulled onto the surface",
    )
    custom_detail: bpy.props.FloatProperty(
        name="Detail", default=0.6, min=0.0, max=1.0, subtype='FACTOR',
        description="How closely the last iterations follow small surface "
                    "features",
    )
    custom_relax: bpy.props.IntProperty(
        name="Edge Relax", default=2, min=0, max=10,
        description="Tangential relaxation passes after the final snap, "
                    "evening out edge lengths along the surface",
    )

    # --- editor ---
    xray: bpy.props.BoolProperty(
        name="X-Ray Curves", default=True, update=redraw_views,
        description="Show curve segments hidden behind the mesh faintly",
    )
    symmetry: bpy.props.BoolProperty(
        name="Symmetry", default=False, update=redraw_views,
        description="X symmetry: strokes can't cross the X=0 plane, points "
                    "near it snap onto it, a completed pair gets a mirrored "
                    "twin, and moving a point moves its mirror image too",
    )
    symmetry_tolerance: bpy.props.FloatProperty(
        name="Symmetry Tolerance", default=0.004, min=0.0005, max=0.1,
        subtype='FACTOR', precision=4,
        description="Distance from the X=0 plane, as a fraction of the mesh "
                    "size, within which points count as centre-line: they "
                    "are snapped onto the plane and are their own mirror",
    )
    snap_vertices: bpy.props.BoolProperty(
        name="Snap to Vertices", default=False,
        description="Place and move points on mesh vertices",
    )
    link_views: bpy.props.BoolProperty(
        name="Link Views", default=True,
        description="Orbit and zoom the cage and target views together",
    )
    view_world_sync: bpy.props.BoolProperty(
        name="World-Space View Sync", default=True,
        update=_sync_space_changed,
        description="Curve editor: both views share one world-space camera "
                    "(same focus point, orbit and zoom). Off: each view is "
                    "centred on its own mesh and moves to the matching spot "
                    "on it",
    )

    # --- legacy ---
    landmarks: bpy.props.CollectionProperty(type=LegacyLandmark)


CLASSES = (CurvePoint, CurvePair, LegacyLandmark, WrapSettings)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.mesh_wrap = bpy.props.PointerProperty(type=WrapSettings)


def unregister():
    del bpy.types.Scene.mesh_wrap
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
