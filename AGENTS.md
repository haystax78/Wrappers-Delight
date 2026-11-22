# Wrapper's Delight Add-on

(Release zips package the add-on as `wrappers_delight/`; nothing depends on the
folder name. Identifiers stay `meshwrap.*` / `Scene.mesh_wrap` so existing files
and keymaps keep working. Build a release zip with
`git archive --format=zip --prefix=wrappers_delight/ -o wrappers_delight_X.Y.zip HEAD -- . ':!.gitignore' ':!AGENTS.md'`.)

## Overview
Blender add-on that fits a cage mesh onto a target surface from matching
curves drawn on both meshes. Result is stored in a `Wrapped` shape key.

## Architecture
- **model/** — scene data and pure helpers
  - `properties.py`: `Scene.mesh_wrap` (`WrapSettings`) with `pairs` (`CurvePair`:
    own control points per side, colour, sample count, closed, weight, lock). Cage
    points are cage-local on the Basis (rest) shape; target points target-local.
    `landmarks` is a read-only shim for files from older versions.
  - `pairs.py`: side helpers (`'CAGE'`/`'TARGET'`), status, creation, snapshots.
  - `sampling.py`: arc-length resampling, RDP simplify, direction matching (numpy).
  - `display.py`: cached surface-projected draw geometry per (pair, side).
  - `meshdata.py`: bulk mesh array reads (int attributes copied straight from
    memory with a layout check; RNA `foreach_get` takes seconds on dense scans).
  - `legacy.py`, `snapping.py`.
- **solver/** — numpy + mathutils only
  - `transforms.py`: weighted similarity fit, polyharmonic landmark warp.
  - `graph.py`: mesh adjacency, normals, confidence-weighted field diffusion.
  - `surface.py`: `SurfaceProbe` (zero-copy queries on similarity transforms,
    world BVH from arrays otherwise — never BMesh, targets can be millions of tris).
  - `conform.py`: staged conform: facing-filtered closest point (+ ray retry,
    robust outlier drop, neighbour fill), inner pull steps with umbrella
    smoothing of positions (stiffness annealed, tangential from stage 3 so
    rows stay evenly spread in creases), then exact snaps with tangential
    evening-out. Excluded verts ramp back in via `pipeline.exclude_ramp`.
    `CurveLocks`: per-pair Lock holds cage-surface points (triangle +
    barycentric) on their target points, re-applied after every step. The
    whole drawn curve is locked (`pipeline.LOCK_DENSITY` Catmull-Rom points
    per segment, same parameters on both halves), not just the samples.
  - `pipeline.py`: `WrapJob` (validates up front, `run()` generator per stage).
- **operators/** — sidebar ops; `fit.py` runs `WrapJob` modally (live preview, Esc cancels).
  `pairs.py` has `meshwrap.pair_pick` (eyedropper: click a curve in any 3D view).
  `templates.py`: save/import cage templates in `templates/<name>/`
  (`<name>.blend` = cage on its rest shape with materials, images packed;
  `curves.json` = cage halves + pair settings). Import makes it the cage and
  recreates the pairs cage-side only.
- **editor/** — the two-view curve editor
  - `stage.py`: new window or split view, local views, chrome hiding, camera link.
  - `session.py`: modal operator (`meshwrap.editor`), Draw/Edit tools, private undo.
  - Draw tool: drag = freehand, click = place points (live Catmull-Rom, drag
    placed points), then PREVIEW with a point-count dial (wheel/arrows/digits,
    also resamples the waiting half) before LMB confirms. One half ahead only.
  - `guide.py`: pairs waiting for a target curve (template import) are
    walked in list order: next one made active, cage view framed on its
    curve, target view on the matching spot (similarity from finished pairs,
    camera link before that); new target curves are oriented like the cage
    curve; with X symmetry a waiting mirror twin gets the mirrored curve.
  - `picking.py`, `scene_draw.py` (2D screen-space curves through the stored
    points, per-point raycast occlusion; geometry cached per view, GPU batches
    per half cached per view + style), `symmetry.py` (X=0 slice line, cached
    batch, depth-biased), `hud.py` (2D),
    `paint.py`, `theme.py`.
- `updater.py` — GitHub release check (background thread, once per session,
  pref `auto_update`, respects `bpy.app.online_access`) and in-place install
  (keeps `templates/`, refuses a git checkout). Worker threads only touch the
  `state` dict; a main-thread timer repaints.
- **ui/** — `panels.py` (sidebar: Mesh Wrap > Correspondences / Fit / Result), `overlay.py`.

## Verification
```bash
# Syntax check
find . -name "*.py" -exec python -m py_compile {} \;

# Registration + functional test in Blender (Windows path example)
"D:/blender-5.3.0-windows-x64/blender.exe" --background --factory-startup \
  --python-expr "import sys; sys.path.insert(0, '<addons dir>'); import mesh_wrap; mesh_wrap.register(); mesh_wrap.unregister(); print('OK')"
```
The editor needs a real window (it can't run with `--background`); drive it with a
`bpy.app.timers` script and `bpy.ops.screen.screenshot` to check the HUD.

## Gotchas
- Area positions are only final after the first redraw: `Stage.tidy()` fixes
  cage/target order and re-hides chrome on the editor's timer.
- Region visibility props (`show_region_header` …) crash Blender if set without
  a window in context — always wrap in `temp_override(window=..., area=...)`.
- Adding to a `CollectionProperty` can invalidate existing item references;
  copy what you need before `pairs.add()`.
