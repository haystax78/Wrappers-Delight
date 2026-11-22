# Wrapper's Delight 1.0

Wrap a clean cage mesh onto any scan or sculpt by drawing matching curves on both. Wrapper's Delight aligns the cage to your curves, warps it smoothly through them and conforms it to the target surface. The result goes into a `Wrapped` shape key, so the original shape is always one slider away.

## Highlights

- **Two-view curve editor.** The cage and the target open side by side, either in a new window or in a split of your current viewport. Draw a curve on one mesh and the matching curve on the other.
- **Templates with guided placement.** Save a cage once, with its textured material and its curves. On a new target, import the template and the editor walks you through every curve, framing each one for you.
- **Per-curve Lock.** Pin a feature line (lid rim, lip line, jaw) exactly where you drew it while the rest of the surface relaxes.
- **Fast on dense meshes.** Targets with millions of faces open in under a second, and the curves redraw smoothly while you orbit.

## Drawing curves

- **Drag** to draw freehand, or **click** to place points. Placed points are joined by a smooth curve, and you can drag any point to adjust it.
- **Pick the point count** before confirming: use the scroll wheel, the arrow keys, or type a number.
- **Click a mesh border** to turn it into a curve.
- **One curve ahead.** After drawing a curve on one mesh, you're asked for its match on the other before starting a new one. The view waiting for that match gets a blue border.
- **Mirror X.** Strokes snap to the centre line and can't cross it. Finishing a curve on one side creates the mirrored pair automatically. A red line shows the centre on both meshes.
- **Edit tool.** Slide points along the surface, Ctrl+click to insert a point, X to delete.
- **View tools:**
  - Link Views, with sync in world space or per mesh
  - F to focus the surface under the cursor
  - per-view Wire and Texture toggles
  - X-Ray for curves
- **Private undo and redo** inside the editor. Closing the editor adds a single step to Blender's undo history.

## Templates

- **Save Template** stores the cage in the add-on's `templates` folder:
  - the mesh on its rest shape
  - its material, with the texture packed in
  - the cage side of every curve, including names, colours, point counts, Influence and Lock
- **Import Template** adds that cage to the open scene, sets it as the cage and restores the textured material. Its curves wait for their matches on the target.
- **Guided placement:**
  - each waiting curve is framed in turn on both meshes, from the same angle;
  - arrowheads show which way the template curve runs;
  - the curve you're working on is in colour and the others are in grey;
  - progress shows as "5 of 46".
- **Direction and mirrors are handled for you.** A target curve drawn the opposite way is reversed to match. With Mirror X on, each finished curve also fills in its mirror.
- **Back** reopens the previous curve so you can redraw it.

## Fitting

- **Presets:** Quick, Standard, Precise or Custom. The fit shows a live preview as it runs, and Esc stops it.
- **Conform to Surface** pulls the cage onto the target. Spacing stays even across creases such as the fold above the eyelid.
- **Exclude Group** keeps a vertex group as warped instead of snapping it, for example inner lips, eye sockets or a mouth bag. Vertices between the group and the nearest curves ease smoothly into the full wrap.
- **Lock** (per pair, 0–1) holds the whole drawn curve on its target curve while the fit relaxes. 1 keeps it exactly in place.
- **Influence** (per pair) sets how strongly a pair pulls the initial alignment.
- **Result panel:**
  - blend between Original and Wrapped;
  - bake the result to a new object;
  - remove the result.

## Sidebar

- Pick the cage and the target. Below that is a pair list with colour, status, enable and hide controls, and an **eyedropper** to select a pair by clicking its curve in the viewport.
- Changing a pair's **Samples** respaces both of its curves evenly, using the same resampling as the editor.
- Reverse, mirror, clear one side and recolour pairs.
- Correspondences are drawn in the regular viewport too, and can be hidden from the panel header.

## Updates

- **Update checker.** When Blender starts, the add-on looks for a newer release here on GitHub. If it finds one, a notice appears under the panel title: **Update available · click to install**. One click downloads and installs the update, keeping your saved templates. Restart Blender to finish.
- Turn it off with **Check for Updates** in the add-on preferences. The preferences also have a **Check for Updates** button to check right away.
- The checker only goes online when Blender's **Preferences › System › Allow Online Access** is on.

## Requirements

- Blender 4.2 or newer. Developed and tested on Blender 5.3.
- No extra Python packages.

## Installation

1. Download **wrappers_delight_1.0.zip** from this release's assets. It contains the `wrappers_delight` folder.
2. In Blender: **Edit › Preferences › Add-ons › Install from Disk…**, then choose the zip.
3. Enable **Wrapper's Delight**. The panel is in the 3D Viewport sidebar (**N**), under the **Wrapper's Delight** tab.

## Notes

- Files made with earlier development builds open as before: the scene data and operator names are unchanged. Older landmark data can be converted with **Convert Old Landmarks** in the sidebar.
- Saved templates can be large, because the texture is packed into the template's `.blend` file.
