# Post-Processing Tools

## ICP Loop Closure Modal

### Opening the Modal
- **Single mode**: Right-click sphere > "Loop begin" (target), then right-click another > "Loop end" (source). If target is set first, "Loop end" triggers ICP immediately.
- **Multi-source mode**: Click multiple "Loop end" first (accumulates sources in green), then "Loop begin" to merge all sources and open modal.
- **Auto-align teams**: Use the Source Finder tool to identify and align two opposing travel directions automatically.

### Data Source
- **SD** / **HD** buttons: switch between standard-density and high-density point clouds. HD loads instantly (covariances computed lazily on first registration).
- **Intensity** button: toggle between yellow/green flat colors and TURBO intensity colormap for visual alignment assessment.

### Manual Alignment
- **3D viewport**: drag to rotate, scroll to zoom. Gizmo handles for translate/rotate.
- **Helper gizmo**: spawns a secondary translate gizmo for fine positioning without occluding the main gizmo.
- **Ortho/Perspective**: toggle orthographic view (useful for top-down alignment).
- **Translate / Rotate sliders**: manual XYZ offset and Yaw/Pitch/Roll in degrees.

### Registration
- **Global registration**: FPFH feature-based alignment (RANSAC or GNC). Use when clouds are far apart.
  - `fpfh_radius`: neighbour search radius (~2.5m indoor, ~5m outdoor)
  - `4dof`: restrict to XYZ + yaw (ground vehicles)
- **GICP**: point-to-plane ICP using surface covariances. Primary fine registration method.
- **ICP**: point-to-point ICP. Simpler, for comparison.
- `max_corr_dist`: maximum correspondence distance for scan matching.
- `inf_scale`: information scale for the loop factor confidence.

### Factor Creation
- **Create Factor**: creates a BetweenFactor encoding the current transformation. For team alignment, creates factors for ALL submaps in both teams (not just the central pair).
- **Relax neighbours**: temporarily weaken nearby factors so the optimizer blends the correction smoothly.
  - `Radius`: number of submaps on each side to relax
  - `Scale`: sigma multiplier (5x = loose, 1x = no change)
  - `Between/GPS factors`: choose which constraint types to relax
- **Reset**: restore source cloud to initial position.

### GPS Quality
- GPS sigma displayed in the modal header (yellow = target, green = source).
- Helps decide which direction to trust more when sigma differs.

---

## Source Finder Tool

### Activation
Right-click anywhere > "Source finder". Places a probe at the clicked position (click point = base of cylinder/box).

### Probe Shape
- **Cylinder**: radial probe for single features (lamp posts, trees). Default radius 0.5m, height 5m.
- **Box**: oriented rectangle for linear features (row of trees, curbs). Length/width/yaw controls. Gizmo supports XYZ translate + Y/Z rotation.

### Scan Modes
- **Fast (bbox)**: uses submap bounding boxes for instant detection. Updates live as you drag the gizmo.
- **Precise (points)**: checks every point in every submap. Click "Scan" to run. More accurate, no false positives.

### Team Detection
Highlighted submaps are automatically grouped by sequence continuity (gap > 2 in submap ID = new team). Teams are displayed with role labels:
- **Target** (yellow spheres + yellow lines to probe center)
- **Source** (green spheres + green lines to probe center)
- Extra teams shown in grey (ignored for alignment)

### Actions
- **Swap teams**: switch target/source roles. Updates sphere colors instantly.
- **Auto-align teams**: merges each team's points, opens ICP modal with merged data. Creates cross-team factors for every submap in both teams.

### Identify Source
Right-click any point > "Identify source". Highlights the owning submap sphere in yellow with a connecting line. Useful for tracking down ghost/misaligned features.

---

## Undo Last Factor

Right-click any sphere > "Undo last factor (N)". Removes the last batch of loop closure factors from the ISAM2 graph and re-optimizes. Safe with or without relaxation active. Greyed out when nothing to undo.

---

## SD Regeneration from HD

**Tools > Utils > Regenerate SD from HD**

Rebuilds all submap SD (standard-density) point data from the current HD (high-density) frames. Use after range filtering or any HD modification.

- **Voxel size**: downsampling resolution (default 0.20m, adjustable 0.05-1.0m)
- Computes normals + covariances via CloudCovarianceEstimation
- Preserves aux attributes: intensity, range, gps_time
- Overwrites existing SD data on disk
- Updates in-memory submap frames immediately

---

## Range Filter Secondary Delta

**Far delta** parameter in the Range Filter tool.

In voxels where no safe-range (close) points exist, the far delta controls how much spread is allowed among distant points. Example: points at 50m, 60m, 150m with far_delta=30m — threshold = 50+30 = 80m — the 150m point is removed.

Works alongside the primary delta (which requires safe-range anchor points) to clean up noise in areas only covered by distant scans.

---

## Display Settings

**Tools > Display Settings**

- **Point size**: rendering size (0.001 - 0.5)
- **Point alpha**: opacity (0.05 - 1.0). Lower = more transparent, useful for seeing through dense areas.

---

## Workflow Example

1. Load map, enable HD LOD streaming
2. Navigate to a misaligned area (lamp post with double features)
3. Right-click > Source finder > place cylinder on the feature
4. Switch to Box mode for linear features, rotate to align with road
5. Review teams (yellow = target, green = source)
6. Swap teams if needed (consistent lane assignment)
7. Click "Auto-align teams"
8. In ICP modal: toggle Intensity to verify alignment quality
9. Run GICP fine registration
10. Enable "Relax neighbours" if GPS sigma is similar on both sides
11. Create Factor
12. If result looks wrong: right-click > "Undo last factor"
13. After all corrections: Tools > Utils > Regenerate SD from HD
