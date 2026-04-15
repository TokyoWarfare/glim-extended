# Voxelize HD Data

World-space voxelization of HD point cloud frames for downstream applications (3D Gaussian Splatting initialization, visualization, analysis).

Access via: **Tools > Voxelize HD data**

## Overview

Converts the full-resolution HD frames into a regular voxel grid, producing one representative point per occupied voxel cell. Output is written to `hd_frames_voxelized/` (or `hd_frames_ground/` for ground-only mode) alongside the original HD data.

Processing uses path-aligned spatial chunks along the SLAM trajectory with overlap to handle chunk boundaries. Each output voxel point is assigned to a frame via round-robin distribution.

## Placement Modes

| Mode | Description | Best for |
|------|-------------|----------|
| Voxel center | Regular 3D grid, point at cell center | Uniform spacing |
| Weighted | Centroid of all points in cell | Smooth surfaces |
| XY grid + Z weighted | Regular XY grid, Z = weighted average | **3DGS initialization** (eliminates staircase artifacts on slopes while maintaining horizontal regularity) |

## Ground Only Mode

When **Ground only (1 pt/XY)** is enabled (requires `aux_ground.bin` from Dynamic filter's "Classify ground to scalar"):

- Only ground-classified points are included
- Uses XY-only voxel key — one point per horizontal cell regardless of Z
- Z position = weighted average of all ground points in the cell
- Produces a clean, noise-free ground surface
- Output written to `hd_frames_ground/`

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Voxel size (m) | 0.03 | Cell size. 0.01-0.03m for 3DGS, 0.05-0.10m for visualization |
| Placement | XY grid + Z weighted | Point placement within each voxel cell |
| Chunk size (m) | 60 | Processing chunk size along trajectory |
| Chunk spacing (m) | 30 | Distance between chunk centers |
| Ground only | off | Filter to ground points, one per XY column |

## Usage

1. Set voxel size and placement mode
2. **Preview** — voxelizes one chunk at camera position for quick visual check
3. **Intensity** — toggle intensity colormap on preview
4. **Clear** — remove preview overlay
5. **Apply to all HD** — process full dataset along trajectory

## Output Format

Each output frame in `hd_frames_voxelized/` contains:
- `points.bin` — world-space positions (Eigen::Vector3f)
- `range.bin` — averaged range values (float)
- `intensities.bin` — averaged intensities (float)
- `frame_meta.json` — point count, identity transform (points are world-space), bounding box

The LOD system can switch to voxelized data via the **Memory Manager > Use voxelized HD** checkbox.

## Performance

Processing uses a sliding window frame cache to minimize disk I/O across overlapping chunks. Typical throughput: ~6600 frames / ~170M voxels in ~275 seconds for a 5km dataset.

## Notes

- Original HD data is preserved — voxelization writes to a separate directory
- The identity `T_world_lidar` in output frame_meta.json means the LOD system uses identity transform when loading voxelized frames (points are already in world coordinates)
- Voxel size of 0.03-0.05m with XY+Z weighted placement works well as 3D Gaussian Splatting initialization — regular horizontal spacing eliminates training artifacts while smooth Z preserves surface detail
