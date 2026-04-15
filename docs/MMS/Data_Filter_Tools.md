# Data Filter Tools

Post-processing tools for cleaning HD point cloud data in GLIM's offline viewer. All tools operate on the full-resolution deskewed HD frames (`hd_frames/` directory) using path-aligned spatial chunks along the SLAM trajectory.

Access via: **Tools > Data Filter**

## Pipeline Overview

The recommended processing order is:

1. **SOR** (Statistical Outlier Removal) — remove isolated noise points
2. **Classify ground to scalar** (in Dynamic mode) — generate ground labels
3. **Dynamic** (MapCleaner) — remove moving objects (vehicles, pedestrians)
4. **Range** — remove redundant long-range noise where closer observations exist

Each filter is destructive to the HD frames (rewrites bin files in-place). Use **Tools > Utils > Backup HD frames** before applying.

---

## SOR (Statistical Outlier Removal)

Removes isolated outlier points that have too few neighbors within a search radius.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Search radius (m) | 0.3 | Radius for neighbor search |
| Min neighbors | 5 | Points with fewer neighbors are removed |
| Chunk size (m) | 100 | Spatial processing cube size |

### Usage

1. Set parameters
2. **Process chunk** to preview one chunk at camera position
3. **Apply SOR to HD frames** to process full dataset

---

## Dynamic (MapCleaner)

Removes dynamic objects (vehicles, pedestrians, etc.) using multi-viewpoint range image voting. Each point in the merged map is projected into individual scan frames' range images. If a scan "sees through" a point (scan range < point range), it votes the point as dynamic.

### Ground Exclusion

Ground points are protected from dynamic removal. The pipeline:

1. **PatchWork++** classifies ground per-frame using concentric zone plane fitting
2. **Frame accumulation** (optional) merges N neighboring frames in sensor-local coords before PatchWork++, giving denser input for better classification
3. **Z-column refinement** (optional) revokes false ground labels for points above the lowest Z in each XY column (cross-frame, chunk-based)
4. **Intensity refinement** revokes high-intensity ground (reflective signs/plates)

Ground classification can be saved as a scalar field (`aux_ground.bin`) via **Classify ground to scalar**, then reused in subsequent dynamic filter runs.

### Trail Refinement

After MapCleaner voting, candidates are clustered using BFS on a voxel grid. Clusters are evaluated for trail-like shape (minimum length, aspect ratio, density). Non-trail clusters are rejected as false positives. Gap-fill extends confirmed trails to catch missed points.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Voxel size (m) | 0.64 | Spatial grid for point grouping |
| Range threshold (m) | 0.8 | How close measured vs expected range must match for STATIC vote |
| Observation range (m) | 30 | Max distance from sensor to vote on a point |
| Min observations | 15 | Minimum frame votes before classifying |
| Chunk size (m) | 120 | Processing chunk along trajectory |
| Chunk spacing (m) | 60 | Distance between chunk centers (50% overlap) |

#### PatchWork++ Config

| Parameter | Default | Description |
|-----------|---------|-------------|
| Sensor height (m) | 1.723 | LiDAR height above ground |
| Max range (m) | 80 | Maximum range for classification |
| Seed threshold | 0.125 | Initial seed selection threshold |
| Ground thickness (m) | 0.125 | Max distance from plane to count as ground |
| Uprightness thr | 0.707 | Surface flatness requirement (0.707 = 45 deg, higher = stricter) |
| Frame accumulation | off | Merge neighboring frames for denser PatchWork++ input |
| Prior/next frames | 10 | Number of neighbors to accumulate |

#### Trail Refinement Config

| Parameter | Default | Description |
|-----------|---------|-------------|
| Min length (m) | 7.0 | Minimum cluster extent along longest axis |
| Min aspect ratio | 5.0 | Longest / shortest axis ratio |
| Min density | 11.0 | Points per occupied voxel volume |
| Refine voxel (m) | 0.23 | Voxel size for clustering |

### Usage

1. Configure PatchWork++ params, preview with **Preview chunk** (ground mode)
2. Run **Classify ground to scalar** to save ground labels
3. Configure MapCleaner params
4. **Process chunk** to preview dynamic detection at camera position
5. **Apply dynamic filter to HD** — popup asks whether to reuse existing ground classification

---

## Range

Removes redundant long-range noise in areas where closer observations exist. Works per-voxel: if a voxel has both close-range and far-range points, far points beyond a delta threshold are removed.

### Affect Only Ground

When enabled (requires `aux_ground.bin`), only ground-classified points enter the range discrimination. Non-ground points pass through untouched. Useful for aggressive road cleanup (safe_range=10-11m) without destroying walls, vegetation, or overhead structures.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Criteria | Range | Range or GPS Time discrimination |
| Voxel size (m) | 1.0 | Spatial grid cell size |
| Safe range (m) | 20 | Points within this range always kept |
| Range delta (m) | 10 | Remove points >delta further than closest safe-range point |
| Far delta (m) | 30 | For voxels without safe-range points: remove if > min_range + far_delta |
| Min close points | 3 | Minimum safe-range points to trigger primary delta |
| Chunk size (m) | 60 | Processing chunk size |
| Chunk spacing (m) | 30 | Chunk center spacing |

### GPS Time Mode

Per-voxel temporal clustering: groups points by GPS time, keeps the dominant cluster, removes overlapping pass noise.

### Usage

1. Set parameters (optionally enable "Affect only ground")
2. **Process chunk** to preview at camera position
3. **Apply to HD frames (chunked)** to process full dataset

---

## Common Controls

- **Display chunks** — visualize processing chunks as wireframe boxes along trajectory
- **Clear preview** — remove preview overlay and restore normal view
- **Filter preview** — hide removed (red) points to see cleaned result
- **Toggle intensity** — switch preview between flat color and intensity colormap
- **Range highlight** — slider to tint points above a range threshold (requires preview data)
- **Reset defaults** — restore current mode's parameters to defaults

---

## Acknowledgements

### PatchWork++

Ground segmentation uses [PatchWork++](https://github.com/url-kaist/patchwork-plusplus) by Hyungtae Lim et al. (IROS 2022). Concentric zone model with region-wise vertical plane fitting and temporal ground revert. Licensed under BSD-2-Clause.

> H. Lim, et al., "Patchwork++: Fast and Robust Ground Segmentation Solving Partial Under-Segmentation Using 3D Point Cloud," IROS 2022.

The integration creates a fresh PatchWork++ instance per frame call (avoiding accumulated state corruption) and supports frame accumulation — merging neighboring frames in sensor-local coordinates to synthesize denser input coverage, particularly beneficial for non-repetitive scan patterns (e.g., Livox Horizon).

### MapCleaner

Dynamic object detection is inspired by the MapCleaner approach described in:

> H. Fu, et al., "MapCleaner: Efficiently Removing Moving Objects from Point Cloud Maps in Autonomous Driving Scenarios," IEEE RA-L, 2024.

This is a **clean-room reimplementation**, not a fork of any existing code. Key differences from the original:

- **Range image voting**: Uses closest-range storage (not max-range) in spherical projection with configurable resolution
- **Ground handling**: Ground points excluded entirely from the voting cloud via PatchWork++ classification (original used height-based filtering)
- **Cross-frame Z refinement**: Post-classification ground label correction using multi-frame XY column analysis
- **Trail clustering**: BFS-based spatial clustering with shape criteria (length, aspect ratio, density) to validate dynamic candidates and reject isolated false positives
- **Chunk-based processing**: Path-aligned spatial chunks with overlap for boundary handling, core-area-only result writing
- **Frame skip**: Adaptive frame decimation to maintain performance with large frame counts while preserving local observation density
