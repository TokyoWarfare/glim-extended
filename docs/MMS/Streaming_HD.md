# GLIM MMS Extensions — Data Streaming, HD Support & Multi-Session

## Overview

These extensions transform GLIM's offline viewer from a simple map viewer into a professional MMS (Mobile Mapping System) post-processing suite. The key additions are:

- **LOD memory streaming** — distance-based GPU memory management for datasets of any size
- **HD frame capture and viewing** — full-resolution sensor data preserved and viewable
- **Multi-session alignment** — load, align, and export multiple SLAM runs
- **Tiled export** — UTM, JGD2011, and mesh-code tile naming conventions
- **Session management** — per-session visibility, export control, and clean unloading

---

## LOD Memory Streaming

### Problem
Large SLAM datasets (15+ GB, 40+ km highway mapping) exhaust GPU VRAM when all submaps are loaded simultaneously, even on high-end GPUs (RTX 3090, 24 GB VRAM).

### Solution
Distance-based Level of Detail (LOD) system with async loading:

| LOD Level | Content | VRAM Cost | When |
|-----------|---------|-----------|------|
| UNLOADED | Nothing | 0 | Outside frustum + far |
| BBOX | Wire cube bounding box | ~0 | Beyond SD range |
| LOADING | Preparing on background thread | 0 | Transitioning to SD |
| SD | Full submap point cloud | 2-10 MB/submap | Within SD range |
| LOADING_HD | Loading HD from disk | 0 | Transitioning to HD |
| HD | Full-resolution frame data | 20-100+ MB/submap | Within HD range (opt-in) |

### Key Features
- **Async loading**: Background worker thread prepares CPU data, `invoke()` uploads to GPU on render thread — zero render stutter
- **Frustum culling**: 6-plane AABB test per submap, submaps outside view are unloaded
- **Distance priority**: Closest submaps loaded first when multiple are queued
- **VRAM budget**: Configurable limit (1-24 GB), promotions blocked when exceeded, enforced in GPU upload callback
- **Hysteresis**: 20% distance band prevents flickering at LOD boundaries
- **Initial state**: All submaps start at BBOX — prevents OOM on load

### Memory Manager UI (Tools menu)
- Enable LOD streaming checkbox
- Enable HD (LOD 0) checkbox — greyed when no HD frames available
- SD range slider (50-2000 m, default 300 m)
- VRAM budget slider (1-24 GB, default 4 GB)
- Show bounding boxes toggle
- Live stats: VRAM usage bar, submap counts per LOD, SD/HD point counts
- **Load full SD map** / **Load full HD map** / **Unload all** buttons
- Settings persist to `~/.glim/memory_settings.json`

---

## HD Frame Support

### HD Frame Saver (`hd_frame_saver.so`)

Extension module that captures full-resolution sensor data during live SLAM.

**Setup**: Add `"libhd_frame_saver.so"` to `extension_modules` in `config_ros.json`. Requires `keep_raw_points: true` in `config_ros.json` — module warns with a popup if not set.

**Per-frame output** (`<dump>/hd_frames/<frame_id>/`):

| File | Type | Content |
|------|------|---------|
| `points.bin` | float32×3 | Deskewed xyz in sensor-local frame |
| `normals.bin` | float32×3 | Viewpoint-oriented normals (sensor at origin) |
| `intensities.bin` | float32 | Sensor intensity/reflectivity |
| `times.bin` | float32 | Per-point relative timestamps |
| `range.bin` | float32 | Distance to sensor |
| `rings.bin` | uint16 | Ring/channel ID |
| `frame_meta.json` | JSON | frame_id, stamp, num_points, T_world_lidar, bbox |

**Processing pipeline**:
1. Raw points from `raw_frame->raw_points` (full sensor resolution)
2. Deskewed using IMU-rate trajectory via `CloudDeskewing`
3. KdTree built, k-nearest neighbors found
4. Normals computed via `CloudCovarianceEstimation` with viewpoint orientation
5. All attributes saved as binary files + JSON metadata
6. Background thread for I/O — non-blocking to SLAM pipeline

**Why normals are computed at save time**: The viewpoint orientation check (`point.dot(normal) > 0 → flip`) requires the sensor at the coordinate origin. HD frames are in sensor-local frame, so this works. Recomputing normals later in world frame would lose the sensor position reference, causing random normal inversions.

### HD Viewer Loading

When HD frames are available:
- Detected automatically via `frame_meta.json` scan on map load
- Total HD point count computed without loading data
- HD LOD loads: points, normals, intensities, range, gps_time per frame
- Points transformed by optimized per-frame poses (not submap-level)
- Near-sensor noise filtered (range < 1.5 m — removes vehicle body/mount points)
- All color modes work: rainbow, session, normal, intensity, range, gps_time

### HD Export

**Export HD** checkbox in Save menu. Loads all HD frames from disk, transforms by optimized poses, applies coordinate system (UTM/JGD2011 + geoid), writes PLY with: double x/y/z, normals, intensity, range, gps_time (double), session_id. Supports single file and tiled export.

---

## Multi-Session Alignment

### Problem
Multiple SLAM runs of the same area use different UTM datum origins. Loading them into the same viewer shows them offset by the datum difference.

### Solution
Datum offset translation applied at load time:

1. First map's datum stored as **reference**
2. Each additional map: offset = (new_datum - reference_datum) applied to:
   - All submap `T_world_origin` poses
   - All GTSAM `Pose3` values in the factor graph
   - All `PoseTranslationPrior` (GNSS) factor targets — **translated, not stripped**
3. Export uses reference datum for coordinate transforms

### Why GNSS Factors Are Preserved
Stripping GNSS factors (original GLIM behavior for additional maps) leaves the second map as a free-floating chain. A manual loop closure at one point would deform the entire map — no GNSS anchors to resist. Translating the factors preserves geographic anchoring: corrections from loop closures fade naturally with distance.

### Session Manager (Sessions menu)
- Appears when >1 session loaded
- Per-session: visibility toggle, export-include toggle
- **Unload session**: full cleanup — nulls submaps, removes factors, removes menu entry, cleans all references. As if the session was never loaded.
- Reference session (id=0) protected from unloading
- Session-colored spheres and trajectory lines (6-color palette)
- UTM zone mismatch warning on cross-zone loading

### Per-Session HD Paths
Each session stores its own HD frames directory. Prevents cross-session frame ID collision (both sessions start from frame 0) which caused data corruption ("twirls") when a single global path was used.

---

## Tiled Export

### Coordinate Systems
- **UTM WGS84**: auto-detect zone, per-point zone correction option
- **JGD2011**: 19-zone Plane Rectangular CS, prefecture auto-detection from `japan_prefectures.geojson`

### Tile Presets (Coordinates → Tiles)

| Preset | Tile Size | Naming | Coord System |
|--------|-----------|--------|--------------|
| PNOA Spain (2022-2025) | 1×1 km | `PNOA_MMS_EEE_NNNN.ply` | UTM WGS84 |
| ICGC Cat (2021-2023) | 1×1 km | `EEENNN.ply` | UTM WGS84 |
| Japan (JGD2011) | 300m×400m | `ZZLLRRCC.ply` (図郭500) | JGD2011 |
| Default | Configurable | `TILE_EEEEEEE_NNNNNNN.ply` | Any |

### Export Attributes
- Double-precision x, y, z (absolute coordinates)
- Normals (float)
- Intensity (float, full sensor precision preserved end-to-end)
- Range (float)
- GPS time (double, full precision)
- Session ID (int, multi-session)
- UTM zone (int, when zone crossings detected)

---

## Configuration

### config_ros.json
```json
{
  "glim_ros": {
    "keep_raw_points": true,
    "extension_modules": ["libgnss_global.so", "libhd_frame_saver.so"]
  }
}
```

### config_hd_frame_saver.json
```json
{
  "hd_frames": {
    "output_path": "",
    "num_threads": 4,
    "k_neighbors": 10
  }
}
```

### Memory settings (~/.glim/memory_settings.json)
Auto-saved on exit, auto-loaded on startup. Contains LOD distances, VRAM budget, HD enable state.
