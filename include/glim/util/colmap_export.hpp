#pragma once

#include <cmath>
#include <string>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <glim/util/lidar_colorizer.hpp>

namespace glim {

/// 2D (top-view) bounds used for trimming, rotatable around world Z.
/// Stored as centre + half-extents + yaw so that OBB tests are cheap.
struct ExportBounds2D {
  float x_min, x_max;
  float y_min, y_max;
  float z_min, z_max;    // z range informational only -- not used for trimming
  float yaw_deg = 0.0f;  // rotation around world Z, CCW positive
  bool contains_xy(const Eigen::Vector3f& p) const {
    const float cx = 0.5f * (x_min + x_max);
    const float cy = 0.5f * (y_min + y_max);
    const float th = -yaw_deg * static_cast<float>(M_PI) / 180.0f;  // world -> local = rotate by -yaw
    const float c = std::cos(th), s = std::sin(th);
    const float dx = p.x() - cx, dy = p.y() - cy;
    const float lx = c * dx - s * dy;
    const float ly = s * dx + c * dy;
    return lx >= (x_min - cx) && lx <= (x_max - cx) &&
           ly >= (y_min - cy) && ly <= (y_max - cy);
  }
};

/// A single LiDAR point (world frame, NOT yet re-origined). Optional fields
/// land in the PLY only when their `has_*` flag is set on the parent
/// PointSource -- so callers that only have XYZ+RGB don't pay for normals
/// and the writer doesn't emit empty columns.
struct ColoredPoint {
  Eigen::Vector3f xyz;
  Eigen::Vector3f rgb     = Eigen::Vector3f(0.5f, 0.5f, 0.5f);  // [0,1]
  Eigen::Vector3f normal  = Eigen::Vector3f::Zero();             // unit; (0,0,0) if missing
  float           intensity = 0.0f;                              // raw scanner intensity
};

/// A named point cloud emission. The exporter writes ONE PLY per source
/// (and an optional .txt sidecar for COLMAP convention) so HD and Voxelized
/// stay separate -- that's the merge-bug fix. Field flags pick which
/// per-point columns make it into the PLY: callers can ship XYZ+RGB only,
/// or XYZ+RGB+intensity+normal, etc., and the writer adapts the header.
struct PointSource {
  std::string                filename_stem;   // e.g. "points3D_hd" or "points3D_voxelized" or "points3D"
  std::vector<ColoredPoint>  points;          // world frame; trimmed by caller
  bool emit_color     = true;
  bool emit_intensity = true;
  bool emit_normal    = true;
  // When true, also write a parallel .txt with COLMAP's points3D.txt format
  // (POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]). Some SFM tools (Reality
  // Scan) check for both .ply and .txt; emitting both is safe + cheap.
  bool emit_text_sidecar = false;
};

/// A camera frame to export: pose + image path + source index (for cameras.txt).
/// Pose is T_world_cam (our convention: +X fwd, +Y left, +Z up).
struct ExportCameraFrame {
  std::string source_image_path;   // absolute path to original .jpg/.png
  std::string source_mask_path;    // absolute path to mask, or empty
  int source_idx;                  // maps to which ImageSource (its intrinsics)
  std::string export_name;         // filename to write, e.g. "src0_00123.jpg"
  Eigen::Isometry3d T_world_cam;   // in our frame convention
  // UTC seconds-since-epoch from the camera's RMC/GNSS timestamp + the
  // source's effective time_shift. Optional -- 0.0 means "no time" and the
  // EXIF write will skip DateTimeOriginal/SubSecTimeOriginal for this image.
  double timestamp = 0.0;
};

struct ExportOptions {
  std::string output_dir;
  bool copy_images = true;         // false = symlink
  bool binary_format = false;      // false = .txt, true = .bin (COLMAP binary)
  // WARNING: leave false for multi-tile datasets. When true, points+cameras
  // are shifted so tile center is at origin -- different tiles get DIFFERENT
  // shifts and are no longer fusible with a single constant offset (kills
  // aerial-LiDAR fusion, cross-tile stitching). Only set true for single-tile
  // 3DGS runs where small coords are helpful for float precision.
  bool re_origin = false;
  float overlap_margin_m = 3.0f;   // cameras outside bounds but within this distance are still included
  float voxel_size_m = 0.03f;      // recorded in manifest; not applied here (caller pre-voxelizes)
  // Rotate the exported world from our convention (X fwd, Y left, Z up) to a
  // 3DGS-style right-handed Y-up frame (X right, Y up, Z back). LichtFeld /
  // gsplat / nerfstudio viewers orbit around Y-up by default -- without this,
  // the scene loads rotated 90 deg. Training works regardless (axis convention
  // doesn't affect projection math), this is purely for viewer ergonomics.
  bool rotate_to_y_up = true;
  // If true, images are undistorted on export (cv::remap) and cameras.txt
  // uses PINHOLE model. If false, images are copied raw and cameras.txt uses
  // OPENCV model with the original k1/k2/p1/p2 lens parameters -- pick this
  // for downstream tools that prefer to handle distortion themselves (Bundler
  // exports, or if the 3DGS stack undistorts internally). When undistorting,
  // symlinks are not valid (would point to distorted source), so copy_images
  // is effectively forced to true.
  bool export_undistorted = true;
  // Output folder structure -- each target produces a self-contained layout
  // tailored to its consumer. No cross-target compromise: the writer routes
  // image files, mask files, and bundle metadata to the paths each tool
  // expects natively. UI exposes a single radio that selects target +
  // adjusts emit_colmap implicitly.
  // SFM target = a self-contained, complete export specification. Each
  // target dictates folder structure, mask naming convention, AND which
  // bundle files (cameras.txt / blocks_exchange.xml) get emitted. Pick
  // ONE target per export -- targets are mutually exclusive at the UI
  // level. Internally `emit_colmap` and `export_blocks_exchange` are
  // derived from this enum; the caller still sets them on ExportOptions
  // (so the writer keeps a single source of truth) but the UI does not
  // expose them as separate toggles any more.
  enum class TargetLayout {
    Colmap          = 0,  // cameras.txt + images.txt + sparse/0/points3D.{ply,txt} +
                          //   images/ + masks/<stem>.png
    RealityScan     = 1,  // .geometry/ + .mask/<stem>.png + point_cloud/
                          //   (RS auto-detects layers by separator-prefixed folder names)
    Metashape       = 2,  // images/ + masks/<stem>_mask.png + point_cloud/
                          //   (user points Metashape at masks/ via Import Masks)
    BlocksExchange  = 3,  // blocks_exchange.xml + images/ + masks/<stem>.png + point_cloud/
                          //   (BE references images by path; structure is generic)
  };
  TargetLayout target_layout = TargetLayout::Colmap;

  // Bundle emission flags -- derived from target_layout at the call site
  // (UI has only the 4-way radio). Kept as fields so the writer dispatches
  // off them without a redundant switch. The caller is responsible for
  // setting both consistently with target_layout; setting them out of band
  // is a power-user override path and intentionally not policed here.
  bool emit_colmap            = true;   // cameras.txt + images.txt + COLMAP text files
  bool export_blocks_exchange = false;  // blocks_exchange.xml at <out>/

  // ---- Camera-pose-prior carriers (orthogonal to target_layout) ----
  // Each metadata channel is independently togglable so the user can mix:
  //   * Reality Scan target with EXIF on (test what RS does with EXIF GPS)
  //   * Metashape target with XMP on (full 3D rotation alongside EXIF)
  //   * Any target with both off (no per-image metadata; cameras come
  //     from cameras.txt / blocks_exchange.xml only)
  // UI sets sensible defaults per target_layout but the user has the final
  // say. Both writes silently skip when their preconditions miss
  // (datum_available for EXIF; image-format-supported for libexiv2).
  bool   datum_available             = false;
  int    datum_utm_zone              = 0;
  double datum_utm_easting_origin    = 0.0;
  double datum_utm_northing_origin   = 0.0;
  double datum_alt                   = 0.0;
  // Below-equator zones use southern false-northing (10 000 000 offset);
  // ecef_to_utm_zone returns the zone but doesn't carry hemisphere -- the
  // datum's lat tells us. Cached at call-site population time.
  bool   datum_southern_hemisphere   = false;

  // Metadata channels. Each can be toggled independently of target_layout.
  // EXIF GPS embed needs libexiv2 + datum_available. XMP sidecar is plain
  // XML write (no library). Both fire from write_geo_priors_for_camera()
  // at image-copy time.
  bool   emit_exif_gps               = false;  // GPS lat/lon/alt + heading + DateTime in image
  bool   emit_xmp_sidecar            = false;  // <basename>.xmp next to image (RS xcr: schema)

  // Full UTM mode -- when on AND datum_available, every exported world
  // coordinate (PLY points, COLMAP / BE camera positions, XMP positions)
  // gets the session datum's UTM origin added in. Result: PLY + cameras
  // land in real-world UTM eastings/northings, matching EXIF GPS lat/lon
  // (which is always WGS84 absolute). Required for Metashape / BE workflows
  // where camera GPS must align with the LiDAR cloud.
  // When off (default for COLMAP/3DGS), the session-local frame is preserved
  // -- coordinates are origin-shifted to the datum, smaller numbers, better
  // float precision for 3DGS rendering.
  // Mutually exclusive in spirit with re_origin (re_origin shifts to TILE
  // center, full_utm shifts to ABSOLUTE UTM); the writer applies both
  // independently though, so a power user can combine them if they really
  // know what they're doing.
  bool   full_utm                    = false;

  // Single shared mask per source (instead of per-image). Useful for
  // Metashape / BE workflows where the same rig-hardware mask covers all
  // cameras of a given calibration group -- Metashape's Import Masks dialog
  // can apply one file to "all cameras of the same camera group". Saves
  // disk + clutter when masks are static (no YOLO / per-frame segmentation).
  // When on:
  //   * Each source's mask_path (if set) is copied once to
  //     "<mask_subdir>/srcN_mask.png" (per-source filename)
  //   * Per-image masks (the <stem>_mask.png / <stem>.png files normally
  //     written one-per-frame) are SKIPPED entirely
  // Virtual sources with per-image generated masks (RS XMP workflow) lose
  // their per-image masks under this toggle -- intended for static-mask
  // workflows only.
  bool   single_mask                 = false;

  // Flight log export. Writes a `flight_log.csv` at the output root with
  // one row per kept camera: `# name lon lat alt`. Reality Scan imports
  // it via Workflow > Trajectory (and flexibly maps the columns). User-
  // facing alternative to XMP / EXIF when those don't get auto-detected.
  // Skipped silently when datum_available=false.
  bool   emit_flight_log             = false;

  // Header-only points3D.txt for the RS-via-COLMAP SLAM workflow. When ON,
  // the COLMAP text sidecar is written with the standard header lines but
  // ZERO point rows -- the per-source PLY is shipped alongside as usual,
  // and the user feeds the cloud into RS via `Import LiDAR Scan` instead.
  // Why: the RS SLAM tutorial (XGrids workflow) explicitly says to wipe
  // the points from points3D.txt so RS doesn't trip on COLMAP's sparse-
  // point format -- an oversize / format-quirky points3D.txt surfaces in
  // RS as a "file not found" import error. Default OFF for normal COLMAP
  // / 3DGS exports (those want the points for sparse init); flipped on by
  // the UI when the user is targeting RS via the COLMAP path.
  bool   points3d_text_header_only   = false;
  // Per-photo pose-prior accuracies emitted in BlocksExchange so Metashape BA
  // stays close to the imported poses. OFF -> poses are pure initial estimates,
  // BA can drift scale/origin. ON -> BA treats poses as constrained priors,
  // nailing the block in the session's coordinate frame (key for multi-tile
  // / aerial-LiDAR fusion with a constant offset).
  bool   emit_pose_priors    = false;
  double pose_pos_sigma_m    = 0.05;   // position sigma (metres)
  double pose_rot_sigma_deg  = 2.0;    // rotation sigma (degrees)
};

struct ExportStats {
  size_t points_written = 0;
  size_t cameras_written = 0;
  size_t images_copied = 0;
  size_t masks_copied = 0;
  Eigen::Vector3d origin_offset = Eigen::Vector3d::Zero();
};

/// Write a COLMAP TEXT-format export (cameras.txt, images.txt + per-source
/// PLY/TXT for points + images/ folder) and/or a BlocksExchange XML.
///
/// Each entry in `point_sources` becomes one PLY file under sparse/0/,
/// named "<filename_stem>.ply"; with emit_text_sidecar = true you also
/// get "<filename_stem>.txt" in COLMAP's text format. This is how HD and
/// Voxelized stay separate (no more merged point cloud).
///
/// Cameras are filtered here to those whose position is inside `bounds`
/// (expanded by overlap_margin_m).
///
/// Axis convention: our (X fwd, Y left, Z up) → COLMAP (X right, Y down, Z fwd)
/// is handled inside. Caller passes poses in our convention.
ExportStats write_colmap_export(
  const ExportBounds2D& bounds,
  const std::vector<PointSource>& point_sources,        // one PLY per entry
  const std::vector<ExportCameraFrame>& cameras,        // world frame
  const std::vector<PinholeIntrinsics>& intrinsics_per_source,  // one per source_idx
  const std::vector<CameraType>& camera_type_per_source,         // one per source_idx; Pinhole or Spherical
  const std::vector<bool>& is_virtual_per_source,                 // one per source_idx; true = LiDAR-rendered anchor (tight BE <Accuracy>)
  const ExportOptions& options,
  std::string* error_msg = nullptr);

}  // namespace glim
