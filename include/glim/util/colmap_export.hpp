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

/// A single colored LiDAR point (world frame, NOT yet re-origined).
struct ColoredPoint {
  Eigen::Vector3f xyz;
  Eigen::Vector3f rgb;  // [0,1]
};

/// A camera frame to export: pose + image path + source index (for cameras.txt).
/// Pose is T_world_cam (our convention: +X fwd, +Y left, +Z up).
struct ExportCameraFrame {
  std::string source_image_path;   // absolute path to original .jpg/.png
  std::string source_mask_path;    // absolute path to mask, or empty
  int source_idx;                  // maps to which ImageSource (its intrinsics)
  std::string export_name;         // filename to write, e.g. "src0_00123.jpg"
  Eigen::Isometry3d T_world_cam;   // in our frame convention
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
  // Also emit bundler.out alongside the COLMAP files. Bundler uses OpenGL
  // camera convention (X right, Y up, Z back) -- we convert internally.
  // Metashape imports Bundler directly (File -> Import Cameras... -> Bundler).
  bool export_bundler = false;
  // Also emit blocks_exchange.xml alongside the COLMAP files. BlocksExchange
  // is Bentley ContextCapture's format, also accepted by Metashape and
  // RealityCapture. Supports full Brown-Conrady distortion (k1/k2/k3/p1/p2)
  // unlike Bundler (k1/k2 only), so it's the better option when images are
  // exported RAW and downstream needs the distortion model.
  bool export_blocks_exchange = false;
  // Per-photo pose-prior accuracies emitted in BlocksExchange so Metashape BA
  // stays close to the imported poses. OFF -> poses are pure initial estimates,
  // BA can drift scale/origin. ON -> BA treats poses as constrained priors,
  // nailing the block in the session's coordinate frame (key for multi-tile
  // / aerial-LiDAR fusion with a constant offset). Bundler has no per-camera
  // accuracy field; these values are ignored there.
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

/// Write a COLMAP TEXT-format export (cameras.txt, images.txt, points3D.txt +
/// images/ folder) to options.output_dir.
///
/// Points are expected to already be filtered + voxelized by the caller.
/// Cameras will be further filtered here to those whose position is inside
/// the bounds (expanded by overlap_margin_m).
///
/// Axis convention: our (X fwd, Y left, Z up) → COLMAP (X right, Y down, Z fwd)
/// is handled inside. Caller passes poses in our convention.
ExportStats write_colmap_export(
  const ExportBounds2D& bounds,
  const std::vector<ColoredPoint>& points,              // world frame
  const std::vector<ExportCameraFrame>& cameras,        // world frame
  const std::vector<PinholeIntrinsics>& intrinsics_per_source,  // one per source_idx
  const std::vector<CameraType>& camera_type_per_source,         // one per source_idx; Pinhole or Spherical
  const ExportOptions& options,
  std::string* error_msg = nullptr);

}  // namespace glim
