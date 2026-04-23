#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <glim/util/lidar_colorizer.hpp>

namespace glim {

/// Options controlling how virtual LiDAR-rendered cameras are placed + rendered.
struct VirtualCameraOptions {
  std::string output_dir;

  // Placement: walk trajectory and emit an anchor every `interval_m` metres.
  float interval_m = 10.0f;

  // Which of the 6 cube faces to render at each anchor. Indices:
  //   0 +X (forward)  1 -X (back)
  //   2 +Y (left)     3 -Y (right)
  //   4 +Z (up)       5 -Z (down)
  // The sky face (+Z) is skipped by default -- mostly black, rarely has features.
  std::array<bool, 6> face_enabled = { true, true, true, true, false, true };

  // Face resolution (square). Higher = more features per face, slower render.
  int face_size = 1920;

  // Context window around each anchor (metres). Points within this radius of
  // the anchor are included in the render. Narrower = faster, less wrap-around
  // leakage; wider = more context from further surfaces.
  float context_radius_m = 60.0f;

  // If true, only points flagged as ground (PatchWork++ aux_ground.bin) are
  // rendered. Useful when asphalt / kerbs / road markings are the best matchable
  // features (Livox Horizon use case).
  bool ground_only = false;

  // Also render an RGB image per face using the per-point aux_rgb.bin data
  // (requires a prior Colorize Apply run). Intensity is always rendered.
  bool render_rgb = false;

  // Embed UTM coordinates into JPEG EXIF GPS tags. Uses the session's
  // gnss_datum.json UTM origin + the anchor's session-local position.
  bool embed_exif_gps = true;
};

/// A single anchor location along the trajectory, with its world pose.
struct VirtualCameraAnchor {
  int index = 0;                                     // sequential 0..N-1
  double stamp = 0.0;                                // trajectory timestamp at this anchor
  Eigen::Isometry3d T_world_cam = Eigen::Isometry3d::Identity();   // GLIM convention (X fwd, Y left, Z up)
  double cumulative_dist_m = 0.0;                    // along-trajectory distance
};

/// Output stats from a render run.
struct VirtualCameraStats {
  size_t anchors_placed = 0;
  size_t faces_rendered = 0;
  size_t rgb_faces_rendered = 0;
  std::string error_msg;
};

/// Walk the trajectory and emit an anchor every `interval_m` metres. Uses
/// cumulative_dist_m from TimedPose-like records -- caller supplies a list
/// already sorted by time and tagged with cumulative distance. Returns the
/// anchors with poses interpolated between the given records. Does no IO.
struct TrajRecord {
  double stamp;
  Eigen::Isometry3d pose;
  double cumulative_dist;
};
std::vector<VirtualCameraAnchor> place_virtual_cameras(
  const std::vector<TrajRecord>& trajectory,
  float interval_m);

}  // namespace glim
