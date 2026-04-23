#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <glim/util/lidar_colorizer.hpp>
#include <glim/mapping/sub_map.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>

namespace glim {

/// A small world-frame LiDAR slice built for calibration. Unlike a submap it
/// can be scoped tightly around a single camera (N frames before/after, time
/// window, or directional filter so outbound and return-pass data don't mix).
struct CalibrationContext {
  std::vector<Eigen::Vector3f> world_points;
  std::vector<float>           intensities;
  std::vector<Eigen::Vector3f> world_normals;  // may be empty
  // Anchor info for logging + directional filter
  double anchor_stamp = 0.0;
  Eigen::Vector3f anchor_forward = Eigen::Vector3f::UnitX();  // world-frame forward direction of the anchor LiDAR pose
};

/// Options for building the calibration context.
struct CalibContextOptions {
  int  n_frames_before = 15;
  int  n_frames_after  = 15;
  bool use_time_window = false;   // if true, uses time_before/after instead of frame counts
  double time_before_s = 3.0;
  double time_after_s  = 3.0;
  // Directional filter — discard LiDAR frames whose forward direction disagrees
  // with the anchor's forward direction by more than directional_threshold_deg.
  bool  directional_filter = true;
  float directional_threshold_deg = 60.0f;
  // Range gate (world meters from anchor camera)
  float min_range = 0.5f;
  float max_range = 80.0f;
  // Subsample to keep the projected cloud manageable (0 = off)
  int   max_points = 500000;
};

/// Result of a rendering pass: an intensity image + per-pixel 3D lookup.
struct RenderedIntensity {
  cv::Mat image;                     // CV_8UC1, intensity 0–255
  cv::Mat depth;                     // CV_32F, depth along camera forward; 0 = empty
  std::vector<int> pixel_to_point;   // size = W*H; stores the index into the ctx's
                                     // world_points of the closest point that wrote that pixel, or -1
};

/// A 2D↔3D correspondence produced by LightGlue + pixel→3D lookup.
struct CalibCorrespondence {
  Eigen::Vector2d uv_image;   // pixel in the REAL camera image
  Eigen::Vector3d xyz_world;  // world-frame 3D point it maps to (via rendered image)
  float confidence = 0.0f;    // LightGlue match score
};

/// Build a CalibrationContext by walking the session's submaps and accumulating
/// HD points around `anchor_stamp`. Applies a frame-count or time-seconds window
/// (see `opts.use_time_window`), an optional directional filter against
/// `anchor_forward`, and a world-range gate against `anchor_pos`. HD loading is
/// delegated to a caller-supplied callback so the builder stays viewer-agnostic;
/// the Auto-calibrate and Virtual Camera tools share this definition. `timed_traj`
/// is used only when the directional filter is on (to resolve each submap's
/// middle-frame forward direction).
CalibrationContext build_calibration_context(
  const std::vector<SubMap::ConstPtr>& submaps,
  const std::vector<TimedPose>& timed_traj,
  double anchor_stamp,
  const Eigen::Vector3f& anchor_pos,
  const Eigen::Vector3f& anchor_forward,
  const CalibContextOptions& opts,
  std::function<gtsam_points::PointCloudCPU::Ptr(int)> load_hd_for_submap);

/// Tuning knobs for the intensity rasterizer. Defaults match the values the
/// Auto-calibrate path has used historically -- callers that don't care can
/// pass the default-constructed struct.
struct IntensityRenderOptions {
  // How the per-point splat radius is chosen.
  //  - Formula:    clamp(round(3 * near_depth_m / depth), min, max). Smooth 1/depth taper.
  //  - Fixed:      same splat size for every depth (fixed_splat_px).
  //  - LinearRamp: user-defined (depth, size) knots. Between two consecutive knots
  //                the size linearly interpolates, so each knot's size matches the
  //                next knot's start without a step. Beyond the last knot the size
  //                stays flat at that knot's value.
  enum class SplatMode { Formula = 0, Fixed = 1, LinearRamp = 2 };
  SplatMode splat_mode = SplatMode::Formula;

  // Formula mode parameters.
  double near_depth_m = 3.0;
  int    min_splat_px = 2;
  int    max_splat_px = 6;

  // Fixed mode parameter.
  int fixed_splat_px = 3;

  // LinearRamp knots. Sorted ascending by start_depth_m (rasterizer re-sorts
  // defensively). When empty the rasterizer falls back to fixed_splat_px so
  // there's always a valid answer. Field name kept as SplatRange / splat_ranges
  // for continuity; semantic is "knot at depth d with size s" rather than
  // "constant over a range".
  struct SplatRange { double start_depth_m = 0.0; int splat_px = 3; };
  std::vector<SplatRange> splat_ranges;

  // When true, apply the gamma-0.5 lift + top-5% hard-clip to 250-255 so
  // retroreflective markings pop. When false, a straight linear stretch from
  // (2nd..99th percentile) to 0-255 is used instead -- useful for comparing
  // how much the non-linear shaping helps LightGlue matching.
  bool   non_linear_intensity = true;

  // Intensity-range lock. Per-frame percentile compute makes two rasterizations
  // of the same scene look like they have different exposure -- a synthetic
  // camera should be steady. When locked, the rasterizer skips the percentile
  // computation and uses the values captured below instead (usually grabbed
  // from a reference preview frame). The three percentile anchors are:
  //   imin   (2nd percentile) -- dark clip point
  //   ibulk  (95th percentile) -- split point between bulk and retroreflective spike (non-linear)
  //   imax   (99th percentile) -- bright clip point (linear) or spike upper end (non-linear)
  bool   intensity_locked = false;
  float  intensity_locked_imin  = 0.0f;
  float  intensity_locked_ibulk = 230.0f;
  float  intensity_locked_imax  = 250.0f;
  // Map the final 8-bit intensity to an 8UC3 colormap when non-zero.
  // 0 = Linear grayscale (CV_8UC1), 1 = Inverted grayscale, 2 = Turbo,
  // 3 = Viridis, 4 = Cividis. Colormap indices match glk::COLORMAP where
  // applicable so the dropdown wiring is trivial.
  enum class Colormap { Grayscale = 0, Inverted = 1, Turbo = 2, Viridis = 3, Cividis = 4 };
  Colormap colormap = Colormap::Grayscale;

  // When on (default), apply the chosen colormap, then convert the resulting
  // RGB back to grayscale (BT.601 luminance). The colormap's non-linear hue
  // sweeps act as a contrast remapping -- e.g. Turbo's cyan->yellow->red
  // transitions translate to sharper luminance steps than a straight linear
  // grayscale, pulling out faint intensity features (road markings, sign
  // glyphs) that a direct grayscale would wash out. LightGlue prefers
  // grayscale input, so the final tensor stays single-channel.
  bool return_to_grayscale_after_colormap = true;

  // Round vs square splats. Square is the AABB dy/dx loop (cheap, stamps a
  // filled square). Round adds a dx*dx + dy*dy <= r*r gate, producing filled
  // disks -- matches how the 3D viewer paints points. Default OFF because
  // square's axis-aligned corner edges sometimes give LightGlue extra cheap
  // features and the ~30% extra coverage can push match counts up. Flip on
  // if you want a more natural look or are A/B testing match yield.
  bool round_splats = false;
};

/// Compute the 2nd / 95th / 99th percentiles of a point cloud's intensity
/// channel -- the same anchors the rasterizer uses internally. Exposed so the
/// Virtual Camera UI can grab them off a reference frame to lock the intensity
/// range for consistent contrast across subsequent renders.
void compute_intensity_percentiles(
  const std::vector<float>& intensities,
  float& imin, float& ibulk, float& imax);

/// Bucket LightGlue match confidences into high / mid / low tiers so the UI
/// can show "X high, Y mid, Z low" instead of just a total. Thresholds are
/// the defaults LightGlue produces usable matches at (>=0.8 is typically a
/// green-line match, 0.5-0.8 mid, <0.5 noise).
struct MatchQualityStats {
  int   total = 0;
  int   high = 0;           // score >= high_thresh
  int   mid  = 0;           // mid_thresh <= score < high_thresh
  int   low  = 0;           // score < mid_thresh
  float high_thresh = 0.8f;
  float mid_thresh  = 0.5f;
};
MatchQualityStats compute_match_quality(
  const std::vector<float>& scores,
  float high_thresh = 0.8f,
  float mid_thresh  = 0.5f);

/// Render an intensity image using the current predicted camera pose + intrinsics.
/// Uses the same projection convention as LidarColorizerSimple (X fwd, Y left, Z up).
RenderedIntensity render_intensity_image(
  const CalibrationContext& ctx,
  const Eigen::Isometry3d& T_world_cam,
  const PinholeIntrinsics& intrinsics,
  int width, int height,
  const IntensityRenderOptions& opts = {});

/// Load LightGlue matches from JSON produced by scripts/lightglue_match.py.
/// The JSON format is:
///   { "matches": [ {"real":[u,v], "rendered":[u,v], "score":float}, ... ] }
std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> load_lightglue_matches(
  const std::string& json_path, std::vector<float>* confidences = nullptr);

/// Map rendered-image pixel coordinates back to 3D world points via the render's
/// pixel_to_point table, and return (uv_real, xyz_world) correspondences.
std::vector<CalibCorrespondence> matches_to_correspondences(
  const std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>>& matches_real_rendered,
  const std::vector<float>& confidences,
  const RenderedIntensity& render,
  const CalibrationContext& ctx);

/// Solve PnP on the correspondences → refined T_world_cam.
/// Returns false if not enough inliers. residual_px is populated with the mean
/// reprojection error on inliers (in pixels).
bool refine_extrinsic_pnp(
  const std::vector<CalibCorrespondence>& corrs,
  const PinholeIntrinsics& intrinsics,
  const Eigen::Isometry3d& T_world_cam_init,
  Eigen::Isometry3d& T_world_cam_refined,
  int& n_inliers, double& residual_px);

/// Joint LM refinement of intrinsics (fx, fy, cx, cy, k1, k2, p1, p2, k3) and
/// optionally extrinsic. Runs after a successful PnP extrinsic refinement.
bool refine_intrinsics_lm(
  const std::vector<CalibCorrespondence>& corrs,
  PinholeIntrinsics& intrinsics,            // in/out
  Eigen::Isometry3d& T_world_cam,           // in/out
  bool lock_extrinsic,
  double& residual_px);

}  // namespace glim
