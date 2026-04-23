#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>

namespace glim {

/// Pinhole camera intrinsics with optional Brown-Conrady distortion.
struct PinholeIntrinsics {
  double fx = 1920.0, fy = 1920.0;  // focal length in pixels (Elgato Facecam Pro 90° FOV default)
  double cx = 1920.0, cy = 1080.0;  // principal point (image center)
  int width = 3840, height = 2160;   // image resolution
  // Brown-Conrady distortion coefficients
  double k1 = 0.0, k2 = 0.0;       // radial distortion
  double p1 = 0.0, p2 = 0.0;       // tangential distortion
  double k3 = 0.0;                  // higher-order radial
};

/// Single camera frame with EXIF-derived metadata and placed pose.
struct CameraFrame {
  std::string filepath;
  double timestamp = 0.0;           // UNIX epoch (from EXIF DateTimeOriginal + SubSec)
  double lat = 0.0, lon = 0.0, alt = 0.0;  // from EXIF GPS
  Eigen::Isometry3d T_world_cam = Eigen::Isometry3d::Identity();
  bool located = false;
  // Optional pre-loaded image data. When non-null, colorizers use this instead
  // of cv::imread(filepath). Populated by the spherical-source cube-face
  // expansion so the pipeline sees virtual pinhole cameras with in-memory faces.
  std::shared_ptr<cv::Mat> image_override;
  std::shared_ptr<cv::Mat> mask_override;
};

/// Produces the six 90 deg cube faces of an equirectangular image. Face indices
/// (match our camera convention X fwd, Y left, Z up):
///   0 = +X (forward)   1 = -X (back)
///   2 = +Y (left)      3 = -Y (right)
///   4 = +Z (up)        5 = -Z (down)
/// `face_size` is the output resolution (square). Uses cv::remap with a
/// precomputed map cached by (equirect_size, face_size).
std::array<std::shared_ptr<cv::Mat>, 6> slice_equirect_cubemap(const cv::Mat& equirect, int face_size);

/// Rotation R_camera_face such that face-local forward (+X in face frame)
/// maps to one of the six cube directions in the parent equirect camera frame.
/// Composed with the equirect's T_world_cam to obtain the face's world pose.
Eigen::Matrix3d cube_face_rotation(int face_idx);

/// Synthetic pinhole intrinsics for a cube face at the given size. 90 deg FOV:
/// fx = fy = size/2, cx = cy = size/2, zero distortion.
PinholeIntrinsics cube_face_intrinsics(int face_size);

/// Camera projection model. Public enum so any UI / pipeline stage can branch
/// on it and skip features that don't apply (distortion params, rectification,
/// etc). Integer values are persisted in colorize_config.json so DO NOT reorder
/// existing entries -- only append.
///   Pinhole   - standard perspective lens, Brown-Conrady distortion via PinholeIntrinsics.
///   Spherical - 2:1 equirectangular 360 panorama, focal derived as f = w / (2*pi).
///   Fisheye   - RESERVED for a future OpenCV fisheye model; not yet implemented.
enum class CameraType {
  Pinhole       = 0,
  Spherical     = 1,
  Fisheye       = 2,   // reserved; pipelines should fall back to Pinhole behavior
};

/// Human-readable label for a camera type -- used for UI badges and logs so
/// every window shows the same wording.
inline const char* camera_type_label(CameraType t) {
  switch (t) {
    case CameraType::Pinhole:   return "Pinhole";
    case CameraType::Spherical: return "Spherical";
    case CameraType::Fisheye:   return "Fisheye";
  }
  return "?";
}

/// True if this camera type uses Brown-Conrady k1/k2/k3/p1/p2 distortion.
/// UI paths that offer "Rectify", "Distortion preview", etc should gate on this.
inline bool camera_type_has_brown_conrady(CameraType t) {
  return t == CameraType::Pinhole;
}

/// View-selection strategy for LiDAR colorization. Declared up here so
/// ColorizeParams (below) can hold it directly instead of an int.
enum class ViewSelectorMode {
  SimpleNearest,   // closest camera wins per point
  WeightedTop1,    // top-1 weighted (range × center × incidence)
  WeightedTopK,    // softmax blend across top-K
  Matched          // future: LightGlue-assisted, per-pixel warp, semantic-gated
};

/// Trajectory point with timestamp for camera placement. Declared ahead of
/// ImageSource so the per-source alternative `camera_trajectory` vector can
/// use it as a complete type.
struct TimedPose {
  double stamp;
  Eigen::Isometry3d pose;  // T_world_lidar
};

/// Per-source colorize tuning. Previously these lived as viewer-level globals,
/// which meant flipping the Source dropdown reused the last-touched tuning
/// across sources (and masks / intrinsics bled similarly). Nesting them here
/// makes every knob follow the source it was tuned on, and round-trips via the
/// JSON serializer for free.
///
/// Adding a new field: append it (defaults matter -- old configs must still
/// load cleanly), give it a sensible default for both Pinhole and Spherical,
/// and handle it in the ImGui block + serializer. Removing one: drop the field
/// here, the loader's `.value(...)` call fallsback gracefully.
struct ColorizeParams {
  // Locate criteria: how to place cameras on the trajectory.
  //   0 = Time: interpolate SLAM trajectory at cam_time + time_shift.
  //   1 = Coords -> SLAM: snap each EXIF-GPS frame to the nearest SLAM point
  //       (useful when camera timestamps aren't GPS-synced).
  //   2 = Coords -> own path: build an alternative trajectory from this
  //       source's EXIF GPS track and interpolate THAT at cam_time + time_shift.
  //       Used when colouring a SLAM map with a second-pass GPS+camera sweep
  //       that doesn't share the original capture timebase.
  int   locate_mode = 0;

  // Range gates (metres) for projection.
  float min_range = 3.0f;                // skip close-up points
  float max_range = 30.0f;               // skip far-away points

  // Blend / exposure controls.
  bool  blend = false;                   // legacy: average across multiple cameras
  bool  intensity_blend = false;         // mix RGB with intensity colormap
  float intensity_mix = 0.5f;            // 0 = pure RGB, 1 = pure intensity
  bool  nonlinear_int = false;           // sqrt-compress high intensities

  // View-selection strategy + its tunables.
  ViewSelectorMode view_selector_mode = ViewSelectorMode::SimpleNearest;
  float range_tau = 10.0f;
  float center_exp = 2.0f;
  float incidence_exp = 1.0f;
  int   topK = 1;

  // Incidence hard cutoff -- reject points whose surface normal makes too
  // steep an angle with the view ray.
  bool  use_incidence_hard = false;
  float incidence_hard_deg = 60.0f;

  // NCC patch rejection.
  bool  use_ncc = false;
  float ncc_threshold = 0.3f;
  int   ncc_half = 3;                    // 7x7 patch

  // Occlusion reverse-Z-buffer check.
  bool  use_occlusion = false;
  float occlusion_tolerance = 0.05f;     // fraction of depth
  int   occlusion_downscale = 4;

  // Time-slice gating -- on forward-facing pinhole a tight slice is correct;
  // on Spherical a looser slice is the right default (the 360 sees every
  // direction the LiDAR has ever swept through while the vehicle moves).
  bool  time_slice_hard = false;
  bool  time_slice_soft = false;
  float time_slice_sigma = 0.05f;        // seconds

  // Per-image exposure normalization (FAST-LIVO2 trick).
  bool  use_exposure_norm = false;
  float exposure_target = 0.5f;
  bool  exposure_simple = false;         // false = surface-mean, true = image-mean
};

/// Reasonable defaults for a ColorizeParams given a camera type. Applied when
/// a source is freshly added (load paths merge JSON on top of these defaults).
/// Centralised here so the UI + loader + source-creation path agree.
inline ColorizeParams default_colorize_params_for(CameraType t) {
  ColorizeParams p;
  if (t == CameraType::Spherical) {
    // 360 camera sees everything the vehicle has swept past -- widen range,
    // relax time-slice. Weighted Top-1 gives better coverage than nearest-only
    // when many cameras see the same distant point from different angles.
    p.max_range = 60.0f;
    p.time_slice_hard = false;
    p.time_slice_soft = true;
    p.time_slice_sigma = 0.25f;
    p.view_selector_mode = ViewSelectorMode::WeightedTop1;
  }
  return p;
}

/// Per-moment time-shift snapshot. When a source has 2+ anchors, the effective
/// time_shift at a given query cam_time is interpolated linearly between the
/// neighboring anchors; outside the anchor range the nearest anchor's value
/// is held (no extrapolation).
///
/// Use case: slow time drift during long drives -- the camera clock vs LiDAR
/// clock can drift by fractions of a second over kilometers. User nails sync
/// at frame A, drives, re-syncs at frame B; the pipeline smoothly interpolates
/// between them per frame.
///
/// Scope note: lever_arm + rotation_rpy stay as single scalar fields on
/// ImageSource (no per-anchor extrinsic drift yet). The struct can be extended
/// later without breaking the JSON schema (loaders use `.value(key, default)`).
///
/// cam_time is source-local (frame.timestamp BEFORE applying time_shift) so
/// the anchor key is stable regardless of subsequent time_shift nudges.
struct CalibAnchor {
  double cam_time = 0.0;                       // source-local timestamp at capture moment
  double time_shift = 0.0;                     // seconds -- replaces source.time_shift for this moment
};

/// Effective calibration values at a specific cam_time.
struct EffectiveCalib {
  double time_shift;
  Eigen::Vector3d lever_arm;
  Eigen::Vector3d rotation_rpy;
};

/// A folder of images treated as one camera source.
struct ImageSource {
  std::string path;                  // folder path
  std::string name;                  // display name (folder basename)
  std::vector<CameraFrame> frames;
  Eigen::Vector3d lever_arm = Eigen::Vector3d::Zero();  // baseline camera offset from LiDAR in sensor-local frame
  Eigen::Vector3d rotation_rpy = Eigen::Vector3d::Zero();  // baseline camera rotation relative to LiDAR (roll, pitch, yaw) in degrees
  double time_shift = 0.0;          // baseline seconds offset, slides cameras along trajectory
  PinholeIntrinsics intrinsics;     // camera model (pinhole-only; ignored for Spherical)
  std::string mask_path;            // path to mask image (black = exclude from projection)
  CameraType camera_type = CameraType::Pinhole;  // projection model used by the colorizer
  ColorizeParams params;            // all tuning knobs (locate mode, ranges, view selector, gates)
  // Calibration drift anchors. When empty (size 0) the scalar time_shift /
  // lever_arm / rotation_rpy above are used as-is. When size >= 1, the scalar
  // fields are IGNORED and calibration at any cam_time is resolved from the
  // anchors (single anchor: constant; 2+: linear interp between neighbors).
  // Kept sorted by cam_time after every insert/remove.
  std::vector<CalibAnchor> anchors;
  // Alternative trajectory derived from this source's EXIF GPS track, used
  // only when params.locate_mode == 2 ("Coords -> own path"). Built by
  // build_camera_trajectory() on source load or datum change; empty otherwise.
  // Kept out of the JSON serialiser because it's fully derivable from EXIF +
  // datum and would bloat configs with redundant data.
  std::vector<TimedPose> camera_trajectory;
  // Time Matcher anchor state -- persisted so dumb-frames sources (no EXIF
  // timestamps) keep their back-filled timestamps across session reloads.
  // A value of tm_anchor1_idx >= 0 means this source has been matched and its
  // frame timestamps were back-filled at load time using the formula:
  //   t[i] = tm_anchor1_time + (i - tm_anchor1_idx) * interval
  // where interval is solved from two anchors (if tm_anchor2_idx >= 0) or
  // derived from tm_fps otherwise.
  int    tm_anchor1_idx = -1;
  double tm_anchor1_time = 0.0;
  int    tm_anchor2_idx = -1;
  double tm_anchor2_time = 0.0;
  float  tm_fps = 30.0f;
};

/// Resolve the effective calibration (time_shift + lever_arm + rotation_rpy)
/// at the given source-local cam_time. Rules:
///   0 anchors  -> returns src.time_shift, src.lever_arm, src.rotation_rpy (baseline).
///   1 anchor   -> returns that anchor's values regardless of cam_time.
///   2+ anchors -> linear interpolation between the two neighboring anchors;
///                 clamped to nearest anchor outside the anchor range.
EffectiveCalib effective_calib(const ImageSource& src, double cam_time);

/// Convenience -- just the time_shift field. Cheap enough that call sites can
/// use this instead of EffectiveCalib when they only need the time.
inline double effective_time_shift(const ImageSource& src, double cam_time) {
  return effective_calib(src, cam_time).time_shift;
}

/// Build `src.camera_trajectory` from its EXIF GPS (lat/lon/alt + timestamp).
/// Skips frames missing timestamp or GPS. Sorts by timestamp, converts
/// lat/lon to world coords via (UTM - datum origin), derives yaw from the
/// velocity vector between successive positions; roll/pitch = 0 (rigid
/// rotation_rpy/lever_arm on the source still composes on top). Lazy: call
/// once per source after load, and whenever the datum origin changes. Safe
/// to call with empty EXIF -- leaves camera_trajectory empty so callers fall
/// back to the SLAM trajectory via trajectory_for().
void build_camera_trajectory(ImageSource& src,
                              int utm_zone,
                              double easting_origin,
                              double northing_origin,
                              double alt_origin);

/// Return the trajectory the given source should place itself on. Routes to
/// src.camera_trajectory when locate_mode == 2 and the alt trajectory is
/// populated; otherwise falls back to the shared SLAM trajectory. All Time-
/// based placement call sites (locate_by_time, alignment-check interp, anchor
/// gizmos, live preview) funnel through here so the alt path behaves
/// identically to the primary path just with a different pose source.
const std::vector<TimedPose>& trajectory_for(const ImageSource& src,
                                              const std::vector<TimedPose>& slam_trajectory);

/// Result of color projection for a set of points.
struct ColorizeResult {
  std::vector<Eigen::Vector3f> points;   // world-space positions
  std::vector<Eigen::Vector3f> colors;   // RGB [0-1]
  std::vector<float> intensities;        // original intensities (for fallback)
  size_t colored = 0;                    // points that received at least one color sample
  size_t total = 0;
  // Per-point diagnostics (filled by weighted strategies; empty for simple)
  std::vector<int>   winner_cam;         // index of the winning camera per point, -1 if none
  std::vector<float> winner_weight;      // weight value of the winner per point
};

/// Parameters common to all colorizer strategies. Strategy-specific fields
/// (weight exponents, topK, etc.) are ignored by strategies that don't use them.
struct BlendParams {
  // Range gate (all strategies)
  double max_range = 50.0;
  double min_range = 0.0;
  // Simple strategy knob: true = average all contributing cameras, false = nearest-camera color
  bool   simple_average = true;
  // Optional mask (all strategies)
  cv::Mat mask;
  // Weighted strategy knobs (ignored by Simple)
  float range_tau     = 10.0f;  // w_range = exp(-dist / tau)
  float center_exp    = 2.0f;   // w_center = (1 - r_norm^2)^center_exp
  float incidence_exp = 1.0f;   // w_incidence = max(0, cos(theta))^incidence_exp
  int   topK          = 1;      // 1 = hard best, >1 = softmax over top-K
  // Hard gates — applied BEFORE weighting, drop the camera-point pair outright.
  // Inspired by FAST-LIVO2's getCloseViewObs hard 60° gate.
  float incidence_hard_cos = -1.0f;  // -1 = disabled; else reject if |cos(normal,ray)| < this
  // NCC cross-check (top-K blend only). -2 = disabled. Range [-1, 1]; typical 0.2–0.5.
  // After top-K is picked, each non-winner patch is NCC-compared against the winner.
  // Non-winners below threshold are dropped from the blend. Unaffected for top-1.
  float ncc_threshold = -2.0f;
  int   ncc_patch_half = 3;      // 3 -> 7x7 patch
  // Depth-buffer occlusion (shared by Simple + Weighted). Per-camera z-buffer is
  // built at (image_w / zbuf_downscale) × (image_h / zbuf_downscale); a point is
  // rejected if its depth is farther than zbuf[u,v] * (1 + occlusion_tolerance).
  bool  use_depth_occlusion = false;
  float occlusion_tolerance = 0.05f;  // 5% of depth
  int   occlusion_z_downscale = 4;
  // Time-slicing (FAST-LIVO2 style). For each point, restrict candidate cameras
  // to the temporally closest one(s). Hard mode = only the single closest camera
  // is allowed to contribute; soft mode = adds a multiplicative w_time weight
  // exp(-|Δt| / sigma). Requires world_point_times to be provided; no-op otherwise.
  bool  time_slice_hard = false;
  bool  time_slice_soft = false;        // independent of hard; adds w_time to weighted strategies
  float time_slice_sigma = 0.05f;       // seconds; decay for w_time (soft mode)
  // Per-image exposure normalization (FAST-LIVO2 trick #1). Each loaded image gets
  // a scalar gain = (exposure_target * 255) / current_image_mean, applied to sampled
  // RGB before it contributes. Kills seams caused by auto-exposure jumps between frames.
  bool  exposure_normalize = false;
  float exposure_target = 0.5f;          // desired mean brightness [0,1]; 0.5 = neutral
  // When true, use the simpler legacy mean: downscaled full-image mean. When false,
  // use the surface-pixel mean (only pixels where LiDAR points project). Surface-pixel
  // is unbiased by sky/ceiling but can over-boost sunny scenes. Simple is a bit less
  // accurate but visually more balanced on outdoor bright data.
  bool  exposure_simple = false;
};

/// Abstract strategy for projecting camera colors onto 3D points.
/// Each implementation lives in its own .cpp (lidar_colorizer_simple.cpp,
/// lidar_colorizer_weighted.cpp, lidar_colorizer_matched.cpp).
///
/// world_normals: optional per-point surface normals in world frame. Pass an
/// empty vector if unavailable — strategies that don't need normals (Simple)
/// will ignore it; strategies that do (Weighted's w_incidence) will skip that
/// weight term if the vector is empty or sized mismatched.
class ILidarColorizer {
public:
  virtual ~ILidarColorizer() = default;
  virtual ColorizeResult project(
    const std::vector<CameraFrame>& cameras,
    const PinholeIntrinsics& intrinsics,
    const std::vector<Eigen::Vector3f>& world_points,
    const std::vector<float>& intensities,
    const std::vector<Eigen::Vector3f>& world_normals,
    const std::vector<double>& world_point_times,
    const BlendParams& params) = 0;
};

/// Factory. Returns the concrete strategy for the given mode.
std::unique_ptr<ILidarColorizer> make_colorizer(ViewSelectorMode mode);

class Colorizer {
public:
  /// Scan a folder for images (.jpg/.jpeg/.png), read EXIF GPS + timestamp.
  static ImageSource load_image_folder(const std::string& folder_path);

  /// Place cameras along a trajectory using timestamp matching.
  static int locate_by_time(ImageSource& source, const std::vector<TimedPose>& trajectory);

  /// Place cameras along a trajectory using GPS coordinates.
  static int locate_by_coordinates(ImageSource& source, const std::vector<TimedPose>& trajectory,
                                    int utm_zone, double easting_origin, double northing_origin, double alt_origin);

  /// Interpolate pose on trajectory at given timestamp.
  static Eigen::Isometry3d interpolate_pose(const std::vector<TimedPose>& trajectory, double stamp);

  /// Build T_lidar_camera from lever arm + RPY rotation (degrees).
  static Eigen::Isometry3d build_extrinsic(const Eigen::Vector3d& lever_arm, const Eigen::Vector3d& rotation_rpy);

  /// Build T_lidar_camera using anchor-aware calibration resolution. When the
  /// source has no anchors, this is identical to build_extrinsic(lever_arm, rpy).
  /// When it has 2+, returns the extrinsic interpolated at cam_time.
  static Eigen::Isometry3d effective_extrinsic(const ImageSource& src, double cam_time);

  /// Solve camera-to-lidar extrinsic from 2D-3D correspondences.
  /// @param pts_3d  World-space 3D points clicked in the viewer
  /// @param pts_2d  Corresponding 2D pixel coords clicked in the image
  /// @param intrinsics  Camera intrinsics
  /// @param T_world_lidar  LiDAR pose at the camera's timestamp
  /// @return T_lidar_camera (transform from camera frame to lidar frame)
  static Eigen::Isometry3d solve_extrinsic(
    const std::vector<Eigen::Vector3d>& pts_3d,
    const std::vector<Eigen::Vector2d>& pts_2d,
    const PinholeIntrinsics& intrinsics,
    const Eigen::Isometry3d& T_world_lidar);

private:
  static bool read_exif(const std::string& filepath, double& timestamp, double& lat, double& lon, double& alt);
};

}  // namespace glim
