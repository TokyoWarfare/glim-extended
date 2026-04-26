#pragma once

#include <vector>
#include <string>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <glim/util/post_processing.hpp>
#include <patchwork/patchworkpp.h>

namespace glim {

/**
 * @brief MapCleaner-style dynamic point identification.
 *
 * Algorithm (adapted from MapCleaner_Unofficial):
 * 1. Merge all HD frame points into a single world-space cloud (downsampled)
 * 2. Build a KD-tree on the merged cloud
 * 3. For each scan frame:
 *    a. Build a range image from the raw scan (sensor-local coords)
 *    b. Find merged-cloud points near the sensor (radius search)
 *    c. Transform those points to the scan's local frame
 *    d. Project each point to the range image, compare ranges
 *    e. Vote: CASE_A (match) → static, CASE_C (seen through) → dynamic
 * 4. Final: static_votes >= dynamic_votes → keep, else remove
 */
class MapCleanerFilter {
public:
  // Two cleaner modes:
  //   Voting       -- single-pass cross-frame vote (default, current behaviour).
  //   RemoveRevert -- two-pass remove-then-revert (Removert-style):
  //                   coarse pass with looser threshold/resolution flags wide,
  //                   fine pass with native params re-checks. A point is
  //                   dynamic only if BOTH passes agree -- much higher
  //                   precision, mildly lower recall on subtle dynamics.
  enum class Mode { Voting = 0, RemoveRevert = 1 };

  struct Params {
    Mode mode;              // see Mode enum above
    float coarse_thresh_mult;  // RemoveRevert: range_threshold multiplier for coarse pass
    float coarse_res_mult;     // RemoveRevert: res_h / res_v multiplier for coarse pass

    float fov_h;            // horizontal FOV (radians)
    float fov_v;            // vertical FOV (radians)
    float res_h;            // horizontal resolution (radians per pixel)
    float res_v;            // vertical resolution (radians per pixel)
    float range_threshold;  // metres — tolerance for range match
    int delta_h;            // neighborhood pixels horizontal
    int delta_v;            // neighborhood pixels vertical
    float lidar_range;      // max sensor range for submap extraction
    float submap_update_dist;  // rebuild submap every N metres
    float voxel_size;       // downsampling resolution for merged cloud
    int frame_skip;         // process every (skip+1)th frame
    float min_range;        // skip points closer than this

    bool exclude_ground_pw; // skip ground points using PatchWork++

    // ---- Robustness extensions (defaults preserve current behaviour) ----
    // Margin between vote_dynamic and vote_static required to flag dynamic.
    // 0 = simple majority (vote_dynamic > vote_static). Higher values demand
    // a clear dynamic signal and resist parallax / occlusion noise.
    int   vote_margin = 0;
    // Require the point to have been positively observed at least this many
    // times before letting any dynamic verdict win. Guards against single-
    // frame range artifacts on the static side. 1 = current behaviour.
    int   min_static_votes = 1;
    // Minimum sensor displacement (metres) between voting frames. Forces
    // parallax between consecutive votes -- kills "same-speed in front"
    // failure mode where a leading car/cyclist gets static votes from every
    // frame because it stays at the same relative position. 0 = disabled
    // (vote every frame, current behaviour).
    float min_baseline_m = 0.0f;
    // After voting, cluster the dynamic-flagged points and revert clusters
    // smaller than this back to static. Filters out single-point dynamic
    // flags from range image noise. 0 = disabled.
    int   min_dynamic_cluster_size = 0;
    // Voxel size used by the post-vote clustering BFS (metres). Same scale
    // as the trail-refine voxel; only used when min_dynamic_cluster_size > 0.
    float dynamic_cluster_voxel = 0.3f;

    Params()
      : mode(Mode::Voting),
        coarse_thresh_mult(2.0f), coarse_res_mult(2.0f),
        fov_h(2.0f * M_PI), fov_v(0.524f),
        res_h(0.0028f), res_v(0.007f),
        range_threshold(0.5f), delta_h(2), delta_v(1),
        lidar_range(70.0f), submap_update_dist(10.0f),
        voxel_size(0.2f), frame_skip(0), min_range(1.5f),
        exclude_ground_pw(true) {}
  };

  struct Result {
    std::vector<bool> is_dynamic;  // per-point flag (aligned with input cloud)
    int num_static = 0;
    int num_dynamic = 0;
  };

  /// HD frame data needed for processing
  struct FrameData {
    std::string dir;                    // path to HD frame directory
    Eigen::Isometry3d T_world_lidar;    // optimized world pose
    int num_points = 0;
  };

  MapCleanerFilter(const Params& params = Params()) : params_(params) {}

  /// Classify ground points using PatchWork++ for a single frame (sensor-local points)
  /// @param intensities Optional per-point intensity for RNR. Pass empty to skip RNR.
  static std::vector<bool> classify_ground_patchwork(
    const std::vector<Eigen::Vector3f>& local_points, int num_points, float sensor_height,
    const std::vector<float>& intensities = {});

  /// Access PatchWork++ params for UI configuration
  static patchwork::Params& getPatchWorkParams();

  /**
   * @brief Run dynamic point detection on a set of HD frames.
   *
   * @param frames        Ordered list of HD frame metadata
   * @param world_points  Pre-merged world-space points (all frames combined)
   * @param world_ranges  Per-point range from sensor
   * @return Result with per-point dynamic flags
   *
   * The world_points/world_ranges should be the output of merging all HD frames
   * into world coordinates (same as what the preview voxel grid contains).
   */
  Result compute(
    const std::vector<FrameData>& frames,
    const std::vector<Eigen::Vector3f>& world_points,
    const std::vector<float>& world_ranges,
    const std::vector<bool>& is_ground = {});

private:
  /// Build a range image from raw sensor-local points
  void build_range_image(
    const std::vector<Eigen::Vector3f>& local_points,
    const std::vector<float>& ranges,
    int num_points);

  /// Compare submap points against the current range image, accumulate votes
  void compare_and_vote(
    const std::vector<Eigen::Vector3f>& submap_local_pts,
    const std::vector<int>& submap_indices,
    std::vector<int>& vote_static,
    std::vector<int>& vote_dynamic);

  Params params_;

  // Range image storage (reused per frame)
  int im_width_ = 0;
  int im_height_ = 0;
  float res_h_scale_ = 1.0f;
  float res_v_scale_ = 1.0f;
  bool spinning_ = false;
  std::vector<float> range_image_;
};

}  // namespace glim
