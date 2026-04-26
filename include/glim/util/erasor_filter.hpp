#pragma once

#include <string>
#include <vector>
#include <Eigen/Geometry>

namespace glim {

/**
 * @brief Per-scan polar pseudo-occupancy dynamic-point removal (ERASOR-style).
 *
 * Reimplementation of the ERASOR algorithm (Lim et al., RA-L 2021) tailored
 * for offline map cleanup inside the GLIM pipeline. Same shape as
 * MapCleanerFilter so the UI / batch runners can swap them.
 *
 * Algorithm summary:
 *   - Per query frame, around the sensor position, build a polar R-VOI
 *     (rings × sectors). For each polar bin, the "pseudo-occupancy
 *     descriptor" is the max-Z (sensor-local height) of points falling in
 *     the bin.
 *   - Build the same descriptor for the chunk's pre-merged map points
 *     (passed as world_points; transformed to sensor frame here).
 *   - For each bin, ratio = query_max_z / map_max_z. If ratio < threshold
 *     the bin's map points are dynamic candidates -- the query saw less
 *     occupancy than the map says exists, i.e. things have moved away.
 *   - Cumulative voting across all frames in the chunk. A point is dynamic
 *     iff dynamic_votes > static_votes (same comparator as MapCleaner so
 *     downstream gates - PW guarantee, Z-safety, cluster verify - apply
 *     transparently).
 *
 * Inputs match MapCleanerFilter's compute() signature so the chunk-loading
 * code paths in run_erasor / run_dynamic can stay symmetrical.
 */
class ErasorFilter {
public:
  struct Params {
    int   num_rings = 30;
    int   num_sectors = 108;
    float max_range = 30.0f;
    float min_range = 1.0f;
    float ratio_threshold = 0.01f;  // bin is dynamic if query/map < this
    bool  exclude_ground_pw = true; // ground points stay static; same flag as MapCleaner
    int   frame_skip = 0;           // process every (skip+1)th frame; 0 = every frame.
                                    // Auto-set by callers when frames > 200 to keep
                                    // chunk processing tractable on large datasets.
    // Vertical FOV half-angle (degrees). Used to gate which points the LiDAR
    // can physically see at a given range. For each polar bin, a point is
    // included in the R-PoD only if its angular elevation from the sensor
    // satisfies |atan2(z, r)| <= sensor_v_fov_half_deg.
    //
    // Why: narrow-V-FOV sensors (Livox Horizon ~25 deg, half ~12.5 deg) can't
    // see tall structures' tops at distance. The map (built from many vantage
    // points) DOES have those tops. Without this gate, query and map disagree
    // on max-Z and the ratio test false-flags the whole pole as dynamic.
    // With this gate, both query and map only consider points the sensor
    // could actually have seen at that range, so they agree on visible
    // portion, ratio ~1, no false dynamic.
    //
    // Defaults:
    //   Livox Horizon ~12.5
    //   Velodyne VLP-16  ~15
    //   Ouster OS1-64    ~22
    //   Pandar 128       ~25
    //   Set to 90 to disable (= count all points, original ERASOR behaviour).
    float sensor_v_fov_half_deg = 12.5f;
  };

  struct Result {
    std::vector<bool> is_dynamic;
    int num_static = 0;
    int num_dynamic = 0;
  };

  struct FrameData {
    std::string dir;                    // path to HD frame directory
    Eigen::Isometry3d T_world_lidar;    // optimized world pose
    int num_points = 0;
  };

  ErasorFilter() = default;
  explicit ErasorFilter(const Params& params) : params_(params) {}

  /**
   * @brief Run ERASOR-style dynamic detection.
   * @param frames        Ordered list of frame metadata.
   * @param world_points  Pre-merged world-space chunk points (the "map").
   * @param is_ground     Optional per-point ground flag; ground points get
   *                      forced static (matches MapCleaner semantics).
   * @return Per-point dynamic flags + counts.
   */
  Result compute(
    const std::vector<FrameData>& frames,
    const std::vector<Eigen::Vector3f>& world_points,
    const std::vector<bool>& is_ground = {});

private:
  Params params_;
};

}  // namespace glim
