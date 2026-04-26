#include <glim/util/erasor_filter.hpp>
#include <glim/util/post_processing.hpp>

#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>

#include <nanoflann.hpp>

namespace glim {

namespace {

// nanoflann adaptor for vector of Vector3f -- mirrors MapCleaner's local helper.
struct PointCloudAdaptor {
  const std::vector<Eigen::Vector3f>& points;
  PointCloudAdaptor(const std::vector<Eigen::Vector3f>& pts) : points(pts) {}
  inline size_t kdtree_get_point_count() const { return points.size(); }
  inline float kdtree_get_pt(const size_t idx, const size_t dim) const { return points[idx][dim]; }
  template <class BBOX> bool kdtree_get_bbox(BBOX&) const { return false; }
};

using KDTree3f = nanoflann::KDTreeSingleIndexAdaptor<
  nanoflann::L2_Simple_Adaptor<float, PointCloudAdaptor>,
  PointCloudAdaptor, 3>;

// Polar bin index (ring, sector) flattened.
inline int polar_bin(float r, float theta, float min_range, float max_range, int num_rings, int num_sectors) {
  if (r < min_range || r > max_range) return -1;
  int ring = static_cast<int>((r - min_range) / std::max(1e-6f, max_range - min_range) * num_rings);
  ring = std::clamp(ring, 0, num_rings - 1);
  if (theta < 0.0f) theta += 2.0f * M_PI;
  if (theta > 2.0f * M_PI) theta -= 2.0f * M_PI;
  int sector = static_cast<int>(theta / (2.0f * M_PI) * num_sectors);
  sector = std::clamp(sector, 0, num_sectors - 1);
  return ring * num_sectors + sector;
}

}  // namespace

ErasorFilter::Result ErasorFilter::compute(
    const std::vector<FrameData>& frames,
    const std::vector<Eigen::Vector3f>& world_points,
    const std::vector<bool>& is_ground) {

  Result result;
  result.is_dynamic.resize(world_points.size(), false);
  std::cerr << "[ErasorFilter] enter: world_points=" << world_points.size()
            << " frames=" << frames.size() << " is_ground=" << is_ground.size()
            << " (max_range=" << params_.max_range << ", rings=" << params_.num_rings
            << ", sectors=" << params_.num_sectors << ", ratio_thr=" << params_.ratio_threshold
            << ")" << std::endl;
  if (world_points.empty() || frames.empty()) {
    std::cerr << "[ErasorFilter] EARLY EXIT (world_points or frames empty)" << std::endl;
    return result;
  }
  const bool has_ground = !is_ground.empty() && is_ground.size() == world_points.size();

  // Build KD-tree on all map points so each frame can pull its R-VOI quickly.
  PointCloudAdaptor adaptor(world_points);
  KDTree3f kdtree(3, adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10));
  kdtree.buildIndex();

  const float max_range_sq = params_.max_range * params_.max_range;
  const int   num_bins     = params_.num_rings * params_.num_sectors;

  std::vector<int> static_votes(world_points.size(), 0);
  std::vector<int> dynamic_votes(world_points.size(), 0);

  int frames_processed = 0, frames_load_fail = 0, frames_no_matches = 0;
  long long total_dyn_votes = 0, total_stat_votes = 0;
  const int skip = std::max(0, params_.frame_skip);
  const int total_frames = static_cast<int>(frames.size());
  std::cerr << "[ErasorFilter] iterating " << total_frames << " frames (skip=" << skip << ")" << std::endl;

  for (int fi = 0; fi < total_frames; fi++) {
    if (fi % (skip + 1) != 0) continue;
    const auto& fd = frames[fi];
    // Load query scan (sensor-local).
    std::vector<Eigen::Vector3f> scan_pts;
    if (!load_bin(fd.dir + "/points.bin", scan_pts, fd.num_points)) {
      frames_load_fail++;
      continue;
    }
    // Periodic progress so the UI doesn't look hung on big chunks.
    if (frames_processed > 0 && frames_processed % 50 == 0) {
      std::cerr << "[ErasorFilter] processing frame " << fi << "/" << total_frames
                << " (processed=" << frames_processed
                << ", dyn_votes=" << total_dyn_votes
                << ", stat_votes=" << total_stat_votes << ")" << std::endl;
    }

    // Query R-PoD: max-Z per polar bin in the SCAN's sensor frame. Both
    // query and map are gated by the sensor's V-FOV so tall structures the
    // LiDAR can't physically reach at a given range don't become false
    // dynamic candidates.
    const float v_fov_half_rad = params_.sensor_v_fov_half_deg * M_PI / 180.0f;
    const float tan_v_fov = std::tan(v_fov_half_rad);
    auto in_vfov = [tan_v_fov](float z, float r) {
      // r is horizontal distance in sensor XY plane; z is sensor-local height.
      // Point at angular elevation atan2(z, r) -- include if |z| <= r * tan(half).
      return std::abs(z) <= r * tan_v_fov;
    };
    std::vector<float> query_max_z(num_bins, -std::numeric_limits<float>::infinity());
    for (const auto& p : scan_pts) {
      const float r = std::sqrt(p.x() * p.x() + p.y() * p.y());
      if (!in_vfov(p.z(), r)) continue;
      const float theta = std::atan2(p.y(), p.x());
      const int b = polar_bin(r, theta, params_.min_range, params_.max_range,
                              params_.num_rings, params_.num_sectors);
      if (b < 0) continue;
      if (p.z() > query_max_z[b]) query_max_z[b] = p.z();
    }

    // Map R-VOI: radius search around sensor pose, transform to sensor frame.
    const Eigen::Vector3f sensor_pos = fd.T_world_lidar.translation().cast<float>();
    const float qpt[3] = {sensor_pos.x(), sensor_pos.y(), sensor_pos.z()};
    std::vector<std::pair<size_t, float>> matches;
    nanoflann::SearchParams sp;
    sp.sorted = false;
    kdtree.radiusSearch(qpt, max_range_sq, matches, sp);
    if (matches.empty()) { frames_no_matches++; continue; }
    frames_processed++;

    const Eigen::Isometry3f T_lidar_world = fd.T_world_lidar.inverse().cast<float>();

    // Map R-PoD + per-bin index list.
    std::vector<float> map_max_z(num_bins, -std::numeric_limits<float>::infinity());
    std::vector<std::vector<int>> map_bin_indices(num_bins);
    for (const auto& m : matches) {
      const int idx = static_cast<int>(m.first);
      const Eigen::Vector3f pl = T_lidar_world * world_points[idx];
      const float r = std::sqrt(pl.x() * pl.x() + pl.y() * pl.y());
      if (!in_vfov(pl.z(), r)) continue;   // mirror query V-FOV gate
      const float theta = std::atan2(pl.y(), pl.x());
      const int b = polar_bin(r, theta, params_.min_range, params_.max_range,
                              params_.num_rings, params_.num_sectors);
      if (b < 0) continue;
      if (pl.z() > map_max_z[b]) map_max_z[b] = pl.z();
      map_bin_indices[b].push_back(idx);
    }

    // Per-bin ratio test. Mirrors MapCleaner's 4-case logic:
    //   - both bins populated, ratio low      -> dynamic vote (object moved)
    //   - both bins populated, ratio ~1       -> static vote (still there)
    //   - query empty, map populated          -> SKIP (no info; bin likely
    //     outside V-FOV at this range. The original 360-spinner ERASOR
    //     treated this as dynamic; on a narrow-FOV sensor that produces
    //     'crazy segmentator' false positives on lampposts/signs/cars.)
    //   - map empty                           -> nothing to vote on
    for (int b = 0; b < num_bins; b++) {
      if (map_bin_indices[b].empty()) continue;
      const float mz = map_max_z[b];
      const float qz = query_max_z[b];
      const bool query_empty = (qz <= -std::numeric_limits<float>::infinity() * 0.5f);
      if (query_empty) continue;  // no information from this frame for this bin
      bool bin_dynamic = false;
      // Both bins have content. Compare Z-extent above their common floor.
      // qz/mz are sensor-local Z (negative for ground when sensor is up).
      // Anchor at min(mz, qz) so ratio measures how much TALLER the map is
      // vs query, not absolute heights (which can be negative).
      const float floor = std::min(mz, qz);
      const float qspan = std::max(0.01f, qz - floor);
      const float mspan = std::max(0.01f, mz - floor);
      const float ratio = qspan / mspan;
      if (ratio < params_.ratio_threshold) bin_dynamic = true;
      if (bin_dynamic) {
        for (int idx : map_bin_indices[b]) {
          if (params_.exclude_ground_pw && has_ground && is_ground[idx]) {
            static_votes[idx]++; total_stat_votes++;
          } else {
            dynamic_votes[idx]++; total_dyn_votes++;
          }
        }
      } else {
        for (int idx : map_bin_indices[b]) { static_votes[idx]++; total_stat_votes++; }
      }
    }
  }

  std::cerr << "[ErasorFilter] frames: processed=" << frames_processed
            << " load_fail=" << frames_load_fail << " no_matches=" << frames_no_matches
            << " | votes: dynamic=" << total_dyn_votes << " static=" << total_stat_votes << std::endl;

  for (size_t i = 0; i < world_points.size(); i++) {
    if (dynamic_votes[i] > static_votes[i]) {
      result.is_dynamic[i] = true;
      result.num_dynamic++;
    } else {
      result.num_static++;
    }
  }
  return result;
}

}  // namespace glim
