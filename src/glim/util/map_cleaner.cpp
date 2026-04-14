#include <glim/util/map_cleaner.hpp>
#include <glim/util/post_processing.hpp>
#include <patchwork/patchworkpp.h>

#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <fstream>

// Simple KD-tree for radius search (nanoflann-style, minimal)
#include <nanoflann.hpp>

namespace glim {

namespace {

// nanoflann adaptor for vector of Vector3f
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

}  // namespace

void MapCleanerFilter::build_range_image(
    const std::vector<Eigen::Vector3f>& local_points,
    const std::vector<float>& ranges,
    int num_points) {
  // Compute range image dimensions
  const float fov_h = std::min(params_.fov_h, static_cast<float>(2.0 * M_PI));
  const float fov_v = std::min(params_.fov_v, static_cast<float>(M_PI));

  if (params_.res_h < params_.res_v) {
    res_h_scale_ = 1.0f;
    res_v_scale_ = params_.res_h / params_.res_v;
  } else {
    res_h_scale_ = params_.res_v / params_.res_h;
    res_v_scale_ = 1.0f;
  }

  spinning_ = (2.0f * M_PI - fov_h) < params_.res_h;

  // Image dimensions: FOV / resolution (in scaled space)
  im_width_ = static_cast<int>(std::ceil(fov_h * res_h_scale_ / std::min(params_.res_h, params_.res_v)));
  im_height_ = static_cast<int>(std::ceil(fov_v * res_v_scale_ / std::min(params_.res_h, params_.res_v)));

  // Allocate and clear
  range_image_.assign(im_width_ * im_height_, std::numeric_limits<float>::quiet_NaN());

  // Populate
  for (int i = 0; i < num_points; i++) {
    if (ranges[i] < params_.min_range) continue;
    const auto& p = local_points[i];
    if (!std::isfinite(p.x()) || !std::isfinite(p.y()) || !std::isfinite(p.z())) continue;

    const float range = ranges[i];
    const float r_xy = std::sqrt(p.x() * p.x() + p.y() * p.y());
    const float az = std::atan2(p.y(), p.x()) * res_h_scale_;
    const float el = std::atan2(p.z(), r_xy) * res_v_scale_;

    // Map to pixel coordinates
    const int col = static_cast<int>((az / (fov_h * res_h_scale_) + 0.5f) * im_width_);
    const int row = static_cast<int>((el / (fov_v * res_v_scale_) + 0.5f) * im_height_);

    if (col < 0 || col >= im_width_ || row < 0 || row >= im_height_) continue;

    float& cell = range_image_[row * im_width_ + col];
    if (!std::isfinite(cell) || range < cell) {
      cell = range;  // store CLOSEST range (not max — we want the first surface hit)
    }
  }
}

void MapCleanerFilter::compare_and_vote(
    const std::vector<Eigen::Vector3f>& submap_local_pts,
    const std::vector<int>& submap_indices,
    std::vector<int>& vote_static,
    std::vector<int>& vote_dynamic) {
  const float fov_h = std::min(params_.fov_h, static_cast<float>(2.0 * M_PI));
  const float fov_v = std::min(params_.fov_v, static_cast<float>(M_PI));

  #pragma omp parallel for schedule(guided, 64)
  for (int i = 0; i < static_cast<int>(submap_local_pts.size()); i++) {
    const auto& p = submap_local_pts[i];
    const float r_xy = std::sqrt(p.x() * p.x() + p.y() * p.y());
    const float target_range = p.norm();
    const float az = std::atan2(p.y(), p.x()) * res_h_scale_;
    const float el = std::atan2(p.z(), r_xy) * res_v_scale_;

    const int center_col = static_cast<int>((az / (fov_h * res_h_scale_) + 0.5f) * im_width_);
    const int center_row = static_cast<int>((el / (fov_v * res_v_scale_) + 0.5f) * im_height_);

    if (center_col < 0 || center_col >= im_width_ || center_row < 0 || center_row >= im_height_) continue;

    // Neighborhood voting (fused result from MapCleaner)
    bool has_case_a = false;
    bool has_case_c = false;

    for (int dv = -params_.delta_v; dv <= params_.delta_v; dv++) {
      const int row = center_row + dv;
      if (row < 0 || row >= im_height_) continue;

      for (int dh = -params_.delta_h; dh <= params_.delta_h; dh++) {
        int col = center_col + dh;
        if (col < 0) {
          if (spinning_) col += im_width_; else continue;
        }
        if (col >= im_width_) {
          if (spinning_) col -= im_width_; else continue;
        }

        const float scan_range = range_image_[row * im_width_ + col];
        if (!std::isfinite(scan_range)) continue;

        // MapCleaner 4-case comparison:
        // scan_range = what the scan measured, target_range = distance from sensor to map point
        // CASE_A: |diff| <= threshold → ranges match → static
        // CASE_B: target behind scan surface (scan_range < target - threshold) → no info → skip
        // CASE_C: scan sees THROUGH map point (scan_range > target + threshold) → free space → dynamic
        if (std::abs(scan_range - target_range) <= params_.range_threshold) {
          has_case_a = true;   // CASE_A: match → static
        } else if (scan_range > target_range + params_.range_threshold) {
          has_case_c = true;   // CASE_C: scan saw beyond → dynamic
        }
      }
    }

    // Fused result: CASE_A takes priority (if any match found → static vote)
    const int idx = submap_indices[i];
    if (has_case_a) {
      #pragma omp atomic
      vote_static[idx]++;
    } else if (has_case_c) {
      #pragma omp atomic
      vote_dynamic[idx]++;
    }
  }
}

// Static PatchWork++ params — exposed via UI
static patchwork::Params s_pw_params;
static bool s_pw_params_initialized = false;

patchwork::Params& MapCleanerFilter::getPatchWorkParams() {
  if (!s_pw_params_initialized) {
    s_pw_params.verbose = false;
    s_pw_params.enable_RNR = false;
    s_pw_params.sensor_height = 1.723;
    s_pw_params.max_range = 80.0;
    s_pw_params.min_range = 2.0;
    s_pw_params_initialized = true;
  }
  return s_pw_params;
}

std::vector<bool> MapCleanerFilter::classify_ground_patchwork(
    const std::vector<Eigen::Vector3f>& local_points, int num_points, float sensor_height,
    const std::vector<float>& intensities) {
  std::vector<bool> is_ground(num_points, false);
  auto& pw_params = getPatchWorkParams();
  pw_params.sensor_height = sensor_height;

  const bool has_intensity = pw_params.enable_RNR && !intensities.empty() && static_cast<int>(intensities.size()) >= num_points;
  const int cols = has_intensity ? 4 : 3;

  // Build Eigen matrix for PatchWork++
  Eigen::MatrixXf cloud(num_points, cols);
  int valid = 0;
  std::vector<int> valid_to_orig(num_points);
  for (int i = 0; i < num_points; i++) {
    const auto& p = local_points[i];
    if (!std::isfinite(p.x()) || !std::isfinite(p.y()) || !std::isfinite(p.z())) continue;
    cloud(valid, 0) = p.x();
    cloud(valid, 1) = p.y();
    cloud(valid, 2) = p.z();
    if (has_intensity) cloud(valid, 3) = intensities[i];
    valid_to_orig[valid] = i;
    valid++;
  }
  cloud.conservativeResize(valid, cols);

  // Fresh instance per call (PatchWork++ accumulates internal state)
  patchwork::PatchWorkpp pw(pw_params);
  pw.estimateGround(cloud);

  // Map ground indices back to original
  Eigen::VectorXi ground_indices = pw.getGroundIndices();
  for (int i = 0; i < ground_indices.size(); i++) {
    const int pw_idx = ground_indices[i];
    if (pw_idx >= 0 && pw_idx < valid) {
      is_ground[valid_to_orig[pw_idx]] = true;
    }
  }

  return is_ground;
}

MapCleanerFilter::Result MapCleanerFilter::compute(
    const std::vector<FrameData>& frames,
    const std::vector<Eigen::Vector3f>& world_points,
    const std::vector<float>& world_ranges,
    const std::vector<bool>& is_ground) {

  Result result;
  result.is_dynamic.resize(world_points.size(), false);
  const bool has_ground_flags = !is_ground.empty() && is_ground.size() == world_points.size();

  if (world_points.empty() || frames.empty()) return result;

  // Filter: build a non-ground subset for KD-tree and voting
  // Ground points are excluded entirely from the voting cloud (like MapCleaner's ground_above approach)
  std::vector<Eigen::Vector3f> above_points;
  std::vector<int> above_to_orig;  // maps above_points index → original world_points index
  if (has_ground_flags) {
    above_points.reserve(world_points.size());
    above_to_orig.reserve(world_points.size());
    for (size_t i = 0; i < world_points.size(); i++) {
      if (!is_ground[i]) {
        above_to_orig.push_back(static_cast<int>(i));
        above_points.push_back(world_points[i]);
      }
    }
  } else {
    above_points = world_points;
    above_to_orig.resize(world_points.size());
    std::iota(above_to_orig.begin(), above_to_orig.end(), 0);
  }

  // Build KD-tree on non-ground points only
  PointCloudAdaptor adaptor(above_points);
  KDTree3f kdtree(3, adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10));
  kdtree.buildIndex();

  const float lidar_range_sq = params_.lidar_range * params_.lidar_range;

  // Vote lists (indexed by above_points, not world_points)
  std::vector<int> vote_static(above_points.size(), 0);
  std::vector<int> vote_dynamic(above_points.size(), 0);

  Eigen::Vector3f last_sensor_pos = Eigen::Vector3f::Constant(std::numeric_limits<float>::max());
  std::vector<int> submap_indices;  // indices into above_points
  std::vector<Eigen::Vector3f> submap_world_pts;

  int frames_processed = 0;
  for (int fi = 0; fi < static_cast<int>(frames.size()); fi++) {
    if (fi % (params_.frame_skip + 1) != 0) continue;

    const auto& fd = frames[fi];

    // Load raw scan data (sensor-local points + ranges + optionally normals)
    std::vector<Eigen::Vector3f> scan_pts;
    std::vector<float> scan_ranges;
    if (!load_bin(fd.dir + "/points.bin", scan_pts, fd.num_points)) continue;
    if (!load_bin(fd.dir + "/range.bin", scan_ranges, fd.num_points)) continue;

    // Always build range image from ALL points (ground included for stable static votes)
    build_range_image(scan_pts, scan_ranges, fd.num_points);

    // Update submap if sensor moved enough
    const Eigen::Vector3f sensor_pos = fd.T_world_lidar.translation().cast<float>();
    if (submap_world_pts.empty() || (sensor_pos - last_sensor_pos).norm() > params_.submap_update_dist) {
      // Radius search around sensor
      submap_indices.clear();
      submap_world_pts.clear();
      const float query_pt[3] = {sensor_pos.x(), sensor_pos.y(), sensor_pos.z()};

      std::vector<std::pair<size_t, float>> matches;
      nanoflann::SearchParams search_params;
      search_params.sorted = false;
      kdtree.radiusSearch(query_pt, lidar_range_sq, matches, search_params);

      submap_indices.reserve(matches.size());
      submap_world_pts.reserve(matches.size());
      for (const auto& m : matches) {
        const int idx = static_cast<int>(m.first);
        submap_indices.push_back(idx);
        submap_world_pts.push_back(above_points[idx]);
      }
      last_sensor_pos = sensor_pos;
    }

    // Transform submap points to scan's local frame
    const Eigen::Isometry3f T_lidar_world = fd.T_world_lidar.inverse().cast<float>();
    std::vector<Eigen::Vector3f> submap_local(submap_world_pts.size());
    for (size_t i = 0; i < submap_world_pts.size(); i++) {
      submap_local[i] = T_lidar_world * submap_world_pts[i];
    }

    // Compare and vote
    compare_and_vote(submap_local, submap_indices, vote_static, vote_dynamic);
    frames_processed++;
  }

  // Final classification
  // Ground points are already excluded from voting — they stay as static (default false)
  // Map above_points votes back to world_points
  for (size_t ai = 0; ai < above_points.size(); ai++) {
    const int orig_idx = above_to_orig[ai];
    if (vote_dynamic[ai] > vote_static[ai]) {
      result.is_dynamic[orig_idx] = true;
      result.num_dynamic++;
    } else {
      result.num_static++;
    }
  }
  // Count ground points as static
  if (has_ground_flags) {
    for (size_t i = 0; i < world_points.size(); i++) {
      if (is_ground[i]) result.num_static++;
    }
  }

  return result;
}

}  // namespace glim
