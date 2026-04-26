#include <glim/util/map_cleaner.hpp>
#include <glim/util/post_processing.hpp>
#include <patchwork/patchworkpp.h>

#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <fstream>
#include <queue>
#include <unordered_set>

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

  // Remove-Revert mode: two voting passes (coarse + fine), AND-merge.
  // Coarse uses inflated range_threshold and angular resolution -> high recall
  // (many flagged dynamic). Fine uses native params -> high precision. Result
  // = points that flagged dynamic in BOTH. Implements the Removert algorithmic
  // insight inside the same MapCleaner voting infra. We swap params_ for each
  // sub-pass and recurse with mode=Voting to share the existing loop body.
  if (params_.mode == Mode::RemoveRevert) {
    const Params orig = params_;
    // Coarse pass.
    params_.mode = Mode::Voting;
    params_.range_threshold = orig.range_threshold * std::max(1.0f, orig.coarse_thresh_mult);
    params_.res_h = orig.res_h * std::max(1.0f, orig.coarse_res_mult);
    params_.res_v = orig.res_v * std::max(1.0f, orig.coarse_res_mult);
    Result coarse = compute(frames, world_points, world_ranges, is_ground);
    // Fine pass with native params.
    params_.range_threshold = orig.range_threshold;
    params_.res_h = orig.res_h;
    params_.res_v = orig.res_v;
    Result fine = compute(frames, world_points, world_ranges, is_ground);
    // Restore.
    params_ = orig;
    // AND-merge: dynamic iff BOTH passes agree.
    Result merged;
    merged.is_dynamic.resize(world_points.size(), false);
    for (size_t i = 0; i < world_points.size(); i++) {
      if (coarse.is_dynamic[i] && fine.is_dynamic[i]) {
        merged.is_dynamic[i] = true;
        merged.num_dynamic++;
      } else {
        merged.num_static++;
      }
    }
    std::cerr << "[MapCleaner] Remove-Revert: coarse_dyn=" << coarse.num_dynamic
              << " fine_dyn=" << fine.num_dynamic
              << " merged_dyn=" << merged.num_dynamic << std::endl;
    return merged;
  }

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
  // Separately tracked: last frame that actually contributed a vote. Drives
  // the min_baseline_m gate -- forces parallax between voting frames so a
  // car driving in front of the rig at the same speed gets dynamic votes.
  Eigen::Vector3f last_voted_pos = Eigen::Vector3f::Constant(std::numeric_limits<float>::max());
  std::vector<int> submap_indices;  // indices into above_points
  std::vector<Eigen::Vector3f> submap_world_pts;

  int frames_processed = 0;
  for (int fi = 0; fi < static_cast<int>(frames.size()); fi++) {
    if (fi % (params_.frame_skip + 1) != 0) continue;

    const auto& fd = frames[fi];
    const Eigen::Vector3f sensor_pos_pre = fd.T_world_lidar.translation().cast<float>();
    // min_baseline gate: skip frames too close to the last voted one. First
    // frame always passes (last_voted_pos starts at +inf so distance > any).
    if (params_.min_baseline_m > 0.0f &&
        (sensor_pos_pre - last_voted_pos).norm() < params_.min_baseline_m) {
      continue;
    }

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
    last_voted_pos = sensor_pos;
  }

  // Final classification with robustness gates:
  //   - dynamic iff (vote_dynamic - vote_static > vote_margin)
  //   - additionally require vote_static >= min_static_votes-1 OR vote_dynamic
  //     to dominate by 2x (so a never-observed point with 0 static + 1 dynamic
  //     still gets flagged when min_static_votes is loose).
  // Ground points stay static (default false) since they were excluded from voting.
  const int vote_margin = std::max(0, params_.vote_margin);
  const int min_static  = std::max(1, params_.min_static_votes);
  std::vector<int> dynamic_above_idxs;  // for optional cluster verification below
  dynamic_above_idxs.reserve(above_points.size() / 10);
  for (size_t ai = 0; ai < above_points.size(); ai++) {
    const int orig_idx = above_to_orig[ai];
    const int vd = vote_dynamic[ai];
    const int vs = vote_static[ai];
    bool is_dyn = (vd - vs > vote_margin);
    // min_static_votes guard: if static vote count is below threshold, the
    // dynamic side needs to clearly dominate (>= 2x) to win. Skips the guard
    // entirely when no static obs exist (a never-observed point can still be
    // flagged dynamic if it accumulates dynamic votes).
    if (is_dyn && vs > 0 && vs < min_static && vd < 2 * vs) is_dyn = false;
    if (is_dyn) {
      result.is_dynamic[orig_idx] = true;
      dynamic_above_idxs.push_back(static_cast<int>(ai));
    }
  }

  // Optional cluster-size verification: revert isolated dynamic flags that
  // failed to cluster into a cohesive moving object. Same trick DynaBlox uses
  // at object level, lifted into MapCleaner so we don't depend on Voxblox.
  if (params_.min_dynamic_cluster_size > 1 && !dynamic_above_idxs.empty()) {
    const float vsz = std::max(0.05f, params_.dynamic_cluster_voxel);
    const float inv = 1.0f / vsz;
    auto vkey = [inv](const Eigen::Vector3f& p) {
      const int x = static_cast<int>(std::floor(p.x() * inv)) + 1048576;
      const int y = static_cast<int>(std::floor(p.y() * inv)) + 1048576;
      const int z = static_cast<int>(std::floor(p.z() * inv)) + 1048576;
      return (static_cast<uint64_t>(x) << 42) | (static_cast<uint64_t>(y) << 21) | static_cast<uint64_t>(z);
    };
    // Bucket dynamic points by voxel.
    std::unordered_map<uint64_t, std::vector<int>> voxbuckets;
    for (int ai : dynamic_above_idxs) voxbuckets[vkey(above_points[ai])].push_back(ai);
    // BFS over 26-neighbor connectivity.
    std::unordered_map<uint64_t, int> vox_cluster;
    std::vector<std::vector<uint64_t>> clusters;
    for (const auto& kv : voxbuckets) {
      if (vox_cluster.count(kv.first)) continue;
      const int cid = static_cast<int>(clusters.size());
      clusters.emplace_back();
      std::queue<uint64_t> q; q.push(kv.first); vox_cluster[kv.first] = cid;
      while (!q.empty()) {
        const uint64_t k = q.front(); q.pop();
        clusters[cid].push_back(k);
        const int x = static_cast<int>((k >> 42) & 0x1FFFFF);
        const int y = static_cast<int>((k >> 21) & 0x1FFFFF);
        const int z = static_cast<int>(k & 0x1FFFFF);
        for (int dz = -1; dz <= 1; dz++) for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
          if (!dx && !dy && !dz) continue;
          const uint64_t nk = (static_cast<uint64_t>(x + dx) << 42)
                            | (static_cast<uint64_t>(y + dy) << 21)
                            | static_cast<uint64_t>(z + dz);
          if (voxbuckets.count(nk) && !vox_cluster.count(nk)) {
            vox_cluster[nk] = cid;
            q.push(nk);
          }
        }
      }
    }
    // Revert clusters smaller than the threshold.
    int reverted_clusters = 0, reverted_points = 0;
    for (const auto& cluster : clusters) {
      int pts = 0;
      for (uint64_t k : cluster) pts += static_cast<int>(voxbuckets[k].size());
      if (pts >= params_.min_dynamic_cluster_size) continue;
      reverted_clusters++;
      for (uint64_t k : cluster) {
        for (int ai : voxbuckets[k]) {
          const int orig_idx = above_to_orig[ai];
          result.is_dynamic[orig_idx] = false;
          reverted_points++;
        }
      }
    }
    if (reverted_clusters > 0) {
      // Logged via std::cerr -- map_cleaner.cpp doesn't carry an spdlog logger.
      std::cerr << "[MapCleaner] cluster verify: reverted " << reverted_points
                << " points across " << reverted_clusters << " small clusters (min size = "
                << params_.min_dynamic_cluster_size << ")" << std::endl;
    }
  }

  // Tally final counts.
  for (size_t i = 0; i < world_points.size(); i++) {
    if (result.is_dynamic[i]) result.num_dynamic++;
    else                       result.num_static++;
  }
  return result;
}

}  // namespace glim
