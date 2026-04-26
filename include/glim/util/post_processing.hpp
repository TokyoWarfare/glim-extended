#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <boost/filesystem.hpp>
#include <nlohmann/json.hpp>

namespace glim {

// ---------------------------------------------------------------------------
// Voxel key encoding — packs 3 ints into 64-bit key (21 bits each, ±1M voxels)
// ---------------------------------------------------------------------------

inline uint64_t voxel_key(int vx, int vy, int vz) {
  return (static_cast<uint64_t>(vx + 1048576) << 42) |
         (static_cast<uint64_t>(vy + 1048576) << 21) |
         static_cast<uint64_t>(vz + 1048576);
}

inline uint64_t voxel_key(const Eigen::Vector3f& point, float inv_resolution) {
  const int vx = static_cast<int>(std::floor(point.x() * inv_resolution));
  const int vy = static_cast<int>(std::floor(point.y() * inv_resolution));
  const int vz = static_cast<int>(std::floor(point.z() * inv_resolution));
  return voxel_key(vx, vy, vz);
}

inline uint64_t voxel_key(const Eigen::Vector3d& point, double inv_resolution) {
  const int vx = static_cast<int>(std::floor(point.x() * inv_resolution));
  const int vy = static_cast<int>(std::floor(point.y() * inv_resolution));
  const int vz = static_cast<int>(std::floor(point.z() * inv_resolution));
  return voxel_key(vx, vy, vz);
}

// ---------------------------------------------------------------------------
// Chunk — path-aligned spatial region for chunked processing
// ---------------------------------------------------------------------------

struct Chunk {
  Eigen::Vector3d center;
  Eigen::Matrix3d R_world_chunk;   // chunk-local to world rotation
  Eigen::Matrix3d R_chunk_world;   // world to chunk-local rotation
  double half_size;
  double half_height;

  /// Test if a world-space point is inside this chunk
  bool contains(const Eigen::Vector3d& world_point) const {
    const Eigen::Vector3d local = R_chunk_world * (world_point - center);
    return std::abs(local.x()) <= half_size &&
           std::abs(local.y()) <= half_size &&
           std::abs(local.z()) <= half_height;
  }

  bool contains(const Eigen::Vector3f& world_point) const {
    const Eigen::Vector3d wp = world_point.cast<double>();
    const Eigen::Vector3d local = R_chunk_world * (wp - center);
    return std::abs(local.x()) <= half_size &&
           std::abs(local.y()) <= half_size &&
           std::abs(local.z()) <= half_height;
  }

  /// Compute world-aligned AABB for this chunk (for bbox intersection tests)
  Eigen::AlignedBox3d world_aabb() const {
    Eigen::AlignedBox3d box;
    const Eigen::Vector3d half(half_size, half_size, half_height);
    for (int ci = 0; ci < 8; ci++) {
      Eigen::Vector3d local(
        (ci & 1) ? half.x() : -half.x(),
        (ci & 2) ? half.y() : -half.y(),
        (ci & 4) ? half.z() : -half.z());
      box.extend(center + R_world_chunk * local);
    }
    return box;
  }
};

/// Build path-aligned chunks along a trajectory.
/// @param trajectory_points  Vector of (position, cumulative_distance) pairs
/// @param total_dist          Total trajectory distance
/// @param chunk_spacing       Distance between chunk centers (metres)
/// @param chunk_half_size     Half-size of each chunk (metres)
/// @param chunk_half_height   Half-height (default 50m)
template <typename TrajectoryContainer>
std::vector<Chunk> build_chunks(
    const TrajectoryContainer& trajectory,
    double total_dist,
    double chunk_spacing,
    double chunk_half_size,
    double chunk_half_height = 50.0) {
  std::vector<Chunk> chunks;
  for (double d = 0.0; d < total_dist; d += chunk_spacing) {
    size_t idx = 0;
    for (size_t k = 1; k < trajectory.size(); k++) {
      if (trajectory[k].cumulative_dist >= d) { idx = k; break; }
    }
    const Eigen::Vector3d c = trajectory[idx].pose.translation();
    const size_t next = std::min(idx + 1, trajectory.size() - 1);
    Eigen::Vector3d fwd = trajectory[next].pose.translation() - trajectory[idx].pose.translation();
    fwd.z() = 0.0;
    if (fwd.norm() < 0.01) fwd = Eigen::Vector3d::UnitX();
    else fwd.normalize();
    const Eigen::Vector3d up = Eigen::Vector3d::UnitZ();
    const Eigen::Vector3d right = fwd.cross(up).normalized();
    Eigen::Matrix3d R;
    R.col(0) = fwd;
    R.col(1) = right;
    R.col(2) = up;
    chunks.push_back({c, R, R.transpose(), chunk_half_size, chunk_half_height});
  }
  return chunks;
}

// ---------------------------------------------------------------------------
// FrameInfo — HD frame metadata with world-space bounding box
// ---------------------------------------------------------------------------

struct FrameInfo {
  std::string dir;
  Eigen::Isometry3d T_world_lidar;
  Eigen::AlignedBox3d world_bbox;
  int num_points;
  int submap_id = -1;     // which submap this frame belongs to
  int session_id = -1;    // which session
  double stamp = 0.0;     // frame timestamp
};

/// Read frame_meta.json and compute world-space bbox using the given optimized pose.
/// Returns nullopt-equivalent (num_points=0) on failure.
inline FrameInfo frame_info_from_meta(const std::string& frame_dir, const Eigen::Isometry3d& T_world_lidar, int submap_id = -1, int session_id = -1) {
  FrameInfo fi;
  fi.dir = frame_dir;
  fi.T_world_lidar = T_world_lidar;
  fi.num_points = 0;
  fi.submap_id = submap_id;
  fi.session_id = session_id;

  const std::string meta_path = frame_dir + "/frame_meta.json";
  if (!boost::filesystem::exists(meta_path)) return fi;

  std::ifstream meta_ifs(meta_path);
  const auto meta = nlohmann::json::parse(meta_ifs, nullptr, false);
  if (meta.is_discarded()) return fi;

  fi.num_points = meta.value("num_points", 0);
  fi.stamp = meta.value("stamp", 0.0);

  // Compute world bbox from frame_meta's local bbox + optimized pose
  if (meta.contains("bbox_world_min") && meta.contains("bbox_world_max") && meta.contains("T_world_lidar")) {
    const auto& bmin_j = meta["bbox_world_min"];
    const auto& bmax_j = meta["bbox_world_max"];
    const Eigen::Vector3d local_min(bmin_j[0].get<double>() - meta["T_world_lidar"][3].get<double>(),
                                     bmin_j[1].get<double>() - meta["T_world_lidar"][7].get<double>(),
                                     bmin_j[2].get<double>() - meta["T_world_lidar"][11].get<double>());
    const Eigen::Vector3d local_max(bmax_j[0].get<double>() - meta["T_world_lidar"][3].get<double>(),
                                     bmax_j[1].get<double>() - meta["T_world_lidar"][7].get<double>(),
                                     bmax_j[2].get<double>() - meta["T_world_lidar"][11].get<double>());
    for (int ci = 0; ci < 8; ci++) {
      Eigen::Vector3d corner(
        (ci & 1) ? local_max.x() : local_min.x(),
        (ci & 2) ? local_max.y() : local_min.y(),
        (ci & 4) ? local_max.z() : local_min.z());
      fi.world_bbox.extend(T_world_lidar * corner);
    }
  } else {
    const Eigen::Vector3d pos = T_world_lidar.translation();
    fi.world_bbox.extend(pos - Eigen::Vector3d::Constant(200.0));
    fi.world_bbox.extend(pos + Eigen::Vector3d::Constant(200.0));
  }

  return fi;
}

// ---------------------------------------------------------------------------
// HD frame binary loading helpers
// ---------------------------------------------------------------------------

/// Load a binary attribute file into a pre-allocated vector.
/// Returns true if the file was read successfully.
/// NOTE: Doesn't validate actual file size vs sizeof(T) * expected_count -- a
/// mismatched file will read partial / garbage data silently. For runtime-
/// chosen attribute names prefer `load_hd_scalar`, which dispatches on file
/// size and handles the "fake aux" fields (gps_time, intensity, range) that
/// aren't stored as aux_<name>.bin.
template <typename T>
bool load_bin(const std::string& path, std::vector<T>& out, int expected_count) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  out.resize(expected_count);
  f.read(reinterpret_cast<char*>(out.data()), sizeof(T) * expected_count);
  return true;
}

/// Load a per-point visibility mask (1 = visible, 0 = hidden) from an HD
/// frame directory. Returns an all-1s vector of size `num_points` when the
/// file is missing OR the file size doesn't match -- every HD consumer can
/// call this unconditionally and treat "no file" as "everything visible".
/// File format: one `uint8_t` per point, stored raw (no header). Overwritten
/// on every scalar-filter Apply; a Clear visibility wipe removes the files.
inline std::vector<uint8_t> load_hd_visibility(const std::string& frame_dir, int num_points) {
  std::vector<uint8_t> out(num_points, 1);
  if (num_points <= 0) return out;
  const std::string path = frame_dir + "/aux_visibility.bin";
  if (!boost::filesystem::exists(path)) return out;
  if (boost::filesystem::file_size(path) != static_cast<uintmax_t>(num_points)) return out;
  std::ifstream f(path, std::ios::binary);
  if (!f) return out;
  f.read(reinterpret_cast<char*>(out.data()), num_points);
  return out;
}

/// Write a per-point visibility mask. Overwrites. Returns false on I/O fail.
inline bool write_hd_visibility(const std::string& frame_dir, const std::vector<uint8_t>& vis) {
  std::ofstream f(frame_dir + "/aux_visibility.bin", std::ios::binary);
  if (!f) return false;
  f.write(reinterpret_cast<const char*>(vis.data()), vis.size());
  return static_cast<bool>(f);
}

/// Load a per-point scalar from an HD frame directory by attribute name.
/// Handles the "fake aux" fields that the UI lists alongside real aux files
/// but which aren't stored as aux_<name>.bin:
///   - "intensity" -> intensities.bin (float per point)
///   - "range"     -> range.bin       (float per point)
///   - "gps_time"  -> times.bin (float per-point offset) + frame_stamp - base
/// Real aux fields dispatch by on-disk file size to avoid the float-vs-double
/// garbage-read trap `load_bin` has.
///
/// Returns a vector of size `num_points` on success, empty on failure (file
/// missing or size doesn't match sizeof(float)*n nor sizeof(double)*n).
/// Designed for tools whose attribute name isn't known at compile time
/// (scalar filter, any future "pick a scalar" UI). Tools that load a known
/// file can keep using `load_bin` directly.
inline std::vector<float> load_hd_scalar(
  const std::string& frame_dir,
  const std::string& attr_name,
  double frame_stamp,
  double base,
  int num_points) {
  std::vector<float> out;
  if (num_points <= 0) return out;

  if (attr_name == "intensity") {
    if (!load_bin(frame_dir + "/intensities.bin", out, num_points)) out.clear();
    return out;
  }
  if (attr_name == "range") {
    if (!load_bin(frame_dir + "/range.bin", out, num_points)) out.clear();
    return out;
  }
  if (attr_name == "gps_time") {
    // Per-point offset + frame stamp; no aux_gps_time.bin on HD frames.
    std::vector<float> offsets(num_points, 0.0f);
    std::ifstream tf(frame_dir + "/times.bin", std::ios::binary);
    if (!tf) return out;  // empty -> caller knows
    tf.read(reinterpret_cast<char*>(offsets.data()), sizeof(float) * num_points);
    out.resize(num_points);
    const double base_offset = frame_stamp - base;
    for (int i = 0; i < num_points; i++) out[i] = static_cast<float>(base_offset) + offsets[i];
    return out;
  }

  // Real aux: dispatch on actual file size.
  const std::string ap = frame_dir + "/aux_" + attr_name + ".bin";
  if (!boost::filesystem::exists(ap)) return out;
  const auto fsize = boost::filesystem::file_size(ap);
  if (fsize == sizeof(float) * static_cast<size_t>(num_points)) {
    if (!load_bin(ap, out, num_points)) out.clear();
  } else if (fsize == sizeof(double) * static_cast<size_t>(num_points)) {
    std::vector<double> dbuf;
    if (load_bin(ap, dbuf, num_points)) {
      out.resize(num_points);
      for (int i = 0; i < num_points; i++) out[i] = static_cast<float>(dbuf[i] - base);
    }
  }
  // else: file exists but size matches neither float nor double -> leave empty.
  return out;
}

/// Filter and rewrite a binary attribute file, keeping only specified indices.
inline void filter_bin_file(const std::string& path, size_t elem_size, int num_pts,
                            const std::vector<int>& kept_indices, int new_count) {
  if (!boost::filesystem::exists(path)) return;
  std::vector<char> src(num_pts * elem_size);
  { std::ifstream f(path, std::ios::binary); f.read(src.data(), src.size()); }
  std::vector<char> dst(new_count * elem_size);
  for (int j = 0; j < new_count; j++) {
    std::memcpy(dst.data() + j * elem_size, src.data() + kept_indices[j] * elem_size, elem_size);
  }
  { std::ofstream f(path, std::ios::binary); f.write(dst.data(), dst.size()); }
}

// ---------------------------------------------------------------------------
// Team grouping — split sorted IDs by sequence continuity
// ---------------------------------------------------------------------------

/// Group a set of integer IDs into teams based on sequence continuity.
/// IDs with gaps > max_gap are split into separate teams.
/// Returns teams sorted by size (largest first).
inline std::vector<std::vector<int>> group_by_continuity(const std::unordered_set<int>& ids, int max_gap = 2) {
  if (ids.empty()) return {};
  std::vector<int> sorted(ids.begin(), ids.end());
  std::sort(sorted.begin(), sorted.end());
  std::vector<std::vector<int>> teams;
  teams.push_back({sorted[0]});
  for (size_t i = 1; i < sorted.size(); i++) {
    if (sorted[i] - sorted[i - 1] > max_gap) teams.push_back({});
    teams.back().push_back(sorted[i]);
  }
  std::sort(teams.begin(), teams.end(), [](const auto& a, const auto& b) { return a.size() > b.size(); });
  return teams;
}

/// Overload for vector input
inline std::vector<std::vector<int>> group_by_continuity(const std::vector<int>& ids, int max_gap = 2) {
  std::unordered_set<int> id_set(ids.begin(), ids.end());
  return group_by_continuity(id_set, max_gap);
}

// ---------------------------------------------------------------------------
// Submap frame world pose computation
// ---------------------------------------------------------------------------

/// Compute the optimized world-to-lidar transform for a frame within a submap.
inline Eigen::Isometry3d compute_frame_world_pose(
    const Eigen::Isometry3d& T_world_origin,
    const Eigen::Isometry3d& T_origin_endpoint_L,
    const Eigen::Isometry3d& T_odom_imu_first,
    const Eigen::Isometry3d& T_world_imu_frame,
    const Eigen::Isometry3d& T_lidar_imu) {
  const Eigen::Isometry3d T_ep = T_world_origin * T_origin_endpoint_L;
  const Eigen::Isometry3d T_w_imu = T_ep * T_odom_imu_first.inverse() * T_world_imu_frame;
  return T_w_imu * T_lidar_imu.inverse();
}

// ---------------------------------------------------------------------------
// RangeImage — spherical projection for range comparison / panorama generation
// Reusable for: dynamic object detection, 360° panorama rendering, segmentation
// ---------------------------------------------------------------------------

struct RangeImage {
  int az_bins = 720;     // azimuth bins (default 0.5° resolution)
  int el_bins = 360;     // elevation bins (default 0.5° resolution)

  Eigen::Vector3f sensor_pos = Eigen::Vector3f::Zero();
  Eigen::Matrix3f R_sensor_world = Eigen::Matrix3f::Identity();
  std::vector<float> ranges;  // el_bins × az_bins, stores max range per cell

  RangeImage(int az = 720, int el = 360) : az_bins(az), el_bins(el), ranges(az * el, 0.0f) {}

  void set_pose(const Eigen::Isometry3d& T_world_sensor) {
    sensor_pos = T_world_sensor.translation().cast<float>();
    R_sensor_world = T_world_sensor.rotation().cast<float>().transpose();
  }

  /// Convert a sensor-local 3D point to (az_bin, el_bin)
  std::pair<int, int> project(const Eigen::Vector3f& local_pt) const {
    const float r_xy = std::sqrt(local_pt.x() * local_pt.x() + local_pt.y() * local_pt.y());
    const float az = std::atan2(local_pt.y(), local_pt.x());
    const float el = std::atan2(local_pt.z(), r_xy);
    const int az_bin = std::clamp(static_cast<int>((az + M_PI) / (2.0 * M_PI) * az_bins), 0, az_bins - 1);
    const int el_bin = std::clamp(static_cast<int>((el + M_PI / 2.0) / M_PI * el_bins), 0, el_bins - 1);
    return {az_bin, el_bin};
  }

  /// Add a point in sensor-local coordinates with its range
  void add_point(const Eigen::Vector3f& local_pt, float range) {
    if (range < 1.5f) return;
    const auto [az_bin, el_bin] = project(local_pt);
    float& cell = ranges[el_bin * az_bins + az_bin];
    cell = std::max(cell, range);
  }

  /// Query the measured range in the direction of a world-space point
  float query_range(const Eigen::Vector3f& world_point) const {
    const Eigen::Vector3f local = R_sensor_world * (world_point - sensor_pos);
    const auto [az_bin, el_bin] = project(local);
    return ranges[el_bin * az_bins + az_bin];
  }

  /// Query with neighborhood (check ±delta_h, ±delta_v bins). Returns max range in window.
  float query_range_neighborhood(const Eigen::Vector3f& world_point, int delta_h, int delta_v) const {
    const Eigen::Vector3f local = R_sensor_world * (world_point - sensor_pos);
    const auto [az_center, el_center] = project(local);
    float max_range = 0.0f;
    for (int dv = -delta_v; dv <= delta_v; dv++) {
      const int ev = el_center + dv;
      if (ev < 0 || ev >= el_bins) continue;
      for (int dh = -delta_h; dh <= delta_h; dh++) {
        int ah = (az_center + dh) % az_bins;
        if (ah < 0) ah += az_bins;
        max_range = std::max(max_range, ranges[ev * az_bins + ah]);
      }
    }
    return max_range;
  }

  void clear() { std::fill(ranges.begin(), ranges.end(), 0.0f); }
};

}  // namespace glim
