#pragma once

#include <atomic>
#include <thread>
#include <string>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <algorithm>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <spdlog/spdlog.h>
#include <portable-file-dialogs.h>

#include <glim/util/config.hpp>
#include <glim/util/logging.hpp>
#include <glim/util/concurrent_vector.hpp>
#include <glim/odometry/estimation_frame.hpp>
#include <glim/odometry/callbacks.hpp>
#include <glim/common/cloud_deskewing.hpp>
#include <glim/common/cloud_covariance_estimation.hpp>

#include <gtsam_points/ann/kdtree.hpp>

#ifdef GLIM_ROS2
#include <glim/util/extension_module_ros2.hpp>
using HDFrameSaverBase = glim::ExtensionModuleROS2;
#else
#include <glim/util/extension_module_ros.hpp>
using HDFrameSaverBase = glim::ExtensionModuleROS;
#endif

namespace glim {

namespace hd_detail {

template <typename T>
bool write_binary(const std::string& path, const T* data, size_t bytes) {
  std::ofstream ofs(path, std::ios::binary);
  if (!ofs) return false;
  ofs.write(reinterpret_cast<const char*>(data), bytes);
  return ofs.good();
}

inline std::vector<int> find_knn(const Eigen::Vector4d* points, int num_points, int k, int num_threads) {
  gtsam_points::KdTree tree(points, num_points);
  std::vector<int> neighbors(num_points * k);

#pragma omp parallel for num_threads(num_threads) schedule(guided, 8)
  for (int i = 0; i < num_points; i++) {
    std::vector<size_t> k_indices(k, i);
    std::vector<double> k_sq_dists(k);
    tree.knn_search(points[i].data(), k, k_indices.data(), k_sq_dists.data());
    std::copy(k_indices.begin(), k_indices.begin() + k, neighbors.begin() + i * k);
  }
  return neighbors;
}

}  // namespace hd_detail

/**
 * @brief Saves deskewed full-resolution frames with per-point attributes during SLAM.
 *
 * Hooks OdometryEstimationCallbacks::on_new_frame to capture each odometry frame
 * at full sensor resolution (raw points). Deskews using the IMU-rate trajectory,
 * computes normals with viewpoint-consistent orientation, and saves:
 *   points.bin, normals.bin, intensities.bin, times.bin, range.bin, rings.bin, frame_meta.json
 *
 * Automatically forces keep_raw_points=true in GlobalConfig to ensure raw points
 * are available.
 */
class HDFrameSaver : public HDFrameSaverBase {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  HDFrameSaver() : logger(create_module_logger("hd_frame_saver")) {
    logger->info("Initializing HD frame saver");

    // Check that keep_raw_points is enabled — HD frames need raw sensor data
    Config config_ros(GlobalConfig::get_config_path("config_ros"));
    const bool keep_raw = config_ros.param<bool>("glim_ros", "keep_raw_points", false);
    if (!keep_raw) {
      logger->error("[HD] WARNING: keep_raw_points is FALSE in config_ros.json!");
      logger->error("[HD] HD frames will contain downsampled data, not full-resolution sensor data.");
      logger->error("[HD] Set keep_raw_points: true in config_ros.json for true HD export.");
      // Non-blocking popup — SLAM continues but user is warned
      std::thread([] {
        pfd::message("HD Frame Saver Warning",
          "keep_raw_points is set to FALSE in config_ros.json.\n\n"
          "HD frames will be saved at reduced resolution (preprocessed data).\n"
          "For full-resolution HD export, set keep_raw_points: true\n"
          "in your config_ros.json and restart SLAM.\n\n"
          "SLAM will continue normally.",
          pfd::choice::ok, pfd::icon::warning);
      }).detach();
    }

    // Read configuration
    const std::string config_path = GlobalConfig::get_config_path("config_hd_frame_saver");
    Config hd_config(config_path);
    output_path = hd_config.param<std::string>("hd_frames", "output_path", "");
    num_threads = hd_config.param<int>("hd_frames", "num_threads", 4);
    k_neighbors = hd_config.param<int>("hd_frames", "k_neighbors", 10);

    if (output_path.empty()) {
      output_path = "/tmp/dump/hd_frames";
    }
    logger->info("[HD] Output path: {}", output_path);
    logger->info("[HD] Normal estimation: k={}, threads={}", k_neighbors, num_threads);

    OdometryEstimationCallbacks::on_new_frame.add(
      [this](const EstimationFrame::ConstPtr& frame) { frame_queue.push_back(frame); });

    kill_switch = false;
    thread = std::thread([this] { save_task(); });
  }

  ~HDFrameSaver() {
    kill_switch = true;
    thread.join();
    logger->info("[HD] Frame saver stopped. Frames saved to {}", output_path);
  }

  virtual std::vector<GenericTopicSubscription::Ptr> create_subscriptions() override { return {}; }

private:
  void save_task() {
    while (!kill_switch) {
      const auto frames = frame_queue.get_all_and_clear();
      if (frames.empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        continue;
      }
      for (const auto& frame : frames) {
        save_frame(frame);
      }
    }
    // Flush remaining
    const auto remaining = frame_queue.get_all_and_clear();
    for (const auto& frame : remaining) {
      save_frame(frame);
    }
  }

  void save_frame(const EstimationFrame::ConstPtr& frame) {
    const auto& raw_frame = frame->raw_frame;
    if (!raw_frame) {
      logger->warn("[HD] Frame {} has no raw_frame, skipping", frame->id);
      return;
    }

    // Source points: prefer raw (full resolution), fall back to preprocessed
    const std::vector<Eigen::Vector4d>* src_points = nullptr;
    const std::vector<double>* src_times = nullptr;
    const std::vector<double>* src_intensities = nullptr;
    const std::vector<uint32_t>* src_rings = nullptr;

    if (raw_frame->raw_points && !raw_frame->raw_points->points.empty()) {
      src_points = &raw_frame->raw_points->points;
      src_times = &raw_frame->raw_points->times;
      src_intensities = &raw_frame->raw_points->intensities;
      src_rings = &raw_frame->raw_points->rings;
    } else {
      src_points = &raw_frame->points;
      src_times = &raw_frame->times;
      src_intensities = raw_frame->intensities.empty() ? nullptr : &raw_frame->intensities;
      src_rings = nullptr;
      logger->debug("[HD] Frame {}: raw_points unavailable, using preprocessed ({} pts)", frame->id, src_points->size());
    }

    const int num_points = static_cast<int>(src_points->size());
    if (num_points == 0) {
      logger->warn("[HD] Frame {} has 0 points, skipping", frame->id);
      return;
    }

    // --- Deskew raw points using IMU-rate trajectory ---
    std::vector<Eigen::Vector4d> deskewed;
    if (frame->imu_rate_trajectory.cols() >= 2 && src_times && !src_times->empty()) {
      const int n_imu = frame->imu_rate_trajectory.cols();
      std::vector<double> imu_times(n_imu);
      std::vector<Eigen::Isometry3d> imu_poses(n_imu);
      for (int i = 0; i < n_imu; i++) {
        const auto& col = frame->imu_rate_trajectory.col(i);
        imu_times[i] = col[0];
        Eigen::Quaterniond q(col[7], col[4], col[5], col[6]);  // w, x, y, z
        Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
        pose.linear() = q.toRotationMatrix();
        pose.translation() = col.segment<3>(1);
        imu_poses[i] = pose;
      }

      CloudDeskewing deskewing;
      deskewed = deskewing.deskew(
        frame->T_lidar_imu, imu_times, imu_poses,
        raw_frame->stamp, *src_times, *src_points);
    } else {
      deskewed = *src_points;
      logger->debug("[HD] Frame {}: no IMU trajectory, saving undeskewed", frame->id);
    }

    // --- Compute normals (viewpoint-oriented, sensor at origin) ---
    std::vector<int> neighbors = hd_detail::find_knn(deskewed.data(), num_points, k_neighbors, num_threads);
    std::vector<Eigen::Vector4d> normals;
    std::vector<Eigen::Matrix4d> covs;
    CloudCovarianceEstimation cov_estimator(num_threads);
    cov_estimator.estimate(deskewed, neighbors, k_neighbors, normals, covs);

    // --- Prepare per-point attribute arrays ---
    std::vector<Eigen::Vector3f> points_f(num_points);
    std::vector<Eigen::Vector3f> normals_f(num_points);
    std::vector<float> intensities_f(num_points, 0.0f);
    std::vector<float> times_f(num_points, 0.0f);
    std::vector<float> range_f(num_points);

    for (int i = 0; i < num_points; i++) {
      points_f[i] = deskewed[i].head<3>().cast<float>();
      normals_f[i] = normals[i].head<3>().cast<float>();
      range_f[i] = static_cast<float>(deskewed[i].head<3>().norm());
    }
    if (src_intensities && src_intensities->size() == static_cast<size_t>(num_points)) {
      for (int i = 0; i < num_points; i++) intensities_f[i] = static_cast<float>((*src_intensities)[i]);
    }
    if (src_times && src_times->size() == static_cast<size_t>(num_points)) {
      for (int i = 0; i < num_points; i++) times_f[i] = static_cast<float>((*src_times)[i]);
    }

    std::vector<uint16_t> rings_u16;
    if (src_rings && src_rings->size() == static_cast<size_t>(num_points)) {
      rings_u16.resize(num_points);
      for (int i = 0; i < num_points; i++) rings_u16[i] = static_cast<uint16_t>((*src_rings)[i]);
    }

    // --- Compute world-frame bounding box from 8 local AABB corners ---
    Eigen::Vector3d local_min = deskewed[0].head<3>();
    Eigen::Vector3d local_max = deskewed[0].head<3>();
    for (int i = 1; i < num_points; i++) {
      const Eigen::Vector3d p = deskewed[i].head<3>();
      local_min = local_min.cwiseMin(p);
      local_max = local_max.cwiseMax(p);
    }

    const Eigen::Isometry3d& T = frame->T_world_lidar;
    Eigen::Vector3d world_min, world_max;
    for (int c = 0; c < 8; c++) {
      Eigen::Vector3d corner(
        (c & 1) ? local_max.x() : local_min.x(),
        (c & 2) ? local_max.y() : local_min.y(),
        (c & 4) ? local_max.z() : local_min.z());
      Eigen::Vector3d wc = T * corner;
      if (c == 0) { world_min = world_max = wc; }
      else { world_min = world_min.cwiseMin(wc); world_max = world_max.cwiseMax(wc); }
    }

    // --- Write files ---
    char dir_name[16];
    std::snprintf(dir_name, sizeof(dir_name), "%08ld", frame->id);
    const std::string frame_dir = output_path + "/" + dir_name;
    std::filesystem::create_directories(frame_dir);

    hd_detail::write_binary(frame_dir + "/points.bin", points_f.data(), sizeof(Eigen::Vector3f) * num_points);
    hd_detail::write_binary(frame_dir + "/normals.bin", normals_f.data(), sizeof(Eigen::Vector3f) * num_points);
    hd_detail::write_binary(frame_dir + "/intensities.bin", intensities_f.data(), sizeof(float) * num_points);
    hd_detail::write_binary(frame_dir + "/times.bin", times_f.data(), sizeof(float) * num_points);
    hd_detail::write_binary(frame_dir + "/range.bin", range_f.data(), sizeof(float) * num_points);
    if (!rings_u16.empty()) {
      hd_detail::write_binary(frame_dir + "/rings.bin", rings_u16.data(), sizeof(uint16_t) * num_points);
    }

    // --- frame_meta.json ---
    {
      std::ofstream ofs(frame_dir + "/frame_meta.json");
      ofs << std::setprecision(15) << std::fixed;
      ofs << "{\n";
      ofs << "  \"frame_id\": " << frame->id << ",\n";
      ofs << "  \"stamp\": " << raw_frame->stamp << ",\n";
      ofs << "  \"scan_end_time\": " << raw_frame->scan_end_time << ",\n";
      ofs << "  \"num_points\": " << num_points << ",\n";
      ofs << "  \"T_world_lidar\": [";
      for (int r = 0; r < 4; r++) {
        for (int cc = 0; cc < 4; cc++) {
          if (r > 0 || cc > 0) ofs << ", ";
          ofs << T.matrix()(r, cc);
        }
      }
      ofs << "],\n";
      ofs << "  \"bbox_world_min\": [" << world_min.x() << ", " << world_min.y() << ", " << world_min.z() << "],\n";
      ofs << "  \"bbox_world_max\": [" << world_max.x() << ", " << world_max.y() << ", " << world_max.z() << "]\n";
      ofs << "}\n";
    }

    logger->debug("[HD] Saved frame {} ({} points) to {}", frame->id, num_points, frame_dir);
  }

  std::atomic_bool kill_switch;
  std::thread thread;
  ConcurrentVector<EstimationFrame::ConstPtr> frame_queue;

  std::string output_path;
  int num_threads;
  int k_neighbors;
  std::shared_ptr<spdlog::logger> logger;
};

}  // namespace glim

extern "C" glim::ExtensionModule* create_extension_module() {
  return new glim::HDFrameSaver();
}
