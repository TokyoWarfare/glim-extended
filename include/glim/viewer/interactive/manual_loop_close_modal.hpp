#pragma once

#include <spdlog/spdlog.h>
#include <glim/mapping/sub_map.hpp>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam_points/types/point_cloud.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/ann/nearest_neighbor_search.hpp>
#include <glk/drawable.hpp>

namespace guik {
class GLCanvas;
class ModelControl;
class ProgressModal;
class ProgressInterface;
}  // namespace guik

namespace glim {

/**
 * @brief ImGUI modal for manually creating loop factors
 */
class ManualLoopCloseModal {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ManualLoopCloseModal(const std::shared_ptr<spdlog::logger>& logger, int num_threads);
  ~ManualLoopCloseModal();

  bool is_target_set() const;
  bool is_source_set() const;

  void set_target(const gtsam::Key target_key, const gtsam_points::PointCloud::ConstPtr& target, const Eigen::Isometry3d& target_pose);
  void set_source(const gtsam::Key source_key, const gtsam_points::PointCloud::ConstPtr& source, const Eigen::Isometry3d& source_pose);

  void set_submaps(const std::vector<SubMap::ConstPtr>& target_submaps, const std::vector<SubMap::ConstPtr>& source_submaps);

  /// Replace target/source with HD point clouds (with covariances already computed)
  void replace_with_hd(const gtsam_points::PointCloudCPU::Ptr& hd_target, const gtsam_points::PointCloudCPU::Ptr& hd_source);

  /// Callback to load HD data — set by the viewer
  std::function<std::pair<gtsam_points::PointCloudCPU::Ptr, gtsam_points::PointCloudCPU::Ptr>()> load_hd_callback;

  void clear();

  gtsam::NonlinearFactor::shared_ptr run();

private:
  std::pair<gtsam_points::PointCloudCPU::Ptr, gtsam_points::PointCloudCPU::Ptr> preprocess_maps(guik::ProgressInterface& progress);

  std::shared_ptr<Eigen::Isometry3d> align_global(guik::ProgressInterface& progress);

  std::shared_ptr<Eigen::Isometry3d> align(guik::ProgressInterface& progress);
  gtsam::NonlinearFactor::shared_ptr create_factor();
  void ensure_covs(gtsam_points::PointCloudCPU::Ptr& cloud, const std::string& label);
  void draw_canvas();

  bool show_note(const std::string& note);

private:
  const int num_threads;

  bool request_to_open;
  bool helper_gizmo_active = false;
  std::unique_ptr<guik::ModelControl> helper_model_control;
  Eigen::Matrix4f helper_prev_matrix = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f initial_model_matrix = Eigen::Matrix4f::Identity();  // for reset
  std::unique_ptr<guik::GLCanvas> canvas;
  std::unique_ptr<guik::ProgressModal> progress_modal;
  std::unique_ptr<guik::ModelControl> model_control;

  int seed;

  // Map preprocess params
  float min_distance;

  // Global registration params
  float fpfh_radius;
  int global_registration_type;
  bool global_registration_4dof;

  int ransac_max_iterations;
  float ransac_early_stop_rate;
  float ransac_inlier_voxel_resolution;

  int gnc_max_samples;

  // Scan matching and loop factor params
  float information_scale;
  float max_correspondence_distance;

  gtsam::Key target_key;
  gtsam::Key source_key;

public:
  float target_gps_sigma = -1.0f;  // GPS sigma for target submap (-1 = N/A)
  float source_gps_sigma = -1.0f;  // GPS sigma for source submap (-1 = N/A)

  // Neighbor relaxation params (exposed for viewer to read after factor creation)
  bool modal_intensity_mode = false;  // toggle between red/green and intensity rendering
  bool relax_neighbors = false;
  int relax_radius = 5;
  float relax_scale = 5.0f;
  bool relax_between = true;
  bool relax_gps = true;
private:
  Eigen::Isometry3d target_pose;  // Target pose in world frame
  Eigen::Isometry3d source_pose;  // Source pose in world frame

  gtsam_points::PointCloudCPU::Ptr target;  // Gravity aligned target point cloud
  gtsam_points::PointCloudCPU::Ptr source;  // Gravity aligned source point cloud
  gtsam_points::PointCloudCPU::Ptr sd_target;  // Original SD data (for reverting from HD)
  gtsam_points::PointCloudCPU::Ptr sd_source;

  glk::Drawable::ConstPtr target_drawable;  // Gravity aligned target drawable
  glk::Drawable::ConstPtr source_drawable;  // Gravity aligned source drawable
  glk::Drawable::ConstPtr modal_target_intensity_drawable;  // Intensity-colored target
  glk::Drawable::ConstPtr modal_source_intensity_drawable;  // Intensity-colored source

  gtsam_points::NearestNeighborSearch::Ptr target_fpfh_tree;
  gtsam_points::NearestNeighborSearch::Ptr source_fpfh_tree;

  std::vector<SubMap::ConstPtr> target_submaps;
  std::vector<SubMap::ConstPtr> source_submaps;

  std::shared_ptr<void> tbb_task_arena;

  // Logging
  std::shared_ptr<spdlog::logger> logger;
};

}  // namespace glim