#pragma once

#include <atomic>
#include <memory>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <glim/mapping/sub_map.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <glim/util/extension_module.hpp>
#include <glim/util/concurrent_vector.hpp>
#include <gtsam_points/util/gtsam_migration.hpp>

namespace spdlog {
class logger;
}

namespace gtsam {
class Values;
class NonlinearFactor;
class NonlinearFactorGraph;
}  // namespace gtsam

namespace gtsam_points {
class ISAM2Ext;
class ISAM2ResultExt;
}  // namespace gtsam_points

namespace guik {
class ModelControl;
}

namespace glim {

class TrajectoryManager;
class ManualLoopCloseModal;
class BundleAdjustmentModal;

class InteractiveViewer : public ExtensionModule {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  enum class PickType { POINTS = 1, FRAME = (1 << 1), FACTOR = (1 << 2) };
  enum class FactorType { MATCHING_COST, BETWEEN, IMU };

  InteractiveViewer();
  virtual ~InteractiveViewer();

  virtual bool ok() const override;
  void wait();
  void stop();
  void clear();

protected:
  void viewer_loop();

  virtual void setup_ui() {}

  void invoke(const std::function<void()>& task);
  void drawable_selection();
  void on_click();
  void context_menu();
  void run_modals();

  void update_viewer();

  void odometry_on_new_frame(const EstimationFrame::ConstPtr& new_frame);
  void globalmap_on_insert_submap(const SubMap::ConstPtr& submap);
  void globalmap_on_update_submaps(const std::vector<SubMap::Ptr>& updated_submaps);
  void globalmap_on_smoother_update(gtsam_points::ISAM2Ext& isam2, gtsam::NonlinearFactorGraph& new_factors, gtsam::Values& new_values);
  void globalmap_on_smoother_update_result(gtsam_points::ISAM2Ext& isam2, const gtsam_points::ISAM2ResultExt& result);

protected:
  std::atomic_bool request_to_clear;
  std::atomic_bool request_to_terminate;
  std::atomic_bool kill_switch;
  std::thread thread;

  // Process params
  int num_threads;

  // Tasks to be executed in the GUI thread
  std::mutex invoke_queue_mutex;
  std::vector<std::function<void()>> invoke_queue;

  // Visualization params
  int color_mode;
  std::vector<std::string> aux_attribute_names;
  Eigen::Vector2f aux_cmap_range;  // 1st–99th percentile range for the active aux attribute

  static constexpr size_t AUX_SAMPLE_CAP = 500000;
  std::unordered_map<std::string, std::vector<float>> aux_attr_samples;  // filtered (isfinite && >0) samples per attr
  std::unordered_map<std::string, Eigen::Vector2f> hd_attr_ranges;     // per-attribute min/max from HD tile uploads
  double gps_time_base;  // first gps_time seen; subtract before float cast to preserve float32 precision

  float coord_scale;
  float sphere_scale;

  bool draw_current;
  bool draw_traj;
  bool draw_points;
  bool draw_factors;
  bool draw_spheres;
  bool draw_coords;

  float min_overlap;
  bool cont_optimize;

  bool enable_partial_rendering;
  int partial_rendering_budget;

  double point_size;
  bool point_size_metric;
  bool point_shape_circle;
  bool show_display_settings = false;

  Eigen::Vector2f z_range;

  double points_alpha;
  double factors_alpha;

  std::atomic_bool needs_session_merge;

  // Click information
  Eigen::Vector4i right_clicked_info;
  Eigen::Vector3f right_clicked_pos;
  int lc_target_frame_id = -1;  // for HD loop closure callback
  int lc_source_frame_id = -1;
  std::vector<int> lc_source_group;  // accumulated loop-end submap indices
  std::vector<int> lc_target_group;  // target team submap indices (for auto-align)

  // GUI widgets
  std::unique_ptr<ManualLoopCloseModal> manual_loop_close_modal;
  std::unique_ptr<BundleAdjustmentModal> bundle_adjustment_modal;

  // Odometry
  std::unique_ptr<TrajectoryManager> trajectory;

  // Submaps
  std::vector<Eigen::Isometry3d> submap_poses;
  std::vector<SubMap::ConstPtr> submaps;
  std::vector<float> submap_gps_sigma;  // per-submap average GPS sigma (m), -1 if no GNSS factor
  std::unordered_map<int, float> pending_sigma_map;  // sigma values waiting for submap insertion
  std::unordered_set<int> hidden_sessions;  // session IDs to exclude from trajectory/factor rendering
  std::unordered_map<int, std::string> session_hd_paths;  // session_id → HD frames directory

  /// Load HD frames for a submap, return as PointCloudCPU.
  /// @param compute_covs If true, compute normals+covariances (needed for ICP). If false, points+intensity only (faster).
  gtsam_points::PointCloudCPU::Ptr load_hd_for_submap(int submap_index, bool compute_covs = true) const;

  // LOD memory management
  enum class SubmapLOD { UNLOADED = 0, BBOX = 1, LOADING = 2, SD = 3, LOADING_HD = 4, HD = 5 };
  struct SubmapRenderState {
    SubmapLOD current_lod = SubmapLOD::UNLOADED;
    Eigen::AlignedBox3f world_bbox;
    size_t gpu_bytes = 0;
    size_t hd_points = 0;   // HD points loaded for this submap (for counter tracking)
    bool bbox_computed = false;
  };
  std::vector<SubmapRenderState> render_states;
  size_t total_gpu_bytes = 0;

  // LOD settings (managed by Memory Manager UI)
  bool lod_enabled = true;         // distance-based streaming active
  bool lod_hd_enabled = false;     // LOD 0 (HD frames) enabled for nearby submaps
  bool lod_hd_only = false;        // hide SD submaps, show only HD (LOD 0)
  bool lod_hide_all_submaps = false;  // hide all submaps (for preview overlay mode)
  bool lod_load_full_sd = false;   // "Load full SD map" in progress
  bool lod_load_full_hd = false;   // "Load full HD map" in progress
  float lod_hd_range = 150.0f;    // metres — submaps within this distance get HD
  bool lod_show_bboxes = true;     // show wire cube bounding boxes
  float lod_sd_range = 300.0f;     // metres — submaps within this distance get SD
  float lod_vram_budget_mb = 4096.0f;
  bool show_memory_manager = false;
  bool lod_use_voxelized = false;     // load from hd_frames_voxelized/ instead of hd_frames/

  // Frustum culling
  static bool frustum_test_aabb(const Eigen::Matrix4f& vp, const Eigen::AlignedBox3f& box);

  // HD frame availability
  bool hd_available = false;
  std::string hd_frames_path;
  size_t total_hd_points = 0;    // from frame_meta.json scan (no data loading needed)
  size_t loaded_hd_points = 0;   // currently in GPU

  /// Scan hd_frames/ directory for availability and total point count.
  void detect_hd_frames(const std::string& map_path);

  // Async LOD worker
  struct LODWorkItem {
    int submap_index;
    SubMap::ConstPtr submap;
    Eigen::Isometry3d pose;
    int session_id;
    float distance;
    bool load_hd = false;
    std::string hd_path;  // per-session HD frames directory
  };
  void lod_worker_task();

  ConcurrentVector<LODWorkItem> lod_work_queue;
  std::thread lod_worker_thread;
  std::atomic_bool lod_worker_kill{false};

  // Source finder — cylinder/box probe to identify which submaps contribute points to an area
  bool source_finder_active = false;
  int source_finder_mode = 0;   // 0 = fast (bbox), 1 = precise (per-point)
  int source_finder_shape = 0;  // 0 = cylinder, 1 = box
  Eigen::Vector3f source_finder_pos = Eigen::Vector3f::Zero();
  float source_finder_radius = 0.5f;
  float source_finder_height = 5.0f;
  float source_finder_length = 20.0f;  // box: along yaw direction
  float source_finder_width = 2.0f;    // box: perpendicular
  float source_finder_yaw = 0.0f;      // box: Z rotation in degrees
  float source_finder_pitch = 0.0f;   // box: Y rotation in degrees (for slopes)
  std::unordered_set<int> source_finder_hits;
  bool source_finder_teams_swapped = false;
  std::unique_ptr<guik::ModelControl> source_finder_gizmo;
  void source_finder_scan_fast();
  void source_finder_scan_precise();
  void source_finder_color_hits();
  void source_finder_update_cylinder();

  // Undo last factor
  std::vector<size_t> last_factor_indices;  // ISAM2 graph indices of last added factors
  std::mutex last_factor_mutex;
  std::atomic_bool request_undo_last{false};

  // Factor relaxation (noise scaling around loop closure for smooth blending)
  struct RelaxationRequest {
    int center_key = -1;          // submap index around which to relax
    int radius = 5;               // number of submaps on each side
    float scale = 5.0f;           // noise sigma multiplier
    bool relax_between = true;    // relax BetweenFactors
    bool relax_gps = true;        // relax GNSS PoseTranslationPrior
  };
  ConcurrentVector<RelaxationRequest> pending_relaxations;

  // Factors
  std::vector<std::tuple<FactorType, std::uint64_t, std::uint64_t>> global_factors;

  // Factors to be inserted into the global mapping graph
  ConcurrentVector<gtsam_points::shared_ptr<gtsam::NonlinearFactor>> new_factors;

  // Logging
  std::shared_ptr<spdlog::logger> logger;
};
}  // namespace glim